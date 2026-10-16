//! Building a `join_asof` plan from two physical inputs, with polars
//! `join_asof(strategy="backward")` semantics and contract.
//!
//! Contract: with `by`, each input must be sorted by `(by..., on)`; without,
//! by `on`. Nothing is sorted for you (as in polars: sort it yourself).
//!
//! [`JoinAsof::build`]:
//!
//! 1. **Order.** An input whose order DataFusion can prove (a declared file
//!    order surviving filters and projections) runs with only the O(1)
//!    batch- and partition-boundary checks. Otherwise, with
//!    `check_sortedness` (the default), every batch is validated as it
//!    streams, and an unsorted input fails with "left/right input is not
//!    sorted by (...)"; without it, the data is trusted.
//! 2. **Fast mode** when both inputs are re-executable scans of one Parquet
//!    file each (under filters and projections) with a proven order: the
//!    scans are split into ordered byte ranges (right x32, left x4 of
//!    `target_partitions`), each left range seeks its right start by binary
//!    search, and [`OrderedConcatExec`] emits one ordered partition.
//! 3. **Execute-once mode** otherwise: every input partition is executed
//!    exactly once (see the `exec` module).
//! 4. A final projection gives the output polars' columns and names.

use std::collections::HashSet;
use std::fmt;
use std::sync::{Arc, Mutex};

use arrow::compute::SortOptions;
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use datafusion::common::config::ConfigOptions;
use datafusion::common::tree_node::TreeNodeRecursion;
use datafusion::common::{JoinType, Result, plan_datafusion_err, plan_err};
use datafusion::datasource::physical_plan::{FileGroup, FileScanConfigBuilder, ParquetSource};
use datafusion::datasource::source::DataSourceExec;
use datafusion::execution::TaskContext;
use datafusion::physical_expr::expressions::Column;
use datafusion::physical_expr::{
    EquivalenceProperties, LexOrdering, PhysicalExpr, PhysicalExprRef, PhysicalSortExpr,
};
use datafusion::physical_plan::coop::CooperativeExec;
use datafusion::physical_plan::filter::FilterExec;
use datafusion::physical_plan::joins::{HashJoinExec, PartitionMode};
use datafusion::physical_plan::projection::ProjectionExec;
use datafusion::physical_plan::sorts::sort_preserving_merge::SortPreservingMergeExec;
use datafusion::physical_plan::{
    ChildrenPropertiesMode, DisplayAs, DisplayFormatType, ExecutionPlan, ExecutionPlanProperties,
    PlanProperties, RecordBatchStream, ReplaceChildrenOptions, SendableRecordBatchStream,
    displayable,
};
use futures::Stream;

use crate::exec::{AsofOptions, StreamingAsOfJoinExec};
use crate::keys::{RowGroup, is_primitive_group_type, is_supported_ts_type};
use crate::ordered::OrderedConcatExec;

/// Which input an output column comes from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Side {
    Left,
    Right,
}

/// One output column: `column` of `side`, named `name`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OutputColumn {
    pub side: Side,
    pub column: String,
    pub name: String,
}

/// A backward ASOF join, as polars `join_asof(strategy="backward")`: for
/// each left row, the last right row with equal `by` values and
/// `right_on <= left_on` (`<` without `allow_exact_matches`). One output row
/// per left row, in left order.
#[derive(Debug, Clone)]
pub struct JoinAsof {
    pub left_on: String,
    pub right_on: String,
    pub by_left: Vec<String>,
    pub by_right: Vec<String>,
    pub allow_exact_matches: bool,
    pub suffix: String,
    pub check_sortedness: bool,
    pub options: AsofOptions,
}

impl JoinAsof {
    /// Join on column `on` of both inputs.
    pub fn new(on: impl Into<String>) -> Self {
        let on = on.into();
        Self::new_with(on.clone(), on)
    }

    /// Join on `left_on` of the left input and `right_on` of the right.
    pub fn new_with(left_on: impl Into<String>, right_on: impl Into<String>) -> Self {
        Self {
            left_on: left_on.into(),
            right_on: right_on.into(),
            by_left: Vec::new(),
            by_right: Vec::new(),
            allow_exact_matches: true,
            suffix: "_right".to_string(),
            check_sortedness: true,
            options: AsofOptions::default(),
        }
    }

    /// Equality columns, the same names on both sides.
    pub fn by<I: IntoIterator<Item = S>, S: Into<String>>(mut self, by: I) -> Self {
        self.by_left = by.into_iter().map(Into::into).collect();
        self.by_right = self.by_left.clone();
        self
    }

    /// Equality columns, named per side.
    pub fn by_left_right<I, J, S, T>(mut self, left: I, right: J) -> Self
    where
        I: IntoIterator<Item = S>,
        J: IntoIterator<Item = T>,
        S: Into<String>,
        T: Into<String>,
    {
        self.by_left = left.into_iter().map(Into::into).collect();
        self.by_right = right.into_iter().map(Into::into).collect();
        self
    }

    pub fn allow_exact_matches(mut self, allow: bool) -> Self {
        self.allow_exact_matches = allow;
        self
    }

    pub fn suffix(mut self, suffix: impl Into<String>) -> Self {
        self.suffix = suffix.into();
        self
    }

    pub fn check_sortedness(mut self, check: bool) -> Self {
        self.check_sortedness = check;
        self
    }

    pub fn options(mut self, options: AsofOptions) -> Self {
        self.options = options;
        self
    }

    /// Check the key columns of both schemas: present, same types, supported.
    pub fn validate(&self, left: &Schema, right: &Schema) -> Result<()> {
        if self.by_left.len() != self.by_right.len() {
            return plan_err!("join_asof: by_left and by_right must have the same length");
        }
        let find = |s: &Schema, name: &str, which: &str| -> Result<DataType> {
            s.field_with_name(name).map(|f| f.data_type().clone()).map_err(|_| {
                plan_datafusion_err!(
                    "join_asof: column '{name}' not found in the {which} input; columns: [{}]",
                    s.fields().iter().map(|f| f.name().as_str()).collect::<Vec<_>>().join(", ")
                )
            })
        };
        for (s, which) in [(left, "left"), (right, "right")] {
            let mut seen = HashSet::new();
            for f in s.fields() {
                if !seen.insert(f.name()) {
                    return plan_err!(
                        "join_asof: the {which} input has two columns named '{}'; \
                         rename one first",
                        f.name()
                    );
                }
            }
        }
        let lt = find(left, &self.left_on, "left")?;
        let rt = find(right, &self.right_on, "right")?;
        if lt != rt {
            return plan_err!(
                "join_asof: 'on' columns have different types: {lt} (left) and {rt} (right)"
            );
        }
        if !is_supported_ts_type(&lt) {
            return plan_err!(
                "join_asof: unsupported 'on' type {lt} (integers, floats, dates, times, \
                 timestamps and durations are supported)"
            );
        }
        let mut by_types = Vec::new();
        for (bl, br) in self.by_left.iter().zip(&self.by_right) {
            let (l, r) = (find(left, bl, "left")?, find(right, br, "right")?);
            if l != r {
                return plan_err!(
                    "join_asof: 'by' columns '{bl}' and '{br}' have different types: {l} and {r}"
                );
            }
            by_types.push(l);
        }
        let ok = match by_types.as_slice() {
            [] => true,
            [t] if is_primitive_group_type(t) => true,
            ts => !ts.iter().any(|t| t.is_floating()) && RowGroup::supports(ts),
        };
        if !ok {
            return plan_err!(
                "join_asof: unsupported 'by' column types {by_types:?} (float keys are not supported)"
            );
        }
        Ok(())
    }

    /// polars' output: every left column, then every right column in order,
    /// except the `by` keys and the `on` column when it has the left `on`
    /// name. A right name that is already taken gets `suffix`; if that is
    /// taken too, it is an error (as in polars).
    pub fn output_columns(&self, left: &Schema, right: &Schema) -> Result<Vec<OutputColumn>> {
        let mut out: Vec<OutputColumn> = left
            .fields()
            .iter()
            .map(|f| OutputColumn {
                side: Side::Left,
                column: f.name().clone(),
                name: f.name().clone(),
            })
            .collect();
        for f in right.fields() {
            let n = f.name();
            if self.by_right.contains(n) || (n == &self.right_on && self.right_on == self.left_on) {
                continue;
            }
            let mut name = n.clone();
            if out.iter().any(|c| c.name == name) {
                name = format!("{n}{}", self.suffix);
                if out.iter().any(|c| c.name == name) {
                    return plan_err!(
                        "join_asof: column with name '{name}' already exists; rename the \
                         column or use a different suffix"
                    );
                }
            }
            out.push(OutputColumn {
                side: Side::Right,
                column: n.clone(),
                name,
            });
        }
        Ok(out)
    }

    /// Schema of [`Self::output_columns`]: right columns are nullable.
    pub fn output_schema(&self, left: &Schema, right: &Schema) -> Result<SchemaRef> {
        output_schema(&self.output_columns(left, right)?, left, right)
    }

    /// Build the join with polars' output columns.
    pub fn build(
        &self,
        left: Arc<dyn ExecutionPlan>,
        right: Arc<dyn ExecutionPlan>,
        config: &ConfigOptions,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let outputs = self.output_columns(&left.schema(), &right.schema())?;
        self.build_with_outputs(left, right, &outputs, config)
    }

    /// Build the join with an explicit output column list.
    pub fn build_with_outputs(
        &self,
        left: Arc<dyn ExecutionPlan>,
        right: Arc<dyn ExecutionPlan>,
        outputs: &[OutputColumn],
        config: &ConfigOptions,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let (ls, rs) = (left.schema(), right.schema());
        self.validate(&ls, &rs)?;
        let mut options = self.options.clone();
        let tp = config.execution.target_partitions.max(1);

        let col = |s: &Schema, name: &str| -> Result<PhysicalExprRef> {
            Ok(Arc::new(Column::new(name, s.index_of(name)?)) as PhysicalExprRef)
        };
        let left_ts = col(&ls, &self.left_on)?;
        let right_ts = col(&rs, &self.right_on)?;
        let lby = self.by_left.iter().map(|b| col(&ls, b)).collect::<Result<Vec<_>>>()?;
        let rby = self.by_right.iter().map(|b| col(&rs, b)).collect::<Result<Vec<_>>>()?;
        let mut kl = lby.clone();
        kl.push(Arc::clone(&left_ts));
        let mut kr = rby.clone();
        kr.push(Arc::clone(&right_ts));

        // 1. Proven orders; the others are validated while streaming (or
        // trusted). Never sorted.
        let lnf = nulls_first_of(left.equivalence_properties(), &kl)?;
        let rnf = nulls_first_of(right.equivalence_properties(), &kr)?;
        let (left_proven, right_proven) = (lnf.is_some(), rnf.is_some());
        let (lnf, rnf) = (lnf.unwrap_or(true), rnf.unwrap_or(true));
        options.validate_left |= !left_proven && self.check_sortedness;
        options.validate_right |= !right_proven && self.check_sortedness;
        let lsort = sort_exprs(&kl, lnf);
        let rsort = sort_exprs(&kr, rnf);

        // 2. Fast mode: re-executable single-file scans with a proven order.
        let fast = left_proven
            && right_proven
            && config.optimizer.repartition_file_scans
            && options.ordered_output
            && single_file_scan(&left)
            && single_file_scan(&right);
        let (left, right, right_units) = if fast {
            let right = match resplit(&right, tp * options.right_split_factor.max(1), config)? {
                Some(r) if satisfies(&r, &rsort)? => r,
                _ => right,
            };
            let left = match resplit(&left, tp * options.left_split_factor.max(1), config)? {
                Some(l) if satisfies(&l, &lsort)? => l,
                _ => left,
            };
            (left, right, None)
        } else {
            // 3. Execute-once. Range-partitioned inputs (consecutive key
            // ranges: a split scan, maybe under a broadcast join) are read in
            // partition order; the right side's ranges become units when
            // their order is proven. Other multi-partition inputs are merged
            // (their partitions each sorted, proven or validated).
            let left = if range_partitioned(&left) { pin_order(left)? } else { merge(left, &lsort)? };
            let (right, units) = if range_partitioned(&right) {
                let right = pin_order(right)?;
                if right_proven {
                    let r = match resplit(&right, tp * options.left_split_factor.max(1), config)? {
                        Some(r) if satisfies(&r, &rsort)? => r,
                        _ => right,
                    };
                    (r, true)
                } else {
                    (right, false)
                }
            } else {
                (merge(right, &rsort)?, false)
            };
            (left, right, Some(units))
        };

        // The join outputs every left column (the ordered output checks the
        // order on them) and only the right columns that are used.
        let nl = ls.fields().len();
        let mut embedded: Vec<usize> = (0..nl).collect();
        let mut right_used: Vec<usize> = Vec::new();
        for o in outputs.iter().filter(|o| o.side == Side::Right) {
            let j = rs.index_of(&o.column)?;
            if !right_used.contains(&j) {
                right_used.push(j);
                embedded.push(nl + j);
            }
        }
        let on: Vec<(PhysicalExprRef, PhysicalExprRef)> = lby.into_iter().zip(rby).collect();
        let exec = StreamingAsOfJoinExec::try_new(
            left,
            right,
            on,
            left_ts,
            right_ts,
            !self.allow_exact_matches,
            lnf,
            rnf,
            Some(embedded),
            options.clone(),
        )?;
        let exec = match right_units {
            Some(units) => exec.with_execute_once(units),
            None => exec,
        };
        let exec: Arc<dyn ExecutionPlan> = Arc::new(exec);
        let joined: Arc<dyn ExecutionPlan> = if exec.output_partitioning().partition_count() > 1 {
            let window = tp + options.window_extra;
            Arc::new(OrderedConcatExec::try_new(Arc::clone(&exec), window)?)
        } else {
            exec
        };

        // 4. Output column order and names.
        let js = joined.schema();
        let mut exprs: Vec<(Arc<dyn PhysicalExpr>, String)> = Vec::with_capacity(outputs.len());
        for o in outputs {
            let pos = match o.side {
                Side::Left => ls.index_of(&o.column)?,
                Side::Right => {
                    let j = rs.index_of(&o.column)?;
                    nl + right_used.iter().position(|x| *x == j).expect("embedded")
                }
            };
            exprs.push((Arc::new(Column::new(js.field(pos).name(), pos)), o.name.clone()));
        }
        Ok(Arc::new(ProjectionExec::try_new(exprs, joined)?))
    }
}

/// Arrow schema of `outputs` over the two input schemas.
pub fn output_schema(outputs: &[OutputColumn], left: &Schema, right: &Schema) -> Result<SchemaRef> {
    let fields = outputs
        .iter()
        .map(|o| {
            Ok(match o.side {
                Side::Left => left.field_with_name(&o.column)?.clone().with_name(o.name.clone()),
                Side::Right => right
                    .field_with_name(&o.column)?
                    .clone()
                    .with_name(o.name.clone())
                    .with_nullable(true),
            })
        })
        .collect::<Result<Vec<Field>>>()?;
    Ok(Arc::new(Schema::new(fields)))
}

fn sort_exprs(exprs: &[PhysicalExprRef], nulls_first: bool) -> Vec<PhysicalSortExpr> {
    exprs
        .iter()
        .map(|e| PhysicalSortExpr {
            expr: Arc::clone(e),
            options: SortOptions {
                descending: false,
                nulls_first,
            },
        })
        .collect()
}

/// `Some(nulls_first)` if `eq` satisfies an ascending order on `exprs`,
/// trying NULLS FIRST then NULLS LAST (null keys never match, so either
/// placement works).
fn nulls_first_of(eq: &EquivalenceProperties, exprs: &[PhysicalExprRef]) -> Result<Option<bool>> {
    for nf in [true, false] {
        if eq.ordering_satisfy(sort_exprs(exprs, nf))? {
            return Ok(Some(nf));
        }
    }
    Ok(None)
}

fn satisfies(plan: &Arc<dyn ExecutionPlan>, sort: &[PhysicalSortExpr]) -> Result<bool> {
    plan.equivalence_properties().ordering_satisfy(sort.to_vec())
}

/// Merge the (each sorted) partitions of `plan` into one ordered partition.
fn merge(plan: Arc<dyn ExecutionPlan>, sort: &[PhysicalSortExpr]) -> Result<Arc<dyn ExecutionPlan>> {
    if plan.output_partitioning().partition_count() <= 1 {
        return Ok(plan);
    }
    let lex = LexOrdering::new(sort.to_vec())
        .ok_or_else(|| plan_datafusion_err!("join_asof: empty sort key"))?;
    Ok(Arc::new(SortPreservingMergeExec::new(lex, plan)))
}

/// Nodes that keep their input's partitions and row order.
fn passthrough(plan: &Arc<dyn ExecutionPlan>) -> Option<&Arc<dyn ExecutionPlan>> {
    if plan.is::<FilterExec>()
        || plan.is::<ProjectionExec>()
        || plan.is::<CooperativeExec>()
    {
        plan.children().into_iter().next()
    } else {
        None
    }
}

/// The distinct files of a Parquet `DataSourceExec`, if `plan` is one.
fn parquet_files(plan: &Arc<dyn ExecutionPlan>) -> Option<Vec<String>> {
    let dse = plan.downcast_ref::<DataSourceExec>()?;
    let (conf, _) = dse.downcast_to_file_source::<ParquetSource>()?;
    let mut files: Vec<String> = Vec::new();
    for g in &conf.file_groups {
        for f in g.iter() {
            let p = f.object_meta.location.to_string();
            if !files.contains(&p) {
                files.push(p);
            }
        }
    }
    Some(files)
}

/// A re-executable scan of exactly one Parquet file, possibly under filters
/// and projections. Its byte-range partitions are consecutive key ranges.
fn single_file_scan(plan: &Arc<dyn ExecutionPlan>) -> bool {
    match passthrough(plan) {
        Some(child) => single_file_scan(child),
        None => parquet_files(plan).is_some_and(|f| f.len() == 1),
    }
}

/// Partitions are consecutive key ranges: a single-file scan, through
/// order-preserving nodes and the probe side of a `CollectLeft` inner join.
fn range_partitioned(plan: &Arc<dyn ExecutionPlan>) -> bool {
    if let Some(child) = passthrough(plan) {
        return range_partitioned(child);
    }
    if let Some(hj) = plan.downcast_ref::<HashJoinExec>()
        && *hj.partition_mode() == PartitionMode::CollectLeft
        && *hj.join_type() == JoinType::Inner
        && hj.maintains_input_order().get(1).copied().unwrap_or(false)
    {
        return range_partitioned(hj.right());
    }
    single_file_scan(plan)
}

/// Turn off file-stream work stealing in the scan under a range-partitioned
/// plan (DataFusion 55 lets sibling partitions of an unordered scan steal
/// each other's byte ranges, so a partition would no longer be one range).
fn pin_order(plan: Arc<dyn ExecutionPlan>) -> Result<Arc<dyn ExecutionPlan>> {
    let opts = || ReplaceChildrenOptions::new(ChildrenPropertiesMode::Recompute);
    if let Some(child) = passthrough(&plan) {
        let child = pin_order(Arc::clone(child))?;
        return Arc::clone(&plan).replace_children(vec![child], opts());
    }
    if let Some(hj) = plan.downcast_ref::<HashJoinExec>() {
        let (l, r) = (Arc::clone(hj.left()), pin_order(Arc::clone(hj.right()))?);
        return Arc::clone(&plan).replace_children(vec![l, r], opts());
    }
    if plan.is::<DataSourceExec>() {
        return Ok(plan.with_preserve_order(true).unwrap_or(plan));
    }
    Ok(plan)
}

/// Re-split the single-file Parquet scan under `plan` (through filters and
/// projections) into consecutive byte ranges of about `size / target` bytes.
/// For one file this is DataFusion's own `repartition_evenly_by_size`.
/// Returns `None` if `plan` is not such a scan or no split applies.
fn resplit(
    plan: &Arc<dyn ExecutionPlan>,
    target: usize,
    config: &ConfigOptions,
) -> Result<Option<Arc<dyn ExecutionPlan>>> {
    if let Some(child) = passthrough(plan) {
        let Some(new_child) = resplit(child, target, config)? else {
            return Ok(None);
        };
        let node = Arc::clone(plan).replace_children(
            vec![new_child],
            ReplaceChildrenOptions::new(ChildrenPropertiesMode::Recompute),
        )?;
        return Ok(Some(node));
    }
    let Some(dse) = plan.downcast_ref::<DataSourceExec>() else {
        return Ok(None);
    };
    let Some((conf, _)) = dse.downcast_to_file_source::<ParquetSource>() else {
        return Ok(None);
    };
    let Some(file) = conf.file_groups.iter().flat_map(|g| g.iter()).next() else {
        return Ok(None);
    };
    if parquet_files(plan).is_none_or(|f| f.len() != 1) {
        return Ok(None);
    }
    let size = file.object_meta.size;
    if !config.optimizer.repartition_file_scans
        || target <= 1
        || size < config.optimizer.repartition_file_min_size as u64
        || conf.file_compression_type.is_compressed()
    {
        return Ok(None);
    }
    let mut template = file.clone();
    template.range = None;
    let chunk = size.div_ceil(target as u64).max(1);
    let mut groups = Vec::new();
    let mut start = 0u64;
    while start < size {
        let end = size.min(start + chunk);
        groups.push(FileGroup::new(vec![template.clone().with_range(start as i64, end as i64)]));
        start = end;
    }
    if groups.len() <= conf.file_groups.len() {
        return Ok(None);
    }
    let new_conf = FileScanConfigBuilder::from(conf.clone())
        .with_file_groups(groups)
        .build();
    Ok(Some(DataSourceExec::from_data_source(new_conf) as Arc<dyn ExecutionPlan>))
}

static LAST_PLAN: Mutex<Option<String>> = Mutex::new(None);

/// The physical plan of the most recent `asof_join` scan in this process
/// (indented), for debugging and tests.
pub fn last_plan() -> Option<String> {
    LAST_PLAN.lock().ok().and_then(|g| g.clone())
}

pub(crate) fn set_last_plan(plan: &Arc<dyn ExecutionPlan>) {
    let s = displayable(plan.as_ref()).indent(true).to_string();
    if let Ok(mut g) = LAST_PLAN.lock() {
        *g = Some(s);
    }
}

/// Leaf node holding a finished join plan.
///
/// A plan handed to another session (over FFI, or through a table provider)
/// is optimized again by that session, and DataFusion's distribution and
/// sorting rules would see the scans below the join and repartition or merge
/// them. As a leaf with the inner plan's properties (one ordered partition)
/// it is left alone; the inner plan is already in final form.
///
/// If `runtime` is set, `execute` and every poll of the output stream run
/// inside that tokio runtime (needed when the caller lives in another shared
/// library with its own copy of tokio).
pub struct AsofJoinScanExec {
    inner: Arc<dyn ExecutionPlan>,
    runtime: Option<tokio::runtime::Handle>,
    cache: Arc<PlanProperties>,
}

impl fmt::Debug for AsofJoinScanExec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AsofJoinScanExec").field("inner", &self.inner.name()).finish()
    }
}

impl AsofJoinScanExec {
    pub fn new(inner: Arc<dyn ExecutionPlan>, runtime: Option<tokio::runtime::Handle>) -> Self {
        let p = inner.properties();
        let cache = PlanProperties::new(
            EquivalenceProperties::clone(p.equivalence_properties()),
            p.output_partitioning().clone(),
            p.emission_type,
            p.boundedness,
        );
        Self {
            inner,
            runtime,
            cache: Arc::new(cache),
        }
    }

    pub fn inner(&self) -> &Arc<dyn ExecutionPlan> {
        &self.inner
    }
}

impl DisplayAs for AsofJoinScanExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut fmt::Formatter) -> fmt::Result {
        match t {
            DisplayFormatType::TreeRender => write!(f, "AsofJoinScanExec"),
            _ => {
                let inner = displayable(self.inner.as_ref()).indent(false).to_string();
                let inner = inner.trim_end().replace('\n', " <- ");
                write!(f, "AsofJoinScanExec: [{inner}]")
            }
        }
    }
}

impl ExecutionPlan for AsofJoinScanExec {
    fn name(&self) -> &str {
        "AsofJoinScanExec"
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.cache
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![]
    }

    fn apply_expressions(
        &self,
        _f: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> Result<TreeNodeRecursion>,
    ) -> Result<TreeNodeRecursion> {
        Ok(TreeNodeRecursion::Continue)
    }

    fn replace_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
        _options: ReplaceChildrenOptions,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        if !children.is_empty() {
            return plan_err!("AsofJoinScanExec has no children");
        }
        Ok(self)
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        if !children.is_empty() {
            return plan_err!("AsofJoinScanExec has no children");
        }
        Ok(self)
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        match &self.runtime {
            None => self.inner.execute(partition, context),
            Some(rt) => {
                let _g = rt.enter();
                let inner = self.inner.execute(partition, context)?;
                Ok(Box::pin(InRuntime {
                    inner: Some(inner),
                    rt: rt.clone(),
                }))
            }
        }
    }

    fn metrics(&self) -> Option<datafusion::physical_plan::metrics::MetricsSet> {
        self.inner.metrics()
    }
}

/// Polls (and drops) `inner` inside `rt`.
struct InRuntime {
    inner: Option<SendableRecordBatchStream>,
    rt: tokio::runtime::Handle,
}

impl Stream for InRuntime {
    type Item = Result<arrow::array::RecordBatch>;

    fn poll_next(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        let _g = self.rt.enter();
        match self.inner.as_mut() {
            Some(s) => s.as_mut().poll_next(cx),
            None => std::task::Poll::Ready(None),
        }
    }
}

impl RecordBatchStream for InRuntime {
    fn schema(&self) -> SchemaRef {
        self.inner
            .as_ref()
            .map(|s| s.schema())
            .unwrap_or_else(|| Arc::new(Schema::empty()))
    }
}

impl Drop for InRuntime {
    fn drop(&mut self) {
        let _g = self.rt.enter();
        self.inner = None;
    }
}
