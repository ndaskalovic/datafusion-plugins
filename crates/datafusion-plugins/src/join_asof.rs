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
use std::sync::Arc;

use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use datafusion::common::config::ConfigOptions;
use datafusion::common::{Result, plan_datafusion_err, plan_err};
use datafusion::physical_expr::expressions::Column;
use datafusion::physical_expr::{PhysicalExpr, PhysicalExprRef};
use datafusion::physical_plan::projection::ProjectionExec;
use datafusion::physical_plan::{ExecutionPlan, ExecutionPlanProperties};

use crate::exec::{AsofOptions, StreamingAsOfJoinExec};
use crate::keys::{RowGroup, is_primitive_group_type, is_supported_ts_type};
use crate::ordered::OrderedConcatExec;
use crate::plan_bytes::PlanNode;
use crate::scan::{
    merge, nulls_first_of, pin_order, range_partitioned, resplit, satisfies, single_file_scan,
    sort_exprs,
};

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

impl PlanNode for JoinAsof {
    fn schema(&self, inputs: &[SchemaRef]) -> Result<SchemaRef> {
        let [l, r] = two(inputs)?;
        self.validate(l, r)?;
        self.output_schema(l, r)
    }

    fn required_columns(
        &self,
        inputs: &[SchemaRef],
        projection: &[usize],
    ) -> Result<Vec<Option<Vec<String>>>> {
        let [l, r] = two(inputs)?;
        let outputs = self.output_columns(l, r)?;
        let keep = |schema: &Schema, side: Side, keys: Vec<&String>| -> Vec<String> {
            schema
                .fields()
                .iter()
                .map(|f| f.name().clone())
                .filter(|n| {
                    keys.contains(&n)
                        || projection.iter().any(|&i| outputs[i].side == side && &outputs[i].column == n)
                })
                .collect()
        };
        let lk = std::iter::once(&self.left_on).chain(&self.by_left).collect();
        let rk = std::iter::once(&self.right_on).chain(&self.by_right).collect();
        Ok(vec![Some(keep(l, Side::Left, lk)), Some(keep(r, Side::Right, rk))])
    }

    fn build(
        &self,
        schemas: &[SchemaRef],
        inputs: Vec<Arc<dyn ExecutionPlan>>,
        projection: &[usize],
        config: &ConfigOptions,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let [l, r] = two(schemas)?;
        let all = self.output_columns(l, r)?;
        let outputs: Vec<OutputColumn> = projection.iter().map(|i| all[*i].clone()).collect();
        let [left, right]: [Arc<dyn ExecutionPlan>; 2] = inputs
            .try_into()
            .map_err(|_| plan_datafusion_err!("join_asof takes two inputs"))?;
        self.build_with_outputs(left, right, &outputs, config)
    }
}

/// The two schemas of a two-input node.
pub(crate) fn two(inputs: &[SchemaRef]) -> Result<[&Schema; 2]> {
    match inputs {
        [l, r] => Ok([l.as_ref(), r.as_ref()]),
        _ => plan_err!("expected two inputs, got {}", inputs.len()),
    }
}
