//! `join_asof_markouts`: backward ASOF joins at several time offsets
//! ("markouts") in one pass over the right input.
//!
//! For each left row and each markout `h` (seconds): the last right row of the
//! same `by` group with `right.on <= left.on + h` (`<` without exact
//! matches), i.e. polars `left.with_columns(on + h).join_asof(right, ...)`
//! for every `h`, side by side.
//!
//! # Algorithm
//!
//! The range-parallel layout of [`crate::StreamingAsOfJoinExec`] (left split
//! x4, right split x32, a prologue that probes every right range's first
//! key, `OrderedConcatExec` on top), with N cursors per worker instead of
//! one. The cursors walk ONE shared window of right batches, each galloping
//! on its own shifted threshold `pack(group, ts + h)`:
//!
//! * a worker seeks the right range holding the match of its first left row
//!   at the smallest offset, and streams forward as the largest offset needs;
//! * right batches stay in the window from the slowest cursor (smallest
//!   `h`) to the fastest (largest `h`), then retire; retired batches are
//!   compacted to the rows the candidates still reference;
//! * left rows are answered for all offsets, a batch at a time, and emitted.
//!
//! Memory per worker: one left batch, plus the right rows between
//! `left.on + h_min` and `left.on + h_max` (the window), plus O(groups)
//! compacted candidates. A large `h_max - h_min` relative to the right
//! side's density means a large window.
//!
//! Inputs are assumed sorted by `(by..., on)`; only batch and partition
//! boundaries are checked (they guard this operator's own assumption that
//! byte-range partitions are consecutive key ranges).

use std::any::Any;
use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;
use std::sync::atomic::Ordering as AtomicOrdering;
use std::sync::{Arc, Mutex};

use arrow::array::{Array, ArrayRef, RecordBatch, RecordBatchOptions, UInt32Array, new_null_array};
use arrow::compute::{SortOptions, interleave, take};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef, TimeUnit};
use datafusion::common::config::ConfigOptions;
use datafusion::common::tree_node::TreeNodeRecursion;
use datafusion::common::{Result, exec_err, internal_datafusion_err, plan_datafusion_err, plan_err};
use datafusion::execution::TaskContext;
use datafusion::physical_expr::expressions::Column;
use datafusion::physical_expr::{
    EquivalenceProperties, LexOrdering, PhysicalExpr, PhysicalExprRef, PhysicalSortExpr,
};
use datafusion::physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion::physical_plan::projection::ProjectionExec;
use datafusion::physical_plan::stream::RecordBatchReceiverStreamBuilder;
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, ExecutionPlanProperties, Partitioning,
    PlanProperties, ReplaceChildrenOptions, SendableRecordBatchStream, apply_expression_roots,
};
use futures::StreamExt;
use tokio::sync::mpsc::Sender;

use crate::exec::{
    AsofOptions, JoinSpec, Prologue, SharedRun, init_prologue, next_left,
};
use crate::join_asof::two;
use crate::kernel::{NONE, gather_from, left_keys, make_ref, ref_row, ref_src, right_keys};
use crate::keys::{
    GroupKind, Key, NoGroup, PrimGroup, is_i32_like, is_primitive_group_type,
    pack_key, packed_group, packed_ts,
};
use crate::ordered::OrderedConcatExec;
use crate::plan_bytes::PlanNode;
use crate::scan::{nulls_first_of, pin_order, resplit, single_file_scan};

/// A markout: an exact decimal number of seconds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Markout {
    /// value = mantissa / 10^scale seconds, with no trailing zeros.
    mantissa: i128,
    scale: u32,
}

impl Markout {
    /// Parse a plain decimal (`"60"`, `"-5"`, `"0.5"`).
    pub fn parse(s: &str) -> Result<Self> {
        let t = s.trim();
        let (neg, body) = match t.strip_prefix('-') {
            Some(b) => (true, b),
            None => (false, t.strip_prefix('+').unwrap_or(t)),
        };
        let (int, frac) = body.split_once('.').unwrap_or((body, ""));
        let ok = |p: &str| p.bytes().all(|b| b.is_ascii_digit());
        if (int.is_empty() && frac.is_empty()) || !ok(int) || !ok(frac) || frac.len() > 30 {
            return plan_err!("join_asof_markouts: invalid markout {s:?}");
        }
        let digits = format!("{int}{frac}");
        let mut mantissa: i128 = digits
            .trim_start_matches('0')
            .parse()
            .or_else(|_| if digits.bytes().all(|b| b == b'0') { Ok(0) } else { Err(()) })
            .map_err(|_| plan_datafusion_err!("join_asof_markouts: markout {s:?} is too large"))?;
        let mut scale = frac.len() as u32;
        while scale > 0 && mantissa % 10 == 0 {
            mantissa /= 10;
            scale -= 1;
        }
        if mantissa == 0 {
            scale = 0;
        }
        Ok(Self {
            mantissa: if neg { -mantissa } else { mantissa },
            scale,
        })
    }

    /// The minimal decimal form, for column names: `1`, `0.5`, `-5`.
    pub fn label(&self) -> String {
        let neg = self.mantissa < 0;
        let digits = self.mantissa.unsigned_abs().to_string();
        let s = self.scale as usize;
        let body = if s == 0 {
            digits
        } else if digits.len() > s {
            format!("{}.{}", &digits[..digits.len() - s], &digits[digits.len() - s..])
        } else {
            format!("0.{}{}", "0".repeat(s - digits.len()), digits)
        };
        if neg { format!("-{body}") } else { body }
    }

    /// The markout in units of `10^-digits` seconds, if exact and in range.
    fn in_unit(&self, digits: u32) -> Option<i64> {
        if self.scale > digits {
            return None;
        }
        let v = self.mantissa.checked_mul(10i128.checked_pow(digits - self.scale)?)?;
        i64::try_from(v).ok()
    }
}

impl From<i64> for Markout {
    fn from(v: i64) -> Self {
        Self {
            mantissa: v as i128,
            scale: 0,
        }
    }
}

fn unit_digits(u: &TimeUnit) -> u32 {
    match u {
        TimeUnit::Second => 0,
        TimeUnit::Millisecond => 3,
        TimeUnit::Microsecond => 6,
        TimeUnit::Nanosecond => 9,
    }
}

/// One output column of the node.
#[derive(Debug, Clone, PartialEq, Eq)]
enum MCol {
    Left(String),
    /// Right column `column` at markout `k` (index into the given list).
    Right { k: usize, column: String },
}

/// Backward ASOF joins of `left.on + h` against `right.on` for every markout
/// `h`, within equal `by` groups. One wide output row per left row, in left
/// order: the left columns, then for each markout (in the given order) the
/// right columns except `on` and `by`, suffixed `_{h}s`.
#[derive(Debug, Clone)]
pub struct JoinAsofMarkouts {
    pub left_on: String,
    pub right_on: String,
    pub by_left: Vec<String>,
    pub by_right: Vec<String>,
    pub markouts: Vec<Markout>,
    pub allow_exact_matches: bool,
    pub options: AsofOptions,
}

impl JoinAsofMarkouts {
    pub fn new(on: impl Into<String>, markouts: Vec<Markout>) -> Self {
        let on = on.into();
        Self {
            left_on: on.clone(),
            right_on: on,
            by_left: Vec::new(),
            by_right: Vec::new(),
            markouts,
            allow_exact_matches: true,
            options: AsofOptions::default(),
        }
    }

    pub fn by<I: IntoIterator<Item = S>, S: Into<String>>(mut self, by: I) -> Self {
        self.by_left = by.into_iter().map(Into::into).collect();
        self.by_right = self.by_left.clone();
        self
    }

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

    pub fn options(mut self, options: AsofOptions) -> Self {
        self.options = options;
        self
    }

    /// Validate; return the markouts in the `on` column's unit.
    fn shifts(&self, left: &Schema, right: &Schema) -> Result<Vec<i64>> {
        let field = |s: &Schema, n: &str, which: &str| {
            s.field_with_name(n).map(|f| f.data_type().clone()).map_err(|_| {
                plan_datafusion_err!("join_asof_markouts: column '{n}' not found in the {which} input")
            })
        };
        let (lt, rt) = (field(left, &self.left_on, "left")?, field(right, &self.right_on, "right")?);
        let DataType::Timestamp(unit, _) = &lt else {
            return plan_err!("join_asof_markouts: 'on' must be a Timestamp column, got {lt} (left)");
        };
        if lt != rt {
            return plan_err!(
                "join_asof_markouts: 'on' columns must have the same type, got {lt} (left) and {rt} (right)"
            );
        }
        if self.by_left.len() != self.by_right.len() {
            return plan_err!("join_asof_markouts: by_left and by_right must have the same length");
        }
        if self.by_left.len() > 1 {
            return plan_err!("join_asof_markouts: at most one 'by' column is supported");
        }
        for (bl, br) in self.by_left.iter().zip(&self.by_right) {
            let (l, r) = (field(left, bl, "left")?, field(right, br, "right")?);
            if l != r || !is_primitive_group_type(&l) || l.is_floating() {
                return plan_err!(
                    "join_asof_markouts: 'by' must be an integer-like column of the same type on \
                     both sides, got {l} and {r}"
                );
            }
        }
        if self.markouts.is_empty() {
            return plan_err!("join_asof_markouts: give at least one markout");
        }
        let mut seen = HashSet::new();
        let mut shifts = Vec::with_capacity(self.markouts.len());
        for m in &self.markouts {
            if !seen.insert(*m) {
                return plan_err!("join_asof_markouts: duplicate markout {}", m.label());
            }
            let Some(v) = m.in_unit(unit_digits(unit)) else {
                return plan_err!(
                    "join_asof_markouts: markout {}s is not a whole number of {unit:?}s (or is out of range)",
                    m.label()
                );
            };
            shifts.push(v);
        }
        Ok(shifts)
    }

    /// Output columns: left, then per markout the right payload, suffixed.
    fn columns(&self, left: &Schema, right: &Schema) -> Result<Vec<(MCol, String)>> {
        self.shifts(left, right)?;
        let mut out: Vec<(MCol, String)> = left
            .fields()
            .iter()
            .map(|f| (MCol::Left(f.name().clone()), f.name().clone()))
            .collect();
        let mut names: HashSet<String> = out.iter().map(|c| c.1.clone()).collect();
        for (k, m) in self.markouts.iter().enumerate() {
            for f in right.fields() {
                let n = f.name();
                if n == &self.right_on || self.by_right.contains(n) {
                    continue;
                }
                let name = format!("{n}_{}s", m.label());
                if !names.insert(name.clone()) {
                    return plan_err!("join_asof_markouts: output column name '{name}' is not unique");
                }
                out.push((MCol::Right { k, column: n.clone() }, name));
            }
        }
        Ok(out)
    }

    fn output_schema(&self, left: &Schema, right: &Schema) -> Result<SchemaRef> {
        let fields = self
            .columns(left, right)?
            .into_iter()
            .map(|(c, name)| {
                Ok(match c {
                    MCol::Left(n) => left.field_with_name(&n)?.clone(),
                    MCol::Right { column, .. } => {
                        right.field_with_name(&column)?.clone().with_name(name).with_nullable(true)
                    }
                })
            })
            .collect::<Result<Vec<Field>>>()?;
        Ok(Arc::new(Schema::new(fields)))
    }

    /// Build the plan for the output columns `projection` (all if `None`).
    pub fn build(
        &self,
        left: Arc<dyn ExecutionPlan>,
        right: Arc<dyn ExecutionPlan>,
        projection: Option<&[usize]>,
        config: &ConfigOptions,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let (ls, rs) = (left.schema(), right.schema());
        let shifts = self.shifts(&ls, &rs)?;
        let all = self.columns(&ls, &rs)?;
        let outputs: Vec<(MCol, String)> = match projection {
            Some(p) => p.iter().map(|i| all[*i].clone()).collect(),
            None => all,
        };
        for (plan, which) in [(&left, "left"), (&right, "right")] {
            if !single_file_scan(plan) {
                return plan_err!(
                    "join_asof_markouts: the {which} input must be a scan of one Parquet file \
                     (optionally filtered or projected); other inputs are not supported yet"
                );
            }
        }
        let tp = config.execution.target_partitions.max(1);
        let col = |s: &Schema, n: &str| -> Result<PhysicalExprRef> {
            Ok(Arc::new(Column::new(n, s.index_of(n)?)) as PhysicalExprRef)
        };
        let left_ts = col(&ls, &self.left_on)?;
        let right_ts = col(&rs, &self.right_on)?;
        let lby = self.by_left.iter().map(|b| col(&ls, b)).collect::<Result<Vec<_>>>()?;
        let rby = self.by_right.iter().map(|b| col(&rs, b)).collect::<Result<Vec<_>>>()?;
        let mut kl = lby.clone();
        kl.push(Arc::clone(&left_ts));
        let mut kr = rby.clone();
        kr.push(Arc::clone(&right_ts));
        // Assumed sorted: a proven order only decides NULLS FIRST / LAST.
        let lnf = nulls_first_of(left.equivalence_properties(), &kl)?.unwrap_or(true);
        let rnf = nulls_first_of(right.equivalence_properties(), &kr)?.unwrap_or(true);

        let split = |p: Arc<dyn ExecutionPlan>, f: usize| -> Result<Arc<dyn ExecutionPlan>> {
            let p = resplit(&p, tp * f.max(1), config)?.unwrap_or(p);
            pin_order(p)
        };
        let left = split(left, self.options.left_split_factor)?;
        let right = split(right, self.options.right_split_factor)?;

        // The exec outputs every left column, then the projected right ones.
        let mut payload_idx: Vec<usize> = Vec::new();
        let mut right_out = Vec::new();
        for (c, name) in &outputs {
            if let MCol::Right { k, column } = c {
                let j = rs.index_of(column)?;
                let p = match payload_idx.iter().position(|x| *x == j) {
                    Some(p) => p,
                    None => {
                        payload_idx.push(j);
                        payload_idx.len() - 1
                    }
                };
                right_out.push((*k, p, name.clone()));
            }
        }
        let exec = MarkoutsExec::try_new(
            left,
            right,
            lby.into_iter().zip(rby).collect(),
            left_ts,
            right_ts,
            shifts,
            !self.allow_exact_matches,
            (lnf, rnf),
            payload_idx,
            right_out,
            self.options.clone(),
        )?;
        let exec: Arc<dyn ExecutionPlan> = Arc::new(exec);
        let joined: Arc<dyn ExecutionPlan> = if exec.output_partitioning().partition_count() > 1 {
            Arc::new(OrderedConcatExec::try_new(Arc::clone(&exec), tp + self.options.window_extra)?)
        } else {
            exec
        };
        let js = joined.schema();
        let exprs = outputs
            .iter()
            .map(|(c, name)| {
                let i = match c {
                    MCol::Left(n) => ls.index_of(n)?,
                    MCol::Right { .. } => js.index_of(name)?,
                };
                Ok((Arc::new(Column::new(js.field(i).name(), i)) as Arc<dyn PhysicalExpr>, name.clone()))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Arc::new(ProjectionExec::try_new(exprs, joined)?))
    }
}

impl PlanNode for JoinAsofMarkouts {
    fn schema(&self, inputs: &[SchemaRef]) -> Result<SchemaRef> {
        let [l, r] = two(inputs)?;
        self.output_schema(l, r)
    }

    fn required_columns(
        &self,
        inputs: &[SchemaRef],
        projection: &[usize],
    ) -> Result<Vec<Option<Vec<String>>>> {
        let [l, r] = two(inputs)?;
        let cols = self.columns(l, r)?;
        let mut left: Vec<String> = std::iter::once(&self.left_on).chain(&self.by_left).cloned().collect();
        let mut right: Vec<String> =
            std::iter::once(&self.right_on).chain(&self.by_right).cloned().collect();
        for &i in projection {
            match &cols[i].0 {
                MCol::Left(n) => left.push(n.clone()),
                MCol::Right { column, .. } => right.push(column.clone()),
            }
        }
        let keep = |s: &Schema, want: &[String]| -> Vec<String> {
            s.fields().iter().map(|f| f.name().clone()).filter(|n| want.contains(n)).collect()
        };
        Ok(vec![Some(keep(l, &left)), Some(keep(r, &right))])
    }

    fn build(
        &self,
        schemas: &[SchemaRef],
        inputs: Vec<Arc<dyn ExecutionPlan>>,
        projection: &[usize],
        config: &ConfigOptions,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        // Output names depend on the full schemas; the pruned inputs keep
        // every column the projection needs, under the same names.
        let [l, r] = two(schemas)?;
        let all = self.columns(l, r)?;
        let [left, right]: [Arc<dyn ExecutionPlan>; 2] = inputs
            .try_into()
            .map_err(|_| plan_datafusion_err!("join_asof_markouts takes two inputs"))?;
        let pruned = self.columns(&left.schema(), &right.schema())?;
        let proj = projection
            .iter()
            .map(|&i| {
                pruned.iter().position(|c| c.1 == all[i].1).ok_or_else(|| {
                    internal_datafusion_err!("join_asof_markouts: column {} missing", all[i].1)
                })
            })
            .collect::<Result<Vec<_>>>()?;
        self.build(left, right, Some(&proj), config)
    }
}

// ---------------------------------------------------------------------------
// The operator
// ---------------------------------------------------------------------------

/// The markouts operator: one output partition per left partition.
pub struct MarkoutsExec {
    left: Arc<dyn ExecutionPlan>,
    right: Arc<dyn ExecutionPlan>,
    on: Vec<(PhysicalExprRef, PhysicalExprRef)>,
    shifts: Vec<i64>,
    nulls_first: (bool, bool),
    /// Prologue/key spec shared with the asof operator.
    spec: Arc<JoinSpec>,
    m: Arc<MSpec>,
    shared: Mutex<Option<Arc<dyn Any + Send + Sync>>>,
    cache: Arc<PlanProperties>,
}

/// What a markouts worker needs besides the key spec.
#[derive(Debug)]
struct MSpec {
    /// Offsets in `on` units, sorted ascending (slowest cursor first)...
    shifts: Vec<i64>,
    /// ...and, for each given markout, its cursor (index into `shifts`).
    cursor_of: Vec<usize>,
    /// Output right columns: (given markout, payload column).
    right_out: Vec<(usize, usize)>,
    out_schema: SchemaRef,
    n_left_cols: usize,
}

impl fmt::Debug for MarkoutsExec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MarkoutsExec").field("shifts", &self.shifts).finish()
    }
}

impl MarkoutsExec {
    #[allow(clippy::too_many_arguments)]
    fn try_new(
        left: Arc<dyn ExecutionPlan>,
        right: Arc<dyn ExecutionPlan>,
        on: Vec<(PhysicalExprRef, PhysicalExprRef)>,
        left_ts: PhysicalExprRef,
        right_ts: PhysicalExprRef,
        shifts: Vec<i64>,
        strict: bool,
        nulls_first: (bool, bool),
        payload_idx: Vec<usize>,
        right_out: Vec<(usize, usize, String)>,
        options: AsofOptions,
    ) -> Result<Self> {
        let (ls, rs) = (left.schema(), right.schema());
        let payload_schema = Arc::new(rs.project(&payload_idx)?);
        let mut fields: Vec<Field> = ls.fields().iter().map(|f| f.as_ref().clone()).collect();
        for (_, p, name) in &right_out {
            fields.push(payload_schema.field(*p).clone().with_name(name.clone()).with_nullable(true));
        }
        let out_schema = Arc::new(Schema::new(fields));
        let (left_by, right_by): (Vec<_>, Vec<_>) = on.iter().cloned().unzip();
        let mut lk = left_by.clone();
        lk.push(Arc::clone(&left_ts));
        let sort: Vec<PhysicalSortExpr> = lk
            .iter()
            .map(|e| PhysicalSortExpr {
                expr: Arc::clone(e),
                options: SortOptions {
                    descending: false,
                    nulls_first: nulls_first.0,
                },
            })
            .collect();
        let ordering = LexOrdering::new(sort).ok_or_else(|| internal_datafusion_err!("no keys"))?;
        // Left columns keep their indices, so the left order holds as is.
        let eq = EquivalenceProperties::new_with_orderings(Arc::clone(&out_schema), [ordering]);
        let cache = PlanProperties::new(
            eq,
            Partitioning::UnknownPartitioning(left.output_partitioning().partition_count()),
            EmissionType::Incremental,
            Boundedness::Bounded,
        );
        let mut order: Vec<usize> = (0..shifts.len()).collect();
        order.sort_by_key(|&k| shifts[k]);
        let mut cursor_of = vec![0; shifts.len()];
        for (c, &k) in order.iter().enumerate() {
            cursor_of[k] = c;
        }
        let m = MSpec {
            shifts: order.iter().map(|&k| shifts[k]).collect(),
            cursor_of,
            right_out: right_out.iter().map(|(k, p, _)| (*k, *p)).collect(),
            out_schema: Arc::clone(&out_schema),
            n_left_cols: ls.fields().len(),
        };
        let spec = JoinSpec {
            left_ts,
            right_ts,
            left_by,
            right_by,
            strict,
            left_key_desc: String::new(),
            right_key_desc: String::new(),
            out_schema: Arc::clone(&out_schema),
            out_cols: Vec::new(),
            payload_idx,
            payload_schema,
            options,
        };
        Ok(Self {
            left,
            right,
            on,
            shifts,
            nulls_first,
            spec: Arc::new(spec),
            m: Arc::new(m),
            shared: Mutex::new(None),
            cache: Arc::new(cache),
        })
    }

    fn start<G: GroupKind>(
        &self,
        group: G,
        partition: usize,
        ctx: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        let n_left = self.left.output_partitioning().partition_count();
        let run = {
            let mut guard =
                self.shared.lock().map_err(|_| internal_datafusion_err!("poisoned lock"))?;
            match guard.as_ref().map(|a| Arc::clone(a).downcast::<SharedRun<i64, G>>()) {
                Some(Ok(run))
                    if run.claimed.len() == n_left
                        && !run.claimed[partition].swap(true, AtomicOrdering::SeqCst) =>
                {
                    run
                }
                _ => {
                    let run = Arc::new(SharedRun::<i64, G>::new(n_left));
                    run.claimed[partition].store(true, AtomicOrdering::SeqCst);
                    *guard = Some(Arc::clone(&run) as Arc<dyn Any + Send + Sync>);
                    run
                }
            }
        };
        let mut builder = RecordBatchReceiverStreamBuilder::new(Arc::clone(&self.m.out_schema), 2);
        let tx = builder.tx();
        let (left, right) = (Arc::clone(&self.left), Arc::clone(&self.right));
        let (spec, m, g) = (Arc::clone(&self.spec), Arc::clone(&self.m), Arc::new(group));
        builder.spawn(async move {
            let pro = Arc::clone(
                run.prologue
                    .get_or_try_init(|| {
                        init_prologue::<i64, G>(
                            Arc::clone(&left),
                            Arc::clone(&right),
                            Arc::clone(&ctx),
                            Arc::clone(&spec),
                            Arc::clone(&g),
                        )
                    })
                    .await?,
            );
            let start = pro
                .left
                .lock()
                .map_err(|_| internal_datafusion_err!("poisoned lock"))?[partition]
                .take();
            let Some((peeked, stream)) = start else {
                return exec_err!("join_asof_markouts: partition {partition} executed twice");
            };
            let mut w = MWorker::new(partition, pro, spec, m, g, right, ctx, tx);
            w.run(peeked, stream).await
        });
        Ok(builder.build())
    }
}

impl DisplayAs for MarkoutsExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "MarkoutsExec: shifts={:?}, by={}, strict={}",
            self.shifts,
            self.on.len(),
            self.spec.strict
        )
    }
}

impl ExecutionPlan for MarkoutsExec {
    fn name(&self) -> &str {
        "MarkoutsExec"
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.cache
    }

    fn maintains_input_order(&self) -> Vec<bool> {
        vec![true, false]
    }

    fn benefits_from_input_partitioning(&self) -> Vec<bool> {
        vec![false, false]
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.left, &self.right]
    }

    fn apply_expressions(
        &self,
        f: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> Result<TreeNodeRecursion>,
    ) -> Result<TreeNodeRecursion> {
        let keys = self.on.iter().flat_map(|(l, r)| [l, r]);
        apply_expression_roots(keys.chain([&self.spec.left_ts, &self.spec.right_ts]), f)
    }

    fn replace_children(
        self: Arc<Self>,
        mut children: Vec<Arc<dyn ExecutionPlan>>,
        _options: ReplaceChildrenOptions,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        if children.len() != 2 {
            return plan_err!("MarkoutsExec expects two children");
        }
        let right = children.swap_remove(1);
        let left = children.swap_remove(0);
        let names: Vec<String> = self.m.out_schema.fields()[self.m.n_left_cols..]
            .iter()
            .map(|f| f.name().clone())
            .collect();
        let right_out = self
            .m
            .right_out
            .iter()
            .zip(names)
            .map(|((k, p), n)| (*k, *p, n))
            .collect();
        Ok(Arc::new(Self::try_new(
            left,
            right,
            self.on.clone(),
            Arc::clone(&self.spec.left_ts),
            Arc::clone(&self.spec.right_ts),
            self.shifts.clone(),
            self.spec.strict,
            self.nulls_first,
            self.spec.payload_idx.clone(),
            right_out,
            self.spec.options.clone(),
        )?))
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        self.replace_children(
            children,
            ReplaceChildrenOptions::new(datafusion::physical_plan::ChildrenPropertiesMode::Recompute),
        )
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        let by: Vec<DataType> = self
            .on
            .iter()
            .map(|(l, _)| l.data_type(&self.left.schema()))
            .collect::<Result<_>>()?;
        match by.as_slice() {
            [] => self.start(NoGroup, partition, context),
            [t] if is_i32_like(t) => self.start(PrimGroup::<i32>::new(), partition, context),
            [_] => self.start(PrimGroup::<i64>::new(), partition, context),
            _ => exec_err!("join_asof_markouts: at most one by column"),
        }
    }
}

// ---------------------------------------------------------------------------
// The worker
// ---------------------------------------------------------------------------

/// A loaded right batch in the window: its packed keys and payload source.
struct WinBatch {
    src: u32,
    keys: Vec<i128>,
}

/// One cursor: position in the window and the candidate.
#[derive(Clone, Copy)]
struct Cursor {
    /// Absolute batch number (window front = `base`).
    w: usize,
    pos: usize,
    cand: u64,
    gpk: i64,
}

struct MWorker<G: GroupKind> {
    pro: Arc<Prologue<i64, G>>,
    p: usize,
    spec: Arc<JoinSpec>,
    m: Arc<MSpec>,
    group: Arc<G>,
    right_plan: Arc<dyn ExecutionPlan>,
    ctx: Arc<TaskContext>,
    tx: Sender<Result<RecordBatch>>,
    // Right stream.
    cur_idx: usize,
    stream: Option<SendableRecordBatchStream>,
    right_eof: bool,
    right_last: Option<i128>,
    // Window and payloads.
    window: VecDeque<WinBatch>,
    base: usize,
    next_src: u32,
    payload: HashMap<u32, RecordBatch>,
    retired: Vec<u32>,
    cursors: Vec<Cursor>,
    left_last: Option<i128>,
}

/// The packed threshold of `left + shift`: right rows with key `<=` it
/// qualify. Overflow saturates without leaving the group: past i64::MAX the
/// whole group qualifies; below i64::MIN nothing does.
#[inline]
fn threshold(left: i128, shift: i64, strict: bool) -> i128 {
    let g = packed_group(left);
    let v = packed_ts(left) as i128 + shift as i128 - strict as i128;
    if v > i64::MAX as i128 {
        pack_key(g, i64::MAX)
    } else if v < i64::MIN as i128 {
        pack_key(g, i64::MIN).saturating_sub(1)
    } else {
        pack_key(g, v as i64)
    }
}

impl<G: GroupKind> MWorker<G> {
    #[allow(clippy::too_many_arguments)]
    fn new(
        p: usize,
        pro: Arc<Prologue<i64, G>>,
        spec: Arc<JoinSpec>,
        m: Arc<MSpec>,
        group: Arc<G>,
        right_plan: Arc<dyn ExecutionPlan>,
        ctx: Arc<TaskContext>,
        tx: Sender<Result<RecordBatch>>,
    ) -> Self {
        let n = m.shifts.len();
        Self {
            pro,
            p,
            spec,
            m,
            group,
            right_plan,
            ctx,
            tx,
            cur_idx: 0,
            stream: None,
            right_eof: false,
            right_last: None,
            window: VecDeque::new(),
            base: 0,
            next_src: 0,
            payload: HashMap::new(),
            retired: Vec::new(),
            cursors: vec![Cursor { w: 0, pos: 0, cand: NONE, gpk: 0 }; n],
            left_last: None,
        }
    }

    fn packed_first(&self, k: &Key<i64, G>) -> i128 {
        pack_key(self.group.owned_packed(&k.0).unwrap_or(0), k.1)
    }

    /// Last right partition (in key order) whose first key is <= `t`.
    fn find_start(&self, t: i128) -> usize {
        self.pro
            .right_first
            .partition_point(|k| self.packed_first(k) <= t)
            .saturating_sub(1)
    }

    fn open(&mut self, idx: usize) -> Result<()> {
        self.stream = None;
        if self.pro.right_order.is_empty() {
            if idx == 0 && self.right_plan.output_partitioning().partition_count() == 1 {
                self.stream = Some(self.right_plan.execute(0, Arc::clone(&self.ctx))?);
            }
        } else if idx < self.pro.right_order.len() {
            let q = self.pro.right_order[idx];
            self.stream = Some(self.right_plan.execute(q, Arc::clone(&self.ctx))?);
        }
        self.cur_idx = idx;
        Ok(())
    }

    fn frontier(&self) -> usize {
        self.base + self.window.len()
    }

    /// Load the next right batch into the window. `t0` is the slowest
    /// cursor's current threshold: when every cursor has consumed the window,
    /// whole right partitions before its match are skipped.
    async fn load(&mut self, t0: Option<i128>, refs: &mut [Vec<u64>]) -> Result<bool> {
        loop {
            if let Some(t) = t0
                && self.cursors.iter().all(|c| c.w >= self.frontier())
                && self.cur_idx + 1 < self.pro.right_order.len()
                && self.packed_first(&self.pro.right_first[self.cur_idx + 1]) <= t
            {
                let want = self.find_start(t);
                if want > self.cur_idx {
                    self.open(want)?;
                }
            }
            let Some(s) = self.stream.as_mut() else {
                return Ok(false);
            };
            match s.next().await {
                None => {
                    self.stream = None;
                    if self.cur_idx + 1 < self.pro.right_order.len() {
                        self.open(self.cur_idx + 1)?;
                        continue;
                    }
                    return Ok(false);
                }
                Some(b) => {
                    let b = b?;
                    let Some((keys, payload)) = right_keys::<i64, G>(
                        &*self.group,
                        &self.spec.right_ts,
                        &self.spec.right_by,
                        &self.spec.payload_idx,
                        &b,
                    )?
                    else {
                        continue;
                    };
                    let packed = keys.packed.expect("packed keys");
                    if let Some(last) = self.right_last
                        && last > packed[0]
                    {
                        return exec_err!(
                            "join_asof_markouts: right input is not sorted by its keys (a batch \
                             or partition starts before the previous one ended)"
                        );
                    }
                    self.right_last = packed.last().copied();
                    let src = self.next_src;
                    self.next_src += 1;
                    self.payload.insert(src, payload);
                    self.window.push_back(WinBatch { src, keys: packed });
                    self.trim(refs)?;
                    return Ok(true);
                }
            }
        }
    }

    /// Retire batches every cursor has passed; compact retired payloads.
    fn trim(&mut self, refs: &mut [Vec<u64>]) -> Result<()> {
        let min_w = self.cursors.iter().map(|c| c.w).min().unwrap_or(0);
        while self.base < min_w && !self.window.is_empty() {
            let b = self.window.pop_front().expect("non-empty");
            self.retired.push(b.src);
            self.base += 1;
        }
        if self.retired.len() < self.spec.options.max_sources.max(2) {
            return Ok(());
        }
        // Gather the retired rows still referenced into one batch.
        let retired: HashSet<u32> = self.retired.iter().copied().collect();
        let mut idx: Vec<(usize, usize)> = Vec::new();
        let mut remap: HashMap<u64, u64> = HashMap::new();
        let new_src = self.next_src;
        self.next_src += 1;
        let srcs: Vec<u32> = self.retired.clone();
        let pos_of: HashMap<u32, usize> = srcs.iter().enumerate().map(|(i, s)| (*s, i)).collect();
        let mut visit = |r: &mut u64| {
            if *r == NONE || !retired.contains(&(ref_src(*r) as u32)) {
                return;
            }
            let new = *remap.entry(*r).or_insert_with(|| {
                idx.push((pos_of[&(ref_src(*r) as u32)], ref_row(*r)));
                make_ref(new_src, idx.len() - 1)
            });
            *r = new;
        };
        for c in self.cursors.iter_mut() {
            visit(&mut c.cand);
        }
        for set in refs.iter_mut() {
            for r in set.iter_mut() {
                visit(r);
            }
        }
        let batches: Vec<&RecordBatch> = srcs.iter().map(|s| &self.payload[s]).collect();
        let compact = gather_from(&self.spec.payload_schema, &batches, &idx)?;
        for s in &srcs {
            self.payload.remove(s);
        }
        self.payload.insert(new_src, compact);
        self.retired = vec![new_src];
        Ok(())
    }

    /// Answer rows `i..` of `lp` for every cursor, from `(*i, *c)`. Returns
    /// false when a cursor needs a right batch that is not loaded yet.
    fn kernel(&mut self, lp: &[i128], valid: &dyn Fn(usize) -> bool, i: &mut usize, c: &mut usize, refs: &mut [Vec<u64>]) -> bool {
        let strict = self.spec.strict;
        let n = lp.len();
        while *i < n {
            if !valid(*i) {
                *i += 1;
                continue;
            }
            let key = lp[*i];
            let g = packed_group(key);
            while *c < self.cursors.len() {
                let t = threshold(key, self.m.shifts[*c], strict);
                let cur = &mut self.cursors[*c];
                loop {
                    if cur.w >= self.base + self.window.len() {
                        if !self.right_eof {
                            return false;
                        }
                        break;
                    }
                    let wb = &self.window[cur.w - self.base];
                    let (rk, pos) = (&wb.keys, cur.pos);
                    let rlen = rk.len();
                    let e = if pos < rlen && rk[rlen - 1] <= t {
                        rlen
                    } else {
                        crate::kernel::gallop_by(rlen, pos, |j| rk[j] <= t)
                    };
                    if e > pos {
                        cur.cand = make_ref(wb.src, e - 1);
                        cur.gpk = packed_group(rk[e - 1]);
                    }
                    cur.pos = e;
                    if e < rlen {
                        break;
                    }
                    cur.w += 1;
                    cur.pos = 0;
                }
                refs[*c][*i] = if cur.cand != NONE && cur.gpk == g { cur.cand } else { NONE };
                *c += 1;
            }
            *c = 0;
            *i += 1;
        }
        true
    }

    async fn run(
        &mut self,
        mut peeked: VecDeque<RecordBatch>,
        mut stream: Option<SendableRecordBatchStream>,
    ) -> Result<()> {
        // Seek: the right range holding the slowest cursor's first match.
        match &self.pro.left_first[self.p] {
            Some(k) => {
                let first = self.packed_first(k);
                let t = threshold(first, self.m.shifts[0], self.spec.strict);
                self.open(self.find_start(t))?;
            }
            None => self.right_eof = true,
        }
        let nc = self.cursors.len();
        while let Some(batch) = next_left(&mut peeked, &mut stream).await? {
            let n = batch.num_rows();
            if n == 0 {
                continue;
            }
            let lk = left_keys::<i64, G>(&*self.group, &self.spec.left_ts, &self.spec.left_by, &batch)?;
            let lp = lk.packed.as_deref().expect("packed keys");
            if let (Some(f), Some(l)) = (lk.first_valid(), lk.last_valid()) {
                if self.left_last.is_some_and(|p| p > lp[f]) {
                    return exec_err!(
                        "join_asof_markouts: left input is not sorted by its keys (a batch starts \
                         before the previous one ended)"
                    );
                }
                self.left_last = Some(lp[l]);
            }
            let mut refs = vec![vec![NONE; n]; nc];
            let (mut i, mut c) = (0usize, 0usize);
            let valid = |r: usize| lk.is_valid(r);
            while !self.kernel(lp, &valid, &mut i, &mut c, &mut refs) {
                let t0 = threshold(lp[i], self.m.shifts[0], self.spec.strict);
                if !self.load(if c == 0 { Some(t0) } else { None }, &mut refs).await? {
                    self.right_eof = true;
                }
            }
            let out = self.output(&batch, &refs)?;
            if self.tx.send(Ok(out)).await.is_err() {
                return Ok(());
            }
            self.trim(&mut refs)?;
        }
        Ok(())
    }

    fn output(&self, left: &RecordBatch, refs: &[Vec<u64>]) -> Result<RecordBatch> {
        let mut cols: Vec<ArrayRef> = left.columns().to_vec();
        for &(k, p) in &self.m.right_out {
            cols.push(self.take(p, &refs[self.m.cursor_of[k]])?);
        }
        let opts = RecordBatchOptions::new().with_row_count(Some(left.num_rows()));
        Ok(RecordBatch::try_new_with_options(Arc::clone(&self.m.out_schema), cols, &opts)?)
    }

    /// Payload column `p` for `refs` (NONE = null).
    fn take(&self, p: usize, refs: &[u64]) -> Result<ArrayRef> {
        let dt = self.spec.payload_schema.field(p).data_type();
        let mut srcs: Vec<u32> = Vec::new();
        for r in refs.iter().filter(|r| **r != NONE) {
            let s = ref_src(*r) as u32;
            if !srcs.contains(&s) {
                srcs.push(s);
            }
        }
        match srcs.as_slice() {
            [] => Ok(new_null_array(dt, refs.len())),
            [s] => {
                let idx: UInt32Array = refs
                    .iter()
                    .map(|r| (*r != NONE).then(|| ref_row(*r) as u32))
                    .collect();
                Ok(take(self.payload[s].column(p).as_ref(), &idx, None)?)
            }
            _ => {
                let null = new_null_array(dt, 1);
                let mut arrs: Vec<&dyn Array> = vec![null.as_ref()];
                arrs.extend(srcs.iter().map(|s| self.payload[s].column(p).as_ref()));
                let idx: Vec<(usize, usize)> = refs
                    .iter()
                    .map(|r| match srcs.iter().position(|s| *r != NONE && *s == ref_src(*r) as u32) {
                        Some(k) => (k + 1, ref_row(*r)),
                        None => (0, 0),
                    })
                    .collect();
                Ok(interleave(&arrs, &idx)?)
            }
        }
    }
}
