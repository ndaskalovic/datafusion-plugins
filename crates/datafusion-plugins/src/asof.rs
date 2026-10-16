//! `AsofExec`: the streaming backward ASOF join behind `join_asof` (one
//! offset, 0) and `join_asof_markouts` (N offsets).
//!
//! For each left row and each offset `h`, the last right row with the same
//! `by` values and `(by, on) <= (by, left.on + h)` (`<` when strict).
//!
//! Works after any upstream plan; every input partition is executed exactly
//! once and nothing is re-executed:
//!
//! 1. **Order.** Unless `assume_sorted`, the node requires each input
//!    partition sorted by `(by..., on)` (see `ordered`). With
//!    `assume_sorted` the order is trusted and checked while streaming.
//! 2. **Merge.** Each side's partitions are merged into one ordered stream
//!    (see `merge`). Left rows with a NULL key never match; when the input's
//!    NULL placement isn't known they keep their place in their partition.
//! 3. **Distributor.** A serial distributor reads the left stream batch by
//!    batch. For a left batch it makes sure the right buffer reaches past the
//!    last row it can match (`last + h_max`), and hands a worker zero-copy
//!    slices of the buffer from the predecessor of `first + h_min`. Right
//!    batches before that predecessor can't be reached by this or any later
//!    left row (keys only grow), so they are dropped: memory is bounded by
//!    that frontier and the batches in flight, not by the input.
//! 4. **Workers** run the galloping kernel (N cursors over the slices) and
//!    return output batches tagged with their sequence number; a reorder
//!    stage emits them in left order, as one partition.
//!
//! The right buffer and the outputs waiting to be emitted are reserved from
//! the memory pool.

use std::cmp::Ordering;
use std::collections::{BTreeMap, VecDeque};
use std::fmt;
use std::sync::Arc;

use arrow::array::{ArrayRef, RecordBatch, RecordBatchOptions, UInt32Array};
use arrow::compute::{SortOptions, take};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use arrow::row::{RowConverter, Rows, SortField};
use datafusion::common::config::ConfigOptions;
use datafusion::common::tree_node::TreeNodeRecursion;
use datafusion::common::{DataFusionError, Result, exec_err, internal_err, plan_err, resources_err};
use datafusion::execution::TaskContext;
use datafusion::execution::memory_pool::{MemoryConsumer, MemoryLimit, MemoryReservation};
use datafusion::physical_expr::expressions::Column;
use datafusion::physical_expr::{
    EquivalenceProperties, LexOrdering, OrderingRequirements, PhysicalExpr, PhysicalExprRef,
};
use datafusion::physical_expr_common::physical_expr::fmt_sql;
use datafusion::physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion::physical_plan::metrics::{BaselineMetrics, ExecutionPlanMetricsSet, MetricsSet};
use datafusion::physical_plan::projection::ProjectionExec;
use datafusion::physical_plan::stream::RecordBatchReceiverStreamBuilder;
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, Distribution, ExecutionPlan, InputDistributionRequirements, Partitioning,
    PlanProperties, SendableRecordBatchStream, apply_expression_roots,
};
use tokio::sync::mpsc::Sender;

use crate::kernel::{Gather, NONE, eval_side, gallop_by, left_keys, make_ref, ref_row, ref_src, right_keys};
use crate::keys::{GroupKind, Keys, NoGroup, PrimGroup, RowGroup, is_i32_like, is_primitive_group_type, pack_key, packed_group, packed_ts};
use crate::merge::{MergeKeys, MergedStream, Prep, Prepared, merged};
use crate::ordered::{enforce, nulls_first, order_check, requirement, sort_exprs};

/// What the planners give the exec.
#[derive(Debug, Clone)]
pub(crate) struct AsofSpec {
    pub left_on: PhysicalExprRef,
    pub right_on: PhysicalExprRef,
    pub left_by: Vec<PhysicalExprRef>,
    pub right_by: Vec<PhysicalExprRef>,
    /// Offsets added to the left key, in the key's units (`[0]` for a plain
    /// ASOF join).
    pub shifts: Vec<i64>,
    /// `>` instead of `>=`.
    pub strict: bool,
    /// Right columns carried to the output.
    pub payload_idx: Vec<usize>,
    /// Output right columns: (offset index, payload index, name).
    pub right_out: Vec<(usize, usize, String)>,
    /// Trust the inputs' order instead of requiring it.
    pub assume_sorted: bool,
    /// With `assume_sorted`: check the order inside every batch.
    pub check_sortedness: bool,
    /// NULLS FIRST of each input's order, `None` if unknown (set by
    /// [`build`]).
    pub nulls_first: (Option<bool>, Option<bool>),
}

impl AsofSpec {
    fn left_keys(&self) -> Vec<PhysicalExprRef> {
        let mut k = self.left_by.clone();
        k.push(Arc::clone(&self.left_on));
        k
    }

    fn right_keys(&self) -> Vec<PhysicalExprRef> {
        let mut k = self.right_by.clone();
        k.push(Arc::clone(&self.right_on));
        k
    }
}

/// Build the exec over `left` and `right`, let DataFusion enforce its input
/// order (unless `assume_sorted`), and project to `outputs`: indices into
/// `[left columns..., right_out...]` with their output names.
pub(crate) fn build(
    left: Arc<dyn ExecutionPlan>,
    right: Arc<dyn ExecutionPlan>,
    mut spec: AsofSpec,
    outputs: &[(usize, String)],
    config: &ConfigOptions,
) -> Result<Arc<dyn ExecutionPlan>> {
    spec.nulls_first = (
        nulls_first(&left, &spec.left_keys(), spec.assume_sorted)?,
        nulls_first(&right, &spec.right_keys(), spec.assume_sorted)?,
    );
    let assume_sorted = spec.assume_sorted;
    let exec = enforce(Arc::new(AsofExec::try_new(left, right, spec)?), assume_sorted, config)?;
    let schema = exec.schema();
    let exprs = outputs
        .iter()
        .map(|(i, name)| (Arc::new(Column::new(schema.field(*i).name(), *i)) as Arc<dyn PhysicalExpr>, name.clone()))
        .collect::<Vec<_>>();
    Ok(Arc::new(ProjectionExec::try_new(exprs, exec)?))
}

/// Group key representation chosen from the `by` types.
#[derive(Debug, Clone, PartialEq, Eq)]
enum GroupSpec {
    None,
    I32,
    I64,
    Rows(Vec<DataType>),
}

/// The operator.
pub(crate) struct AsofExec {
    left: Arc<dyn ExecutionPlan>,
    right: Arc<dyn ExecutionPlan>,
    spec: Arc<AsofSpec>,
    group: GroupSpec,
    /// Indices of the right `by` and `on` columns.
    right_cols: Vec<usize>,
    out_schema: SchemaRef,
    payload_schema: SchemaRef,
    metrics: ExecutionPlanMetricsSet,
    cache: Arc<PlanProperties>,
}

impl fmt::Debug for AsofExec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AsofExec").field("shifts", &self.spec.shifts).finish()
    }
}

impl AsofExec {
    pub fn try_new(left: Arc<dyn ExecutionPlan>, right: Arc<dyn ExecutionPlan>, spec: AsofSpec) -> Result<Self> {
        let (ls, rs) = (left.schema(), right.schema());
        if spec.shifts.is_empty() {
            return plan_err!("AsofExec: no offsets");
        }
        if spec.left_by.len() != spec.right_by.len() {
            return plan_err!("AsofExec: {} left and {} right 'by' keys", spec.left_by.len(), spec.right_by.len());
        }
        let (lt, rt) = (spec.left_on.data_type(&ls)?, spec.right_on.data_type(&rs)?);
        if lt != rt {
            return plan_err!("AsofExec: 'on' expressions have different types: {lt} and {rt}");
        }
        let mut by = Vec::new();
        for (l, r) in spec.left_by.iter().zip(&spec.right_by) {
            let (lt, rt) = (l.data_type(&ls)?, r.data_type(&rs)?);
            if lt != rt {
                return plan_err!("AsofExec: 'by' expressions {l} and {r} have different types: {lt} and {rt}");
            }
            by.push(lt);
        }
        let group = match by.as_slice() {
            [] => GroupSpec::None,
            [t] if is_i32_like(t) => GroupSpec::I32,
            [t] if is_primitive_group_type(t) => GroupSpec::I64,
            ts if RowGroup::supports(ts) => GroupSpec::Rows(ts.to_vec()),
            ts => return plan_err!("AsofExec: unsupported 'by' types {ts:?}"),
        };
        let right_cols = spec
            .right_by
            .iter()
            .chain([&spec.right_on])
            .map(|e| match e.downcast_ref::<Column>() {
                Some(c) => Ok(c.index()),
                None => plan_err!("AsofExec: right keys must be columns, got {e}"),
            })
            .collect::<Result<Vec<_>>>()?;
        if let Some(&j) = spec.payload_idx.iter().find(|&&j| j >= rs.fields().len()) {
            return plan_err!("AsofExec: payload column {j} out of range");
        }
        if let Some((k, p, _)) = spec.right_out.iter().find(|(k, p, _)| *k >= spec.shifts.len() || *p >= spec.payload_idx.len()) {
            return plan_err!("AsofExec: output (offset {k}, payload {p}) out of range");
        }
        let payload_schema = Arc::new(rs.project(&spec.payload_idx)?);
        let mut fields: Vec<Field> = ls.fields().iter().map(|f| f.as_ref().clone()).collect();
        for (_, p, name) in &spec.right_out {
            fields.push(payload_schema.field(*p).clone().with_name(name.clone()).with_nullable(true));
        }
        let out_schema = Arc::new(Schema::new(fields));
        // Output = left order: the left columns keep their positions. With
        // an unknown NULL placement that order is only declared if the keys
        // can't be NULL.
        let lk = spec.left_keys();
        let nf = match spec.nulls_first.0 {
            Some(nf) => Some(nf),
            None => (!lk.iter().any(|e| e.nullable(&ls).unwrap_or(true))).then_some(true),
        };
        let eq = match nf.and_then(|nf| LexOrdering::new(sort_exprs(&lk, nf))) {
            Some(lex) => EquivalenceProperties::new_with_orderings(Arc::clone(&out_schema), [lex]),
            None => EquivalenceProperties::new(Arc::clone(&out_schema)),
        };
        let cache = PlanProperties::new(eq, Partitioning::UnknownPartitioning(1), EmissionType::Incremental, Boundedness::Bounded);
        Ok(Self {
            left,
            right,
            spec: Arc::new(spec),
            group,
            right_cols,
            out_schema,
            payload_schema,
            metrics: ExecutionPlanMetricsSet::new(),
            cache: Arc::new(cache),
        })
    }

    fn start<G: GroupKind>(&self, g: G, ctx: Arc<TaskContext>) -> Result<SendableRecordBatchStream> {
        let mut builder = RecordBatchReceiverStreamBuilder::new(Arc::clone(&self.out_schema), 2);
        let d = Driver {
            g: Arc::new(g),
            spec: Arc::clone(&self.spec),
            left: Arc::clone(&self.left),
            right: Arc::clone(&self.right),
            right_cols: self.right_cols.clone(),
            out_schema: Arc::clone(&self.out_schema),
            payload_schema: Arc::clone(&self.payload_schema),
            tx: builder.tx(),
            metrics: BaselineMetrics::new(&self.metrics, 0),
        };
        builder.spawn(async move { d.run(ctx).await });
        Ok(builder.build())
    }
}

impl DisplayAs for AsofExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = &self.spec;
        let on = s
            .left_by
            .iter()
            .zip(&s.right_by)
            .map(|(l, r)| format!("({} = {})", fmt_sql(l.as_ref()), fmt_sql(r.as_ref())))
            .collect::<Vec<_>>()
            .join(", ");
        let op = if s.strict { ">" } else { ">=" };
        let (lo, ro) = (fmt_sql(s.left_on.as_ref()), fmt_sql(s.right_on.as_ref()));
        let assumed = if s.assume_sorted { ", assume_sorted" } else { "" };
        write!(f, "AsofExec: on=[{on}], match=[{lo} {op} {ro} + h], h={:?}{assumed}", s.shifts)
    }
}

impl ExecutionPlan for AsofExec {
    fn name(&self) -> &str {
        "AsofExec"
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.cache
    }

    fn input_distribution_requirements(&self) -> InputDistributionRequirements {
        // Any partitioning: the node merges each side's sorted partitions.
        let d = Distribution::UnspecifiedDistribution;
        InputDistributionRequirements::new(vec![d.clone(), d])
    }

    fn required_input_ordering(&self) -> Vec<Option<OrderingRequirements>> {
        let s = &self.spec;
        if s.assume_sorted {
            return vec![None, None];
        }
        vec![
            requirement(&s.left_keys(), s.nulls_first.0.unwrap_or(true)),
            requirement(&s.right_keys(), s.nulls_first.1.unwrap_or(true)),
        ]
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
        let s = &self.spec;
        let keys = s.left_by.iter().chain(&s.right_by);
        apply_expression_roots(keys.chain([&s.left_on, &s.right_on]), f)
    }

    fn with_new_children(self: Arc<Self>, children: Vec<Arc<dyn ExecutionPlan>>) -> Result<Arc<dyn ExecutionPlan>> {
        let Ok([left, right]) = <[_; 2]>::try_from(children) else {
            return internal_err!("AsofExec expects two children");
        };
        Ok(Arc::new(Self::try_new(left, right, self.spec.as_ref().clone())?))
    }

    fn execute(&self, partition: usize, ctx: Arc<TaskContext>) -> Result<SendableRecordBatchStream> {
        if partition != 0 {
            return exec_err!("AsofExec has one output partition, got {partition}");
        }
        match &self.group {
            GroupSpec::None => self.start(NoGroup, ctx),
            GroupSpec::I32 => self.start(PrimGroup::<i32>::default(), ctx),
            GroupSpec::I64 => self.start(PrimGroup::<i64>::default(), ctx),
            GroupSpec::Rows(t) => self.start(RowGroup::try_new(t)?, ctx),
        }
    }

    fn metrics(&self) -> Option<MetricsSet> {
        Some(self.metrics.clone_inner())
    }
}

// ---------------------------------------------------------------------------
// Keys and targets
// ---------------------------------------------------------------------------

/// Left keys of one batch: the merge order (in the input's NULL placement)
/// and the kernel's keys.
struct LKeys<G: GroupKind> {
    order: Rows,
    keys: Arc<Keys<G>>,
    bounds: Option<(usize, usize)>,
    g: Arc<G>,
}

impl<G: GroupKind> MergeKeys for LKeys<G> {
    fn len(&self) -> usize {
        self.keys.len
    }
    fn cmp_at(&self, i: usize, other: &Self, j: usize) -> Ordering {
        self.order.row(i).cmp(&other.order.row(j))
    }
    fn is_sorted(&self) -> bool {
        match self.keys.valid {
            None => self.keys.is_sorted(&self.g),
            Some(_) => (1..self.len()).all(|i| self.cmp_at(i - 1, self, i) != Ordering::Greater),
        }
    }
    fn bounds(&self) -> Option<(usize, usize)> {
        self.bounds
    }
}

/// Right keys (no NULLs) of one batch, comparable in `(by, on)` order.
struct RKeys<G: GroupKind> {
    keys: Keys<G>,
    g: Arc<G>,
}

impl<G: GroupKind> MergeKeys for RKeys<G> {
    fn len(&self) -> usize {
        self.keys.len
    }
    fn cmp_at(&self, i: usize, other: &Self, j: usize) -> Ordering {
        match (&self.keys.packed, &other.keys.packed) {
            (Some(a), Some(b)) => a[i].cmp(&b[j]),
            _ => self
                .g
                .cmp_at(&self.keys.grp, i, &other.keys.grp, j)
                .then(self.keys.ts.as_slice()[i].cmp(&other.keys.ts.as_slice()[j])),
        }
    }
    fn is_sorted(&self) -> bool {
        self.keys.is_sorted(&self.g)
    }
}

/// Bytes reserved from the memory pool, released on drop.
struct Reserved {
    res: Arc<MemoryReservation>,
    size: usize,
}

impl Drop for Reserved {
    fn drop(&mut self) {
        self.res.shrink(self.size);
    }
}

/// A right batch in the buffer: payload columns and keys, reserved until
/// the last worker using it is done.
struct RBatch<G: GroupKind> {
    payload: RecordBatch,
    keys: Arc<RKeys<G>>,
    _mem: Reserved,
}

/// "Right row `<=` this": packed `(group, on + h)` (overflow saturates within
/// the group), or a left row's group with `on + h` as i128.
#[derive(Clone, Copy)]
enum Target {
    Packed(i128),
    Row { i: usize, ts: i128 },
}

#[inline]
fn packed_threshold(left: i128, shift: i64, strict: bool) -> i128 {
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

fn target<G: GroupKind>(lk: &Keys<G>, i: usize, shift: i64, strict: bool) -> Target {
    match &lk.packed {
        Some(p) => Target::Packed(packed_threshold(p[i], shift, strict)),
        None => Target::Row { i, ts: lk.ts.as_slice()[i] as i128 + shift as i128 - strict as i128 },
    }
}

/// Is right row `j` of `rk` `<=` target `t` (of left keys `lk`)?
#[inline]
fn le<G: GroupKind>(g: &G, rk: &Keys<G>, j: usize, lk: &Keys<G>, t: Target) -> bool {
    match t {
        Target::Packed(t) => rk.packed.as_ref().expect("packed")[j] <= t,
        Target::Row { i, ts } => match g.cmp_at(&rk.grp, j, &lk.grp, i) {
            Ordering::Less => true,
            Ordering::Greater => false,
            Ordering::Equal => (rk.ts.as_slice()[j] as i128) <= ts,
        },
    }
}

// ---------------------------------------------------------------------------
// The driver: merges, distributor, workers, reorder
// ---------------------------------------------------------------------------

struct Driver<G: GroupKind> {
    g: Arc<G>,
    spec: Arc<AsofSpec>,
    left: Arc<dyn ExecutionPlan>,
    right: Arc<dyn ExecutionPlan>,
    right_cols: Vec<usize>,
    out_schema: SchemaRef,
    payload_schema: SchemaRef,
    tx: Sender<Result<RecordBatch>>,
    metrics: BaselineMetrics,
}

/// A zero-copy slice of a right buffer batch handed to a worker.
struct RSlice<G: GroupKind> {
    b: Arc<RBatch<G>>,
    start: usize,
    end: usize,
}

/// Worker outputs, emitted in left order and reserved until then.
struct Reorder {
    work: tokio::task::JoinSet<(u64, Result<RecordBatch>)>,
    ready: BTreeMap<u64, (RecordBatch, Reserved)>,
    seq: u64,
    next: u64,
    /// The pool refused the outputs' memory: dispatch nothing new until
    /// they are emitted.
    throttled: bool,
    res: Arc<MemoryReservation>,
}

impl Reorder {
    fn in_flight(&self) -> u64 {
        self.seq - self.next
    }

    fn spawn(&mut self, f: impl FnOnce() -> Result<RecordBatch> + Send + 'static) {
        let n = self.seq;
        self.work.spawn(async move { (n, f()) });
        self.seq += 1;
    }

    /// Wait for a worker and emit what is ready, in order. `None` when
    /// nothing is in flight, `Some(false)` when the output is gone.
    async fn collect(&mut self, tx: &Sender<Result<RecordBatch>>, metrics: &BaselineMetrics) -> Result<Option<bool>> {
        let Some(res) = self.work.join_next().await else {
            return Ok(None);
        };
        let (n, out) = res.map_err(|e| DataFusionError::External(Box::new(e)))?;
        let out = out?;
        let size = out.get_array_memory_size();
        if self.res.try_grow(size).is_err() {
            self.res.grow(size);
            self.throttled = true;
        }
        self.ready.insert(n, (out, Reserved { res: Arc::clone(&self.res), size }));
        while let Some((b, _mem)) = self.ready.remove(&self.next) {
            self.next += 1;
            let rows = b.num_rows();
            if tx.send(Ok(b)).await.is_err() {
                return Ok(Some(false));
            }
            metrics.record_output(rows);
        }
        self.throttled &= !self.ready.is_empty();
        Ok(Some(true))
    }
}

impl<G: GroupKind> Driver<G> {
    /// Left batches keep their columns; keys are the merge order and the
    /// kernel's keys. Null keys never match: when the input's NULL placement
    /// isn't known, a NULL-key row takes the previous valid row's key (or
    /// the batch's first row's), so the partition stays ordered wherever its
    /// NULLs are, and the order check skips it.
    fn left_prep(&self) -> Result<Prep<LKeys<G>>> {
        let (g, s, ls) = (Arc::clone(&self.g), Arc::clone(&self.spec), self.left.schema());
        let carry = s.nulls_first.0.is_none();
        let options = SortOptions { descending: false, nulls_first: s.nulls_first.0.unwrap_or(true) };
        let fields = s
            .left_keys()
            .iter()
            .map(|e| Ok(SortField::new_with_options(e.data_type(&ls)?, options)))
            .collect::<Result<Vec<_>>>()?;
        let conv = RowConverter::new(fields)?;
        Ok(Arc::new(move |batch: RecordBatch| {
            if batch.num_rows() == 0 {
                return Ok(None);
            }
            let (ts, by, valid) = eval_side(&s.left_on, &s.left_by, &batch)?;
            let keys = Keys::new(&*g, &ts, &by, valid)?;
            let mut cols: Vec<ArrayRef> = by;
            cols.push(ts);
            let mut bounds = Some((0, keys.len - 1));
            if carry && keys.valid.is_some() {
                let mut last = 0;
                let idx: UInt32Array = (0..keys.len as u32)
                    .map(|i| {
                        if keys.is_valid(i as usize) {
                            last = i;
                        }
                        last
                    })
                    .collect();
                cols = cols.iter().map(|c| Ok(take(c, &idx, None)?)).collect::<Result<_>>()?;
                bounds = keys.first_valid().zip(keys.last_valid());
            }
            let order = conv.convert_columns(&cols)?;
            let keys = Arc::new(LKeys { order, keys: Arc::new(keys), bounds, g: Arc::clone(&g) });
            Ok(Some(Prepared { batch, keys }))
        }))
    }

    /// Right batches keep the payload columns, then the key columns (so a
    /// merged slice can be re-keyed): `[payload..., by..., on]`.
    fn right_prep(&self) -> Prep<RKeys<G>> {
        let (g, spec) = (Arc::clone(&self.g), Arc::clone(&self.spec));
        let mut idx = spec.payload_idx.clone();
        idx.extend(&self.right_cols);
        Arc::new(move |batch: RecordBatch| {
            let Some((keys, payload)) = right_keys(&*g, &spec.right_on, &spec.right_by, &idx, &batch)? else {
                return Ok(None);
            };
            let keys = Arc::new(RKeys { keys, g: Arc::clone(&g) });
            Ok(Some(Prepared { batch: payload, keys }))
        })
    }

    async fn run(self, ctx: Arc<TaskContext>) -> Result<()> {
        let s = &self.spec;
        let check = |side, k: &[PhysicalExprRef]| order_check(side, k, s.assume_sorted, s.check_sortedness);
        let mut lstream: MergedStream<LKeys<G>> =
            merged(&self.left, &ctx, self.left_prep()?, check("left", &s.left_keys()), usize::MAX)?;
        let mut rstream: MergedStream<RKeys<G>> =
            merged(&self.right, &ctx, self.right_prep(), check("right", &s.right_keys()), usize::MAX)?;
        let pool = ctx.memory_pool();
        let res = Arc::new(MemoryConsumer::new("AsofExec").register(pool));

        // Batches dispatched but not yet emitted: enough that a slow batch
        // doesn't idle the workers.
        let max_inflight = 4 * ctx.session_config().target_partitions().max(1) as u64 + 4;
        let h_min = *s.shifts.iter().min().expect("shifts");
        let h_max = *s.shifts.iter().max().expect("shifts");
        let mut buf: VecDeque<Arc<RBatch<G>>> = VecDeque::new();
        // Bytes of the batches in `buf` (in-flight workers may hold more).
        let mut buf_bytes = 0usize;
        let (mut right_done, mut left_done) = (false, false);
        let mut out = Reorder {
            work: tokio::task::JoinSet::new(),
            ready: BTreeMap::new(),
            seq: 0,
            next: 0,
            throttled: false,
            res: Arc::clone(&res),
        };
        loop {
            if !left_done && out.in_flight() < max_inflight && !out.throttled {
                let Some(m) = lstream.next().await? else {
                    left_done = true;
                    continue;
                };
                let lk = match m.keys {
                    Some(k) => Arc::clone(&k.keys),
                    None => Arc::new(left_keys(&*self.g, &s.left_on, &s.left_by, &m.batch)?),
                };
                let slices = match (lk.first_valid(), lk.last_valid()) {
                    (Some(f), Some(l)) => {
                        let (tmin, tmax) = (target(&lk, f, h_min, s.strict), target(&lk, l, h_max, s.strict));
                        // The buffer must reach past tmax (or the right end).
                        while !right_done
                            && buf.back().is_none_or(|b| le(&*self.g, &b.keys.keys, b.keys.keys.len - 1, &lk, tmax))
                        {
                            match rstream.next().await? {
                                Some(m) => {
                                    let rb = self.rbatch(m.batch, m.keys, &res, buf_bytes, pool.as_ref())?;
                                    buf_bytes += rb._mem.size;
                                    buf.push_back(Arc::new(rb));
                                }
                                None => right_done = true,
                            }
                        }
                        let (sb, sr) = self.predecessor(&buf, &lk, tmin);
                        let (eb, er) = self.upper(&buf, &lk, tmax);
                        let mut slices = Vec::new();
                        for b in sb..buf.len().min(eb + 1) {
                            let start = if b == sb { sr } else { 0 };
                            let end = if b == eb { er } else { buf[b].keys.keys.len };
                            if end > start {
                                slices.push(RSlice { b: Arc::clone(&buf[b]), start, end });
                            }
                        }
                        // Frontier prune: nothing before the predecessor of
                        // tmin is reachable any more.
                        buf_bytes -= buf.drain(..sb).map(|b| b._mem.size).sum::<usize>();
                        slices
                    }
                    _ => Vec::new(),
                };
                let (g, spec, os) = (Arc::clone(&self.g), Arc::clone(&self.spec), Arc::clone(&self.out_schema));
                let ps = Arc::clone(&self.payload_schema);
                let batch = m.batch;
                out.spawn(move || work_batch(&*g, &spec, &os, &ps, &batch, &lk, &slices));
                continue;
            }
            match out.collect(&self.tx, &self.metrics).await? {
                Some(true) => {}
                Some(false) => return Ok(()),
                None => break, // left done, nothing in flight
            }
        }
        self.metrics.done();
        Ok(())
    }

    /// A merged right batch for the buffer: its payload, its keys (from its
    /// key columns, after the payload, for a slice or a gather), reserved.
    /// Fails only if the buffer itself (`buffered` bytes, plus this batch)
    /// exceeds a finite pool; otherwise it grows past the pool, and the
    /// pool's other consumers (the merges' read-ahead, this node's outputs)
    /// give way.
    fn rbatch(
        &self,
        batch: RecordBatch,
        keys: Option<Arc<RKeys<G>>>,
        res: &Arc<MemoryReservation>,
        buffered: usize,
        pool: &dyn datafusion::execution::memory_pool::MemoryPool,
    ) -> Result<RBatch<G>> {
        let p = self.payload_schema.fields().len();
        let keys = match keys {
            Some(k) => k,
            None => {
                let nby = self.spec.right_by.len();
                let (ts, by) = (batch.column(p + nby), &batch.columns()[p..p + nby]);
                Arc::new(RKeys { keys: Keys::new(&*self.g, ts, by, None)?, g: Arc::clone(&self.g) })
            }
        };
        let payload = batch.project(&(0..p).collect::<Vec<_>>())?;
        let size = payload.get_array_memory_size() + 24 * keys.keys.len;
        if res.try_grow(size).is_err() {
            if let MemoryLimit::Finite(limit) = pool.memory_limit()
                && buffered + size > limit
            {
                return resources_err!(
                    "AsofExec: the right rows within reach of one left batch ({} bytes) exceed the memory \
                     limit ({limit} bytes); raise the limit or lower the batch size",
                    buffered + size
                );
            }
            res.grow(size);
        }
        Ok(RBatch { payload, keys, _mem: Reserved { res: Arc::clone(res), size } })
    }

    /// Position of the last buffered row `<= t` (`(0, 0)` if none).
    fn predecessor(&self, buf: &VecDeque<Arc<RBatch<G>>>, lk: &Keys<G>, t: Target) -> (usize, usize) {
        let (b, r) = self.upper(buf, lk, t);
        if r > 0 {
            (b, r - 1)
        } else if b > 0 {
            (b - 1, buf[b - 1].keys.keys.len - 1)
        } else {
            (0, 0)
        }
    }

    /// Position of the first buffered row `> t` (`(len, 0)` if none).
    fn upper(&self, buf: &VecDeque<Arc<RBatch<G>>>, lk: &Keys<G>, t: Target) -> (usize, usize) {
        let g = &*self.g;
        // First batch whose last row is > t.
        let b = buf.partition_point(|b| le(g, &b.keys.keys, b.keys.keys.len - 1, lk, t));
        if b == buf.len() {
            return (b, 0);
        }
        let k = &buf[b].keys.keys;
        (b, gallop_by(k.len, 0, |j| le(g, k, j, lk, t)))
    }
}

/// One left batch: run every cursor over the right slices, gather output.
fn work_batch<G: GroupKind>(
    g: &G,
    spec: &AsofSpec,
    out_schema: &SchemaRef,
    payload_schema: &SchemaRef,
    left: &RecordBatch,
    lk: &Keys<G>,
    slices: &[RSlice<G>],
) -> Result<RecordBatch> {
    let (n, strict) = (lk.len, spec.strict);
    let starts: Vec<usize> = slices.iter().map(|s| s.start).collect();
    let ends: Vec<usize> = slices.iter().map(|s| s.end).collect();
    let valid = |i: usize| lk.is_valid(i);
    let rp: Option<Vec<&[i128]>> = slices.iter().map(|s| s.b.keys.keys.packed.as_deref()).collect();
    let mut refs = vec![vec![NONE; n]; spec.shifts.len()];
    match (&lk.packed, &rp) {
        // Packed keys: plain i128 comparisons.
        (Some(lp), Some(rp)) => scan(
            &starts,
            &ends,
            &spec.shifts,
            valid,
            |i, h| packed_threshold(lp[i], h, strict),
            |s, j, t| rp[s][j] <= t,
            |c, i| packed_group(rp[ref_src(c)][ref_row(c)]) == packed_group(lp[i]),
            &mut refs,
        ),
        _ => scan(
            &starts,
            &ends,
            &spec.shifts,
            valid,
            |i, h| target(lk, i, h, strict),
            |s, j, t| le(g, &slices[s].b.keys.keys, j, lk, t),
            |c, i| same_group(g, slices, c, lk, i),
            &mut refs,
        ),
    }
    let batches: Vec<&RecordBatch> = slices.iter().map(|s| &s.b.payload).collect();
    let mut cols: Vec<ArrayRef> = left.columns().to_vec();
    let mut gathers: Vec<Option<Gather>> = refs.iter().map(|_| None).collect();
    for (k, p, _) in &spec.right_out {
        let gather = gathers[*k].get_or_insert_with(|| Gather::new(&refs[*k]));
        cols.push(gather.column(&batches, *p, payload_schema.field(*p).data_type())?);
    }
    let opts = RecordBatchOptions::new().with_row_count(Some(n));
    Ok(RecordBatch::try_new_with_options(Arc::clone(out_schema), cols, &opts)?)
}

/// One cursor per offset over the slices (`starts..ends` of each): for
/// every valid left row, in order, the last right row `le` its target, kept
/// in `refs` if it is in the row's group (`same`). Rows outer, offsets
/// inner: the cursors move together through a small window of the right
/// rows.
#[allow(clippy::too_many_arguments)]
#[inline]
fn scan<T: Copy>(
    starts: &[usize],
    ends: &[usize],
    shifts: &[i64],
    valid: impl Fn(usize) -> bool,
    target: impl Fn(usize, i64) -> T,
    le: impl Fn(usize, usize, T) -> bool,
    same: impl Fn(u64, usize) -> bool,
    refs: &mut [Vec<u64>],
) {
    let Some(&first) = starts.first() else {
        return;
    };
    // Per offset: (slice, position, candidate).
    let mut cur = vec![(0usize, first, NONE); shifts.len()];
    for i in 0..refs.first().map_or(0, |r| r.len()) {
        if !valid(i) {
            continue;
        }
        for ((c, &h), out) in cur.iter_mut().zip(shifts).zip(refs.iter_mut()) {
            let t = target(i, h);
            let (mut s, mut pos, mut cand) = *c;
            loop {
                let end = ends[s];
                let e = if pos < end && le(s, end - 1, t) { end } else { gallop_by(end, pos, |j| le(s, j, t)) };
                if e > pos {
                    cand = make_ref(s as u32, e - 1);
                }
                pos = e;
                if e < end || s + 1 == starts.len() {
                    break;
                }
                s += 1;
                pos = starts[s];
            }
            *c = (s, pos, cand);
            if cand != NONE && same(cand, i) {
                out[i] = cand;
            }
        }
    }
}

#[inline]
fn same_group<G: GroupKind>(g: &G, slices: &[RSlice<G>], cand: u64, lk: &Keys<G>, i: usize) -> bool {
    let rk = &slices[ref_src(cand)].b.keys.keys;
    let j = ref_row(cand);
    match (&rk.packed, &lk.packed) {
        (Some(r), Some(l)) => packed_group(r[j]) == packed_group(l[i]),
        _ => g.cmp_at(&rk.grp, j, &lk.grp, i) == Ordering::Equal,
    }
}
