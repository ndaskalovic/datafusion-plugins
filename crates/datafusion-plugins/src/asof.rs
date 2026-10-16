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
//!    partition sorted by `(by..., on)`; DataFusion's own optimizer adds a
//!    sort only where that is not provable. With `assume_sorted` the order is
//!    trusted and checked while streaming.
//! 2. **Merge.** Each side's partitions are merged into one ordered stream
//!    (see `merge`).
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

use std::cmp::Ordering;
use std::collections::{BTreeMap, VecDeque};
use std::fmt;
use std::sync::Arc;

use arrow::array::{ArrayRef, RecordBatch, RecordBatchOptions};
use arrow::compute::SortOptions;
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use arrow::row::{RowConverter, Rows, SortField};
use datafusion::common::config::ConfigOptions;
use datafusion::common::tree_node::TreeNodeRecursion;
use datafusion::common::{DataFusionError, Result, exec_err, internal_datafusion_err, plan_err};
use datafusion::execution::TaskContext;
use datafusion::physical_expr::expressions::Column;
use datafusion::physical_expr::{
    EquivalenceProperties, LexOrdering, OrderingRequirements, PhysicalExpr, PhysicalExprRef,
    PhysicalSortExpr,
};
use datafusion::physical_expr_common::physical_expr::fmt_sql;
use datafusion::physical_optimizer::PhysicalOptimizerRule;
use datafusion::physical_optimizer::ensure_requirements::EnsureRequirements;
use datafusion::physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion::physical_plan::metrics::{BaselineMetrics, ExecutionPlanMetricsSet, MetricsSet};
use datafusion::physical_plan::projection::ProjectionExec;
use datafusion::physical_plan::stream::RecordBatchReceiverStreamBuilder;
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, Distribution, ExecutionPlan, ExecutionPlanProperties,
    InputDistributionRequirements, Partitioning, PlanProperties, ReplaceChildrenOptions,
    SendableRecordBatchStream, apply_expression_roots,
};
use tokio::sync::mpsc::Sender;

use crate::kernel::{NONE, eval_side, gallop_by, left_keys, make_ref, ref_row, right_keys, take_refs};
use crate::keys::{
    GroupKind, Keys, NoGroup, PrimGroup, RowGroup, TsNative, is_i32_like, is_primitive_group_type,
    pack_key, packed_group, packed_ts,
};
use crate::merge::{MergeKeys, Merged, MergedStream, OrderCheck, Prep, Prepared, merged};

/// What the planners give the exec.
#[derive(Debug, Clone)]
pub struct AsofSpec {
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
    /// NULLS FIRST per side (null keys never match; this only has to agree
    /// with the inputs' order).
    pub nulls_first: (bool, bool),
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

/// `Some(nulls_first)` if `eq` satisfies an ascending order on `exprs`.
pub(crate) fn nulls_first_of(
    eq: &EquivalenceProperties,
    exprs: &[PhysicalExprRef],
) -> Result<Option<bool>> {
    for nf in [true, false] {
        if eq.ordering_satisfy(sort_exprs(exprs, nf))? {
            return Ok(Some(nf));
        }
    }
    Ok(None)
}

fn key_desc(exprs: &[PhysicalExprRef]) -> String {
    let names: Vec<String> = exprs.iter().map(|e| fmt_sql(e.as_ref()).to_string()).collect();
    format!("({})", names.join(", "))
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
    let (lk, rk) = (spec.left_keys(), spec.right_keys());
    // Keep a NULLS placement the input already has, so nothing is re-sorted.
    spec.nulls_first = (
        nulls_first_of(left.equivalence_properties(), &lk)?.unwrap_or(true),
        nulls_first_of(right.equivalence_properties(), &rk)?.unwrap_or(true),
    );
    let exec: Arc<dyn ExecutionPlan> = Arc::new(AsofExec::try_new(left, right, spec.clone())?);
    let exec = if spec.assume_sorted {
        exec
    } else {
        EnsureRequirements::new().optimize(exec, config)?
    };
    let schema = exec.schema();
    let exprs = outputs
        .iter()
        .map(|(i, name)| {
            (
                Arc::new(Column::new(schema.field(*i).name(), *i)) as Arc<dyn PhysicalExpr>,
                name.clone(),
            )
        })
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
pub struct AsofExec {
    left: Arc<dyn ExecutionPlan>,
    right: Arc<dyn ExecutionPlan>,
    spec: Arc<AsofSpec>,
    group: GroupSpec,
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
        if spec.shifts.is_empty() || spec.left_by.len() != spec.right_by.len() {
            return plan_err!("AsofExec: invalid spec");
        }
        let ts = spec.left_on.data_type(&ls)?;
        if spec.right_on.data_type(&rs)? != ts {
            return plan_err!("AsofExec: 'on' expressions have different types");
        }
        let mut by = Vec::new();
        for (l, r) in spec.left_by.iter().zip(&spec.right_by) {
            let t = l.data_type(&ls)?;
            if r.data_type(&rs)? != t {
                return plan_err!("AsofExec: 'by' expressions have different types");
            }
            by.push(t);
        }
        let group = match by.as_slice() {
            [] => GroupSpec::None,
            [t] if is_i32_like(t) => GroupSpec::I32,
            [t] if is_primitive_group_type(t) => GroupSpec::I64,
            ts if RowGroup::supports(ts) => GroupSpec::Rows(ts.to_vec()),
            _ => return plan_err!("AsofExec: unsupported 'by' types"),
        };
        let payload_schema = Arc::new(rs.project(&spec.payload_idx)?);
        let mut fields: Vec<Field> = ls.fields().iter().map(|f| f.as_ref().clone()).collect();
        for (_, p, name) in &spec.right_out {
            fields.push(payload_schema.field(*p).clone().with_name(name.clone()).with_nullable(true));
        }
        let out_schema = Arc::new(Schema::new(fields));
        // Output = left order: the left columns keep their positions.
        let lex = LexOrdering::new(sort_exprs(&spec.left_keys(), spec.nulls_first.0))
            .ok_or_else(|| internal_datafusion_err!("no keys"))?;
        let eq = EquivalenceProperties::new_with_orderings(Arc::clone(&out_schema), [lex]);
        let cache = PlanProperties::new(
            eq,
            Partitioning::UnknownPartitioning(1),
            EmissionType::Incremental,
            Boundedness::Bounded,
        );
        Ok(Self {
            left,
            right,
            spec: Arc::new(spec),
            group,
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
        let on = self
            .spec
            .left_by
            .iter()
            .zip(&self.spec.right_by)
            .map(|(l, r)| format!("({} = {})", fmt_sql(l.as_ref()), fmt_sql(r.as_ref())))
            .collect::<Vec<_>>()
            .join(", ");
        let op = if self.spec.strict { ">" } else { ">=" };
        write!(
            f,
            "AsofExec: on=[{on}], match=[{} {op} {} + h], h={:?}{}",
            fmt_sql(self.spec.left_on.as_ref()),
            fmt_sql(self.spec.right_on.as_ref()),
            self.spec.shifts,
            if self.spec.assume_sorted { ", assume_sorted" } else { "" }
        )
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
        if self.spec.assume_sorted {
            return vec![None, None];
        }
        let req = |k: Vec<PhysicalExprRef>, nf: bool| {
            LexOrdering::new(sort_exprs(&k, nf)).map(OrderingRequirements::from)
        };
        vec![
            req(self.spec.left_keys(), self.spec.nulls_first.0),
            req(self.spec.right_keys(), self.spec.nulls_first.1),
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

    fn replace_children(
        self: Arc<Self>,
        mut children: Vec<Arc<dyn ExecutionPlan>>,
        _options: ReplaceChildrenOptions,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        if children.len() != 2 {
            return plan_err!("AsofExec expects two children");
        }
        let right = children.swap_remove(1);
        let left = children.swap_remove(0);
        Ok(Arc::new(Self::try_new(left, right, self.spec.as_ref().clone())?))
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

    fn execute(&self, partition: usize, ctx: Arc<TaskContext>) -> Result<SendableRecordBatchStream> {
        if partition != 0 {
            return exec_err!("AsofExec has one output partition, got {partition}");
        }
        match &self.group {
            GroupSpec::None => self.start(NoGroup, ctx),
            GroupSpec::I32 => self.start(PrimGroup::<i32>::new(), ctx),
            GroupSpec::I64 => self.start(PrimGroup::<i64>::new(), ctx),
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

/// Right keys (no NULLs) of one batch, comparable in `(by, on)` order.
struct RKeys<G: GroupKind> {
    keys: Keys<i64, G>,
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

/// A right batch in the buffer: payload columns and keys.
struct RBatch<G: GroupKind> {
    payload: RecordBatch,
    keys: Arc<RKeys<G>>,
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

fn target<G: GroupKind>(lk: &Keys<i64, G>, i: usize, shift: i64, strict: bool) -> Target {
    match &lk.packed {
        Some(p) => Target::Packed(packed_threshold(p[i], shift, strict)),
        None => Target::Row {
            i,
            ts: lk.ts.as_slice()[i] as i128 + shift as i128 - strict as i128,
        },
    }
}

/// Is right row `j` of `rk` `<=` target `t` (of left keys `lk`)?
#[inline]
fn le<G: GroupKind>(g: &G, rk: &Keys<i64, G>, j: usize, lk: &Keys<i64, G>, t: Target) -> bool {
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

type WorkResult = (u64, Result<RecordBatch>);

impl<G: GroupKind> Driver<G> {
    fn left_prep(&self, check: OrderCheck) -> Result<(Prep<Rows>, OrderCheck)> {
        let ls = self.left.schema();
        let keys = self.spec.left_keys();
        let fields = keys
            .iter()
            .map(|e| {
                Ok(SortField::new_with_options(
                    e.data_type(&ls)?,
                    SortOptions {
                        descending: false,
                        nulls_first: self.spec.nulls_first.0,
                    },
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        let conv = Arc::new(RowConverter::new(fields)?);
        let prep: Prep<Rows> = Arc::new(move |batch: RecordBatch| {
            if batch.num_rows() == 0 {
                return Ok(None);
            }
            let cols = keys
                .iter()
                .map(|e| e.evaluate(&batch)?.into_array(batch.num_rows()))
                .collect::<Result<Vec<_>>>()?;
            let keys = Arc::new(conv.convert_columns(&cols)?);
            Ok(Some(Prepared { batch, keys }))
        });
        Ok((prep, check))
    }

    /// Right batches keep the payload columns, then the key columns (so a
    /// merged slice can be re-keyed): `[payload..., by..., on]`.
    fn right_prep(&self) -> Result<Prep<RKeys<G>>> {
        let (g, spec) = (Arc::clone(&self.g), Arc::clone(&self.spec));
        let col = |e: &PhysicalExprRef| {
            e.downcast_ref::<Column>()
                .map(|c| c.index())
                .ok_or_else(|| internal_datafusion_err!("AsofExec: right keys must be columns"))
        };
        let mut idx = spec.payload_idx.clone();
        for e in spec.right_by.iter().chain([&spec.right_on]) {
            idx.push(col(e)?);
        }
        Ok(Arc::new(move |batch: RecordBatch| {
            let Some((keys, payload)) =
                right_keys::<i64, G>(&g, &spec.right_on, &spec.right_by, &idx, &batch)?
            else {
                return Ok(None);
            };
            let keys = Arc::new(RKeys { keys, g: Arc::clone(&g) });
            Ok(Some(Prepared { batch: payload, keys }))
        }))
    }

    async fn run(self, ctx: Arc<TaskContext>) -> Result<()> {
        let s = &self.spec;
        let check = |side: &str, k: &[PhysicalExprRef]| OrderCheck {
            message: format!("{side} input is not sorted by {}", key_desc(k)),
            in_batch: s.assume_sorted && s.check_sortedness,
            hint: if s.assume_sorted {
                "; each input partition must arrive sorted with assume_sorted=True (sort it \
                 upstream, or pass assume_sorted=False to let the node sort)"
                    .to_string()
            } else {
                String::new()
            },
        };
        let (lprep, lcheck) = self.left_prep(check("left", &s.left_keys()))?;
        let mut lstream: MergedStream<Rows> = merged(&self.left, &ctx, lprep, lcheck)?;
        let mut rstream: MergedStream<RKeys<G>> =
            merged(&self.right, &ctx, self.right_prep()?, check("right", &s.right_keys()))?;

        let threads = ctx.session_config().target_partitions().max(1);
        let max_inflight = 2 * threads + 2;
        let h_min = *s.shifts.iter().min().expect("shifts");
        let h_max = *s.shifts.iter().max().expect("shifts");
        let mut buf: VecDeque<Arc<RBatch<G>>> = VecDeque::new();
        let mut right_done = false;
        let mut left_done = false;
        let mut seq: u64 = 0;
        let mut next_out: u64 = 0;
        let mut ready: BTreeMap<u64, RecordBatch> = BTreeMap::new();
        let mut work: tokio::task::JoinSet<WorkResult> = tokio::task::JoinSet::new();

        loop {
            if !left_done && work.len() < max_inflight {
                let Some(Merged { batch, .. }) = lstream.next().await? else {
                    left_done = true;
                    continue;
                };
                let lk = left_keys::<i64, G>(&self.g, &s.left_on, &s.left_by, &batch)?;
                let slices = match (lk.first_valid(), lk.last_valid()) {
                    (Some(f), Some(l)) => {
                        let (tmin, tmax) = (target(&lk, f, h_min, s.strict), target(&lk, l, h_max, s.strict));
                        // The buffer must reach past tmax (or the right end).
                        while !right_done
                            && buf.back().is_none_or(|b| {
                                le(&*self.g, &b.keys.keys, b.keys.keys.len - 1, &lk, tmax)
                            })
                        {
                            match rstream.next().await? {
                                Some(m) => {
                                    let keys = match m.keys {
                                        Some(k) => k,
                                        None => Arc::new(self.rekey(&m.batch)?),
                                    };
                                    let payload = m.batch.project(&(0..self.payload_schema.fields().len()).collect::<Vec<_>>())?;
                                    buf.push_back(Arc::new(RBatch { payload, keys }));
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
                        for _ in 0..sb {
                            buf.pop_front();
                        }
                        slices
                    }
                    _ => Vec::new(),
                };
                let (g, spec, os) = (Arc::clone(&self.g), Arc::clone(&self.spec), Arc::clone(&self.out_schema));
                let ps = Arc::clone(&self.payload_schema);
                let n = seq;
                work.spawn(async move { (n, work_batch(&*g, &spec, &os, &ps, &batch, &lk, &slices)) });
                seq += 1;
                continue;
            }
            // Collect a result; emit in order.
            let Some(res) = work.join_next().await else {
                break; // left done, nothing in flight
            };
            let (n, out) = res.map_err(|e| DataFusionError::External(Box::new(e)))?;
            ready.insert(n, out?);
            while let Some(b) = ready.remove(&next_out) {
                next_out += 1;
                let rows = b.num_rows();
                if self.tx.send(Ok(b)).await.is_err() {
                    return Ok(());
                }
                self.metrics.record_output(rows);
            }
        }
        self.metrics.done();
        Ok(())
    }

    /// Keys of a merged right batch that is a slice or a gather, from its
    /// key columns (after the payload).
    fn rekey(&self, batch: &RecordBatch) -> Result<RKeys<G>> {
        let p = self.payload_schema.fields().len();
        let nby = self.spec.right_by.len();
        let on: PhysicalExprRef = Arc::new(Column::new("on", p + nby));
        let by: Vec<PhysicalExprRef> =
            (0..nby).map(|k| Arc::new(Column::new("by", p + k)) as PhysicalExprRef).collect();
        let (ts, by, _) = eval_side(&on, &by, batch)?;
        let ts = <i64 as TsNative>::extract(&ts)?;
        let grp = self.g.extract(&by)?;
        let len = batch.num_rows();
        let packed = self.g.pack(&grp, ts.as_slice(), len);
        Ok(RKeys {
            keys: Keys { ts, grp, valid: None, len, packed },
            g: Arc::clone(&self.g),
        })
    }

    /// Position of the last buffered row `<= t` (`(0, 0)` if none).
    fn predecessor(&self, buf: &VecDeque<Arc<RBatch<G>>>, lk: &Keys<i64, G>, t: Target) -> (usize, usize) {
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
    fn upper(&self, buf: &VecDeque<Arc<RBatch<G>>>, lk: &Keys<i64, G>, t: Target) -> (usize, usize) {
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
    lk: &Keys<i64, G>,
    slices: &[RSlice<G>],
) -> Result<RecordBatch> {
    let n = lk.len;
    let nc = spec.shifts.len();
    let mut refs = vec![vec![NONE; n]; nc];
    // Cursor state: (slice, pos, candidate, candidate row's slice+row).
    let mut cur: Vec<(usize, usize, u64)> = vec![(0, slices.first().map_or(0, |s| s.start), NONE); nc];
    for i in 0..n {
        if !lk.is_valid(i) {
            continue;
        }
        for (c, &h) in spec.shifts.iter().enumerate() {
            let t = target(lk, i, h, spec.strict);
            let (mut s, mut pos, mut cand) = cur[c];
            while s < slices.len() {
                let sl = &slices[s];
                let rk = &sl.b.keys.keys;
                let end = sl.end;
                let e = if pos < end && le(g, rk, end - 1, lk, t) {
                    end
                } else {
                    gallop_by(end, pos, |j| le(g, rk, j, lk, t))
                };
                if e > pos {
                    cand = make_ref(s as u32, e - 1);
                }
                pos = e;
                if e < end {
                    break;
                }
                if s + 1 < slices.len() {
                    s += 1;
                    pos = slices[s].start;
                } else {
                    break;
                }
            }
            cur[c] = (s, pos, cand);
            refs[c][i] = if cand != NONE && same_group(g, slices, cand, lk, i) { cand } else { NONE };
        }
    }
    let batches: Vec<&RecordBatch> = slices.iter().map(|s| &s.b.payload).collect();
    let mut cols: Vec<ArrayRef> = left.columns().to_vec();
    for (k, p, _) in &spec.right_out {
        cols.push(take_refs(&batches, *p, payload_schema.field(*p).data_type(), &refs[*k])?);
    }
    let opts = RecordBatchOptions::new().with_row_count(Some(n));
    Ok(RecordBatch::try_new_with_options(Arc::clone(out_schema), cols, &opts)?)
}

#[inline]
fn same_group<G: GroupKind>(g: &G, slices: &[RSlice<G>], cand: u64, lk: &Keys<i64, G>, i: usize) -> bool {
    let rk = &slices[crate::kernel::ref_src(cand)].b.keys.keys;
    let j = ref_row(cand);
    match (&rk.packed, &lk.packed) {
        (Some(r), Some(l)) => packed_group(r[j]) == packed_group(l[i]),
        _ => g.cmp_at(&rk.grp, j, &lk.grp, i) == Ordering::Equal,
    }
}
