//! Ordered k-way merge of an input's partitions.
//!
//! Every partition is executed exactly once, concurrently, by a producer task
//! that prepares its batches (evaluates sort keys, optionally checks the
//! order) and sends them over a bounded channel. The merge then emits one
//! ordered stream:
//!
//! * **batch fast path**: when the stream with the smallest head can run past
//!   every other stream's head, the run is found with one comparison (the
//!   whole batch) or a gallop (a prefix), and passed on as a zero-copy slice;
//! * **row level** only where streams actually overlap: short runs from
//!   several streams are gathered with `interleave` into a batch of up to
//!   `batch_size` rows.
//!
//! Disjoint ranges (a sorted file split into byte ranges) therefore cost one
//! comparison per batch. Overlapping streams (round-robin, unions) advance
//! together. With disjoint ranges the merge drains one range at a time, so a
//! later range only runs ahead as far as its buffer: two batches, plus a
//! read-ahead budget shared by all partitions (`DATAFUSION_PLUGINS_READAHEAD_MB`,
//! default 512 MB per input). Many small ranges (a high `target_partitions`
//! where the input is planned) let several later ranges decode completely
//! within the budget while an earlier one drains, which restores upstream
//! parallelism; memory stays bounded by the budget.

use std::cmp::Ordering;
use std::fmt;
use std::sync::Arc;

use arrow::array::RecordBatch;
use arrow::compute::interleave_record_batch;
use arrow::datatypes::SchemaRef;
use arrow::row::{RowConverter, Rows, SortField};
use datafusion::common::tree_node::TreeNodeRecursion;
use datafusion::common::{Result, exec_err, plan_err};
use datafusion::execution::TaskContext;
use datafusion::physical_expr::{LexOrdering, PhysicalExpr};
use datafusion::physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion::physical_plan::stream::RecordBatchReceiverStreamBuilder;
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, ExecutionPlanProperties, Partitioning,
    PlanProperties, ReplaceChildrenOptions, SendableRecordBatchStream,
};
use futures::StreamExt;
use tokio::sync::mpsc;

use crate::kernel::gallop_by;

/// Sort keys of one batch, comparable across batches of the same input.
pub(crate) trait MergeKeys: Send + Sync + 'static {
    fn len(&self) -> usize;
    fn cmp_at(&self, i: usize, other: &Self, j: usize) -> Ordering;
    /// True if the keys are non-decreasing.
    fn is_sorted(&self) -> bool {
        (1..self.len()).all(|i| self.cmp_at(i - 1, self, i) != Ordering::Greater)
    }
}

impl MergeKeys for Rows {
    fn len(&self) -> usize {
        self.num_rows()
    }
    fn cmp_at(&self, i: usize, other: &Self, j: usize) -> Ordering {
        self.row(i).cmp(&other.row(j))
    }
}

/// A prepared batch and its sort keys.
pub(crate) struct Prepared<K> {
    pub batch: RecordBatch,
    pub keys: Arc<K>,
}

/// One output of the merge: a batch, and its keys when it is an unchanged
/// input batch (`None` for slices and gathers).
pub(crate) struct Merged<K> {
    pub batch: RecordBatch,
    pub keys: Option<Arc<K>>,
}

pub(crate) type Prep<K> = Arc<dyn Fn(RecordBatch) -> Result<Option<Prepared<K>>> + Send + Sync>;

/// How the producers check the order of each partition.
#[derive(Clone, Debug)]
pub(crate) struct OrderCheck {
    /// e.g. "left input is not sorted by (g, ts)".
    pub message: String,
    /// Check inside every batch (vectorised), not only at batch boundaries.
    pub in_batch: bool,
    /// Appended to the error.
    pub hint: String,
}

/// Batches each partition may queue regardless of the shared budget, so the
/// stream the merge is waiting on never blocks on it.
const FREE: usize = 2;

/// Read-ahead of one input beyond `FREE` batches per partition, split into
/// equal quotas among the partitions allowed to read ahead: lets later
/// disjoint ranges decode while an earlier one drains.
/// `DATAFUSION_PLUGINS_READAHEAD_MB` (default 512).
fn readahead_bytes() -> usize {
    let mb = std::env::var("DATAFUSION_PLUGINS_READAHEAD_MB")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(512.0);
    ((mb * 1024.0 * 1024.0) as usize).max(1)
}

/// A batch and its size in bytes.
type Item<K> = (Result<Prepared<K>>, usize);

/// What a partition has queued, shared by its producer and the merge.
#[derive(Default)]
struct Queue {
    batches: std::sync::atomic::AtomicUsize,
    bytes: std::sync::atomic::AtomicUsize,
    /// Wakes the producer when the merge takes a batch.
    notify: tokio::sync::Notify,
}

struct Input<K> {
    rx: mpsc::UnboundedReceiver<Item<K>>,
    queue: Arc<Queue>,
    cur: Option<(Prepared<K>, usize)>,
    done: bool,
}

/// The merged stream of all partitions of a plan.
pub(crate) struct MergedStream<K: MergeKeys> {
    inputs: Vec<Input<K>>,
    batch_size: usize,
    _tasks: tokio::task::JoinSet<()>,

}

/// Execute every partition of `plan` once and merge them in key order.
pub(crate) fn merged<K: MergeKeys>(
    plan: &Arc<dyn ExecutionPlan>,
    ctx: &Arc<TaskContext>,
    prep: Prep<K>,
    check: OrderCheck,
) -> Result<MergedStream<K>> {
    let n = plan.output_partitioning().partition_count();
    let mut tasks = tokio::task::JoinSet::new();
    let mut inputs = Vec::with_capacity(n);
    // A partition reads ahead (up to its quota) while fewer than `window`
    // lower-indexed partitions are still producing: with disjoint ranges that
    // keeps `window` ranges decoding, nearest first, each with room to
    // buffer while the merge drains the ones before it.
    // Twice the threads, so I/O waits inside a decode don't idle the CPU.
    let window = 2 * ctx.session_config().target_partitions().max(1);
    let quota = readahead_bytes() / window;
    let finished: Arc<Vec<std::sync::atomic::AtomicBool>> =
        Arc::new((0..n).map(|_| std::sync::atomic::AtomicBool::new(false)).collect());
    let (done_tx, done_rx) = tokio::sync::watch::channel(0usize);
    let done_tx = Arc::new(done_tx);
    for q in 0..n {
        let (tx, rx) = mpsc::unbounded_channel::<Item<K>>();
        let queue = Arc::new(Queue::default());
        let queue_p = Arc::clone(&queue);
        let mut done_p = done_rx.clone();
        let (finished_p, done_tx_p) = (Arc::clone(&finished), Arc::clone(&done_tx));
        let mut stream = plan.execute(q, Arc::clone(ctx))?;
        let (prep, check) = (Arc::clone(&prep), check.clone());
        tasks.spawn(async move {
            // Marks this producer finished (on any exit) and wakes the others.
            struct Finish(Arc<Vec<std::sync::atomic::AtomicBool>>, usize, Arc<tokio::sync::watch::Sender<usize>>);
            impl Drop for Finish {
                fn drop(&mut self) {
                    self.0[self.1].store(true, std::sync::atomic::Ordering::SeqCst);
                    self.2.send_modify(|v| *v += 1);
                }
            }
            let _finish = Finish(Arc::clone(&finished_p), q, done_tx_p);
            let mut last: Option<(Arc<K>, usize)> = None;
            let trace = std::env::var_os("DATAFUSION_PLUGINS_TRACE").is_some();
            let (t0, mut t_dec, mut t_wait, mut nb) = (std::time::Instant::now(), 0f64, 0f64, 0usize);
            let mut t = std::time::Instant::now();
            while let Some(b) = stream.next().await {
                t_dec += t.elapsed().as_secs_f64();
                nb += 1;
                let item = b.and_then(|b| prep(b)).and_then(|p| match p {
                    Some(p) if p.keys.len() > 0 => {
                        let k = &p.keys;
                        if let Some((lk, li)) = &last
                            && lk.cmp_at(*li, k, 0) == Ordering::Greater
                        {
                            return exec_err!(
                                "{} (a batch starts before the previous one ended){}",
                                check.message,
                                check.hint
                            );
                        }
                        if check.in_batch && !k.is_sorted() {
                            return exec_err!("{}{}", check.message, check.hint);
                        }
                        last = Some((Arc::clone(k), k.len() - 1));
                        Ok(Some(p))
                    }
                    _ => Ok(None),
                });
                let item = match item {
                    Ok(None) => continue,
                    Ok(Some(p)) => Ok(p),
                    Err(e) => Err(e),
                };
                let failed = item.is_err();
                use std::sync::atomic::Ordering::SeqCst;
                let size = item.as_ref().map(|p| p.batch.get_array_memory_size()).unwrap_or(0);
                let tw = std::time::Instant::now();
                t_wait -= tw.duration_since(t).as_secs_f64(); // (prep counted as decode)
                t_dec += tw.duration_since(t).as_secs_f64();
                t = tw;
                // A free slot, or room in the quota while in the window. The
                // merge frees slots of the stream it is waiting on, so that
                // stream never waits on the others.
                loop {
                    let (n_q, b_q) = (queue_p.batches.load(SeqCst), queue_p.bytes.load(SeqCst));
                    let in_window = finished_p[..q].iter().filter(|f| !f.load(SeqCst)).count() < window;
                    if n_q < FREE || (in_window && b_q + size <= quota) {
                        queue_p.batches.fetch_add(1, SeqCst);
                        queue_p.bytes.fetch_add(size, SeqCst);
                        break;
                    }
                    tokio::select! {
                        _ = queue_p.notify.notified() => {}
                        r = done_p.changed() => {
                            if r.is_err() {
                                return;
                            }
                        }
                    }
                }
                if tx.send((item, size)).is_err() || failed {
                    return;
                }
                t_wait += t.elapsed().as_secs_f64();
                t = std::time::Instant::now();
            }
            if trace {
                eprintln!(
                    "[merge] partition {q:>3}: {nb:>5} batches, decode {t_dec:.3}s, wait {t_wait:.3}s, done at {:.3}s",
                    t0.elapsed().as_secs_f64()
                );
            }
        });
        inputs.push(Input {
            rx,
            queue,
            cur: None,
            done: false,
        });
    }
    Ok(MergedStream {
        inputs,
        batch_size: ctx.session_config().batch_size().max(1),
        _tasks: tasks,
    })
}

impl<K: MergeKeys> MergedStream<K> {
    /// Head of input `s`: (keys, position).
    fn head(&self, s: usize) -> (&K, usize) {
        let (p, pos) = self.inputs[s].cur.as_ref().expect("live input");
        (&p.keys, *pos)
    }

    /// Does input `a`'s head come before input `b`'s (ties: lower index)?
    fn before(&self, a: usize, b: usize) -> bool {
        let ((ka, pa), (kb, pb)) = (self.head(a), self.head(b));
        match ka.cmp_at(pa, kb, pb) {
            Ordering::Less => true,
            Ordering::Greater => false,
            Ordering::Equal => a < b,
        }
    }

    /// The next merged batch, or `None` at the end.
    pub(crate) async fn next(&mut self) -> Result<Option<Merged<K>>> {
        for inp in self.inputs.iter_mut() {
            while inp.cur.is_none() && !inp.done {
                match inp.rx.recv().await {
                    Some((item, size)) => {
                        use std::sync::atomic::Ordering::SeqCst;
                        inp.queue.batches.fetch_sub(1, SeqCst);
                        inp.queue.bytes.fetch_sub(size, SeqCst);
                        inp.queue.notify.notify_one();
                        inp.cur = Some((item?, 0));
                    }
                    None => inp.done = true,
                }
            }
        }
        let live: Vec<usize> = (0..self.inputs.len()).filter(|&s| self.inputs[s].cur.is_some()).collect();
        if live.is_empty() {
            return Ok(None);
        }
        let mut segs: Vec<(usize, RecordBatch, usize, usize)> = Vec::new(); // (input, batch, start, len)
        let mut total = 0usize;
        loop {
            // The input with the smallest head, and the runner-up.
            let mut s = live[0];
            let mut o: Option<usize> = None;
            for &t in &live[1..] {
                if self.before(t, s) {
                    o = Some(s);
                    s = t;
                } else if o.is_none_or(|u| self.before(t, u)) {
                    o = Some(t);
                }
            }
            let (keys, pos) = self.head(s);
            let n = keys.len();
            let end = match o {
                None => n,
                Some(o) => {
                    let (ko, po) = self.head(o);
                    let ok = |j: usize| match keys.cmp_at(j, ko, po) {
                        Ordering::Less => true,
                        Ordering::Equal => s < o,
                        Ordering::Greater => false,
                    };
                    if ok(n - 1) { n } else { gallop_by(n, pos, ok) }
                }
            };
            let (p, cur) = self.inputs[s].cur.as_mut().expect("live input");
            segs.push((s, p.batch.clone(), pos, end - pos));
            total += end - pos;
            *cur = end;
            if end == n {
                // Whole remaining batch: unchanged when it starts at row 0.
                let (p, _) = self.inputs[s].cur.take().expect("live input");
                if segs.len() == 1 && pos == 0 {
                    return Ok(Some(Merged {
                        batch: p.batch,
                        keys: Some(p.keys),
                    }));
                }
                break;
            }
            if total >= self.batch_size {
                break;
            }
        }
        if let [(_, b, start, len)] = segs.as_slice() {
            return Ok(Some(Merged {
                batch: b.slice(*start, *len),
                keys: None,
            }));
        }
        let batches: Vec<&RecordBatch> = segs.iter().map(|s| &s.1).collect();
        let idx: Vec<(usize, usize)> = segs
            .iter()
            .enumerate()
            .flat_map(|(k, s)| (s.2..s.2 + s.3).map(move |r| (k, r)))
            .collect();
        Ok(Some(Merged {
            batch: interleave_record_batch(&batches, &idx)?,
            keys: None,
        }))
    }
}

/// Merge the (each sorted) partitions of an input into one ordered partition,
/// like `SortPreservingMergeExec`, with the batch-level fast path above.
#[derive(Debug)]
pub struct SortedMergeExec {
    input: Arc<dyn ExecutionPlan>,
    ordering: LexOrdering,
    cache: Arc<PlanProperties>,
}

impl SortedMergeExec {
    pub fn new(input: Arc<dyn ExecutionPlan>, ordering: LexOrdering) -> Self {
        let eq = input.equivalence_properties().clone();
        let cache = PlanProperties::new(
            eq,
            Partitioning::UnknownPartitioning(1),
            EmissionType::Incremental,
            Boundedness::Bounded,
        );
        Self {
            input,
            ordering,
            cache: Arc::new(cache),
        }
    }
}

impl DisplayAs for SortedMergeExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "SortedMergeExec: [{}]", self.ordering)
    }
}

/// Sort keys of `ordering` as Arrow rows.
pub(crate) fn row_prep(ordering: &LexOrdering, schema: &SchemaRef) -> Result<Prep<Rows>> {
    let fields = ordering
        .iter()
        .map(|e| Ok(SortField::new_with_options(e.expr.data_type(schema)?, e.options)))
        .collect::<Result<Vec<_>>>()?;
    let conv = Arc::new(RowConverter::new(fields)?);
    let exprs: Vec<Arc<dyn PhysicalExpr>> = ordering.iter().map(|e| Arc::clone(&e.expr)).collect();
    Ok(Arc::new(move |batch: RecordBatch| {
        if batch.num_rows() == 0 {
            return Ok(None);
        }
        let cols = exprs
            .iter()
            .map(|e| e.evaluate(&batch)?.into_array(batch.num_rows()))
            .collect::<Result<Vec<_>>>()?;
        let keys = Arc::new(conv.convert_columns(&cols)?);
        Ok(Some(Prepared { batch, keys }))
    }))
}

impl ExecutionPlan for SortedMergeExec {
    fn name(&self) -> &str {
        "SortedMergeExec"
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.cache
    }

    fn maintains_input_order(&self) -> Vec<bool> {
        vec![true]
    }

    fn benefits_from_input_partitioning(&self) -> Vec<bool> {
        vec![false]
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }

    fn apply_expressions(
        &self,
        _f: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> Result<TreeNodeRecursion>,
    ) -> Result<TreeNodeRecursion> {
        Ok(TreeNodeRecursion::Continue)
    }

    fn replace_children(
        self: Arc<Self>,
        mut children: Vec<Arc<dyn ExecutionPlan>>,
        _options: ReplaceChildrenOptions,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        if children.len() != 1 {
            return plan_err!("SortedMergeExec expects one child");
        }
        Ok(Arc::new(Self::new(children.swap_remove(0), self.ordering.clone())))
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
        if partition != 0 {
            return exec_err!("SortedMergeExec has one partition");
        }
        let prep = row_prep(&self.ordering, &self.input.schema())?;
        let check = OrderCheck {
            message: format!("input is not sorted by [{}]", self.ordering),
            in_batch: false,
            hint: String::new(),
        };
        let mut m = merged(&self.input, &context, prep, check)?;
        let mut builder = RecordBatchReceiverStreamBuilder::new(self.schema(), 2);
        let tx = builder.tx();
        builder.spawn(async move {
            while let Some(b) = m.next().await? {
                if tx.send(Ok(b.batch)).await.is_err() {
                    break;
                }
            }
            Ok(())
        });
        Ok(builder.build())
    }
}
