//! Ordered k-way merge of an input's partitions.
//!
//! Every partition is executed exactly once, concurrently, by a producer task
//! that prepares its batches (evaluates sort keys, optionally checks the
//! order) and sends them over a channel, then an explicit end marker: a
//! partition that stops without one (a panic, a cancelled task) fails the
//! merge instead of looking finished. The merge then emits one ordered
//! stream:
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
//! together. With disjoint ranges the merge drains one range at a time, so
//! later ranges only run ahead as far as their buffers allow.
//!
//! Memory: every batch a partition holds (queued or being merged) is
//! reserved from the memory pool. A partition holding nothing always gets
//! its next batch (the merge needs a head from every partition); up to
//! [`FREE`] batches double-buffer if `try_grow` allows; more is read-ahead
//! (see [`Budget`]). A failed `try_grow` makes the partition wait, never
//! fail.

use std::cmp::Ordering;
use std::collections::BTreeSet;
use std::fmt;
use std::panic::AssertUnwindSafe;
use std::sync::atomic::Ordering::SeqCst;
use std::sync::atomic::{AtomicBool, AtomicUsize};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use arrow::array::RecordBatch;
use arrow::compute::interleave_record_batch;
use arrow::datatypes::SchemaRef;
use arrow::row::{RowConverter, Rows, SortField};
use datafusion::common::tree_node::TreeNodeRecursion;
use datafusion::common::{DataFusionError, Result, exec_datafusion_err, exec_err, internal_err};
use datafusion::execution::TaskContext;
use datafusion::execution::memory_pool::{MemoryConsumer, MemoryReservation};
use datafusion::physical_expr::{LexOrdering, OrderingRequirements, PhysicalExpr};
use datafusion::physical_plan::stream::RecordBatchReceiverStreamBuilder;
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, ExecutionPlanProperties, Partitioning, PlanProperties,
    SendableRecordBatchStream,
};
use futures::{FutureExt, StreamExt};
use tokio::sync::{Notify, mpsc, watch};

use crate::kernel::gallop_by;

/// Sort keys of one batch, comparable across batches of the same input.
pub(crate) trait MergeKeys: Send + Sync + 'static {
    fn len(&self) -> usize;
    fn cmp_at(&self, i: usize, other: &Self, j: usize) -> Ordering;
    /// True if the keys are non-decreasing.
    fn is_sorted(&self) -> bool {
        (1..self.len()).all(|i| self.cmp_at(i - 1, self, i) != Ordering::Greater)
    }
    /// The first and last rows the order check compares across batch
    /// boundaries (`None`: no row takes part).
    fn bounds(&self) -> Option<(usize, usize)> {
        self.len().checked_sub(1).map(|last| (0, last))
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

/// Batches a partition may hold (queued or being merged) outside the
/// read-ahead budget.
const FREE: usize = 3;

/// Read-ahead memory of one merged input, beyond [`FREE`] batches per
/// partition. It comes from the memory pool, and its target is set by
/// feedback: it starts at a few batches per window partition and doubles
/// only while the merge is starved, i.e. waits on a partition while
/// partitions in the window wait on budget, so more read-ahead would buy
/// parallelism. Growth stops at the ceiling (see [`ceiling`]).
struct Budget {
    /// Everything the input's partitions hold.
    res: MemoryReservation,
    /// Read-ahead bytes queued beyond the free slots...
    extra: AtomicUsize,
    /// ...and their target (0 until the first batch sets the start).
    target: AtomicUsize,
    ceiling: usize,
    window: usize,
    /// Partitions waiting for read-ahead budget.
    blocked: Mutex<BTreeSet<usize>>,
    queues: Vec<Arc<Queue>>,
}

impl Budget {
    /// Try to take `size` bytes of read-ahead.
    fn take(&self, size: usize) -> bool {
        if self.target.load(SeqCst) == 0 {
            let start = (self.window * FREE * size).min(self.ceiling).max(1);
            let _ = self.target.compare_exchange(0, start, SeqCst, SeqCst);
        }
        if self.extra.load(SeqCst) + size > self.target.load(SeqCst) || self.res.try_grow(size).is_err() {
            return false;
        }
        self.extra.fetch_add(size, SeqCst);
        true
    }

    fn blocked(&self) -> MutexGuard<'_, BTreeSet<usize>> {
        self.blocked.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// `size` bytes of read-ahead were consumed: wake the nearest blocked
    /// partition (the one the merge needs soonest).
    fn release(&self, size: usize) {
        self.extra.fetch_sub(size, SeqCst);
        if let Some(&q) = self.blocked().first() {
            self.queues[q].notify.notify_one();
        }
    }

    /// The merge is waiting: if partitions wait on budget, grow and wake
    /// them.
    fn starved(&self) {
        let blocked = self.blocked();
        if blocked.is_empty() {
            return;
        }
        let t = self.target.load(SeqCst);
        let grown = t.saturating_mul(2).min(self.ceiling);
        if grown > t && self.target.compare_exchange(t, grown, SeqCst, SeqCst).is_ok() {
            for &q in blocked.iter() {
                self.queues[q].notify.notify_one();
            }
        }
    }
}

/// Physical memory of the machine, if known.
fn physical_memory() -> Option<usize> {
    // SAFETY: `sysconf` only reads system configuration values.
    let (pages, size) = unsafe { (libc::sysconf(libc::_SC_PHYS_PAGES), libc::sysconf(libc::_SC_PAGESIZE)) };
    (pages > 0 && size > 0).then(|| pages as usize * size as usize)
}

/// Ceiling of one input's read-ahead: an eighth of physical memory (so the
/// two inputs of a node stay under a quarter, even with an unbounded pool),
/// no more than `window` partitions' worth of the input's estimated size
/// (only that many read ahead at a time, so buffering more buys nothing),
/// and `cap`. A memory-limited pool caps it further through `try_grow`.
fn ceiling(plan: &Arc<dyn ExecutionPlan>, window: usize, cap: usize) -> usize {
    let ram = physical_memory().map_or(1 << 30, |m| m / 8);
    let stats = datafusion::physical_plan::StatisticsContext::new()
        .compute(plan.as_ref(), &datafusion::physical_plan::StatisticsArgs::new());
    let input = stats
        .ok()
        .and_then(|s| s.total_byte_size.get_value().copied())
        .filter(|b| *b > 0)
        .map_or(usize::MAX, |b| {
            let n = plan.output_partitioning().partition_count().max(1);
            b / n * window.min(n)
        });
    ram.min(input).min(cap).max(1)
}

/// What a producer sends.
enum Msg<K> {
    /// A batch, its size in bytes, and whether it holds read-ahead budget.
    Batch(Prepared<K>, usize, bool),
    Failed(DataFusionError),
    End,
}

/// What a partition holds, shared by its producer and the merge.
#[derive(Default)]
struct Queue {
    /// Batches queued or being merged.
    held: AtomicUsize,
    /// Wakes the producer: the merge took a batch, or budget was freed.
    notify: Notify,
}

struct Input<K> {
    rx: mpsc::UnboundedReceiver<Msg<K>>,
    queue: Arc<Queue>,
    /// The batch being merged, the position in it, and its size.
    cur: Option<(Prepared<K>, usize, usize)>,
    done: bool,
}

/// The merged stream of all partitions of a plan.
pub(crate) struct MergedStream<K: MergeKeys> {
    inputs: Vec<Input<K>>,
    batch_size: usize,
    budget: Arc<Budget>,
    _tasks: tokio::task::JoinSet<()>,
}

/// Execute every partition of `plan` once and merge them in key order.
/// `cap` bounds the read-ahead (bytes).
pub(crate) fn merged<K: MergeKeys>(
    plan: &Arc<dyn ExecutionPlan>,
    ctx: &Arc<TaskContext>,
    prep: Prep<K>,
    check: OrderCheck,
    cap: usize,
) -> Result<MergedStream<K>> {
    let n = plan.output_partitioning().partition_count();
    // A partition reads ahead on the budget while fewer than `window`
    // lower-indexed partitions are still producing: with disjoint ranges that
    // keeps `window` ranges decoding, nearest first, while the merge drains
    // the ones before. Twice the threads, so I/O waits don't idle the CPU.
    let window = 2 * ctx.session_config().target_partitions().max(1);
    let queues: Vec<Arc<Queue>> = (0..n).map(|_| Arc::default()).collect();
    let budget = Arc::new(Budget {
        res: MemoryConsumer::new("SortedMerge").register(ctx.memory_pool()),
        extra: AtomicUsize::new(0),
        target: AtomicUsize::new(0),
        ceiling: ceiling(plan, window, cap),
        window,
        blocked: Mutex::default(),
        queues: queues.clone(),
    });
    let finished: Arc<Vec<AtomicBool>> = Arc::new((0..n).map(|_| AtomicBool::new(false)).collect());
    let (done_tx, done_rx) = watch::channel(0usize);
    let done_tx = Arc::new(done_tx);
    let mut tasks = tokio::task::JoinSet::new();
    let mut inputs = Vec::with_capacity(n);
    for (q, queue) in queues.into_iter().enumerate() {
        let (tx, rx) = mpsc::unbounded_channel();
        let p = Producer {
            q,
            queue: Arc::clone(&queue),
            budget: Arc::clone(&budget),
            finished: Arc::clone(&finished),
            done: done_rx.clone(),
            prep: Arc::clone(&prep),
            check: check.clone(),
        };
        let stream = plan.execute(q, Arc::clone(ctx))?;
        let done_tx = Arc::clone(&done_tx);
        tasks.spawn(async move {
            let (finished, budget) = (Arc::clone(&p.finished), Arc::clone(&p.budget));
            let msg = match AssertUnwindSafe(p.run(stream, &tx)).catch_unwind().await {
                Ok(Ok(())) => Msg::End,
                Ok(Err(e)) => Msg::Failed(e),
                Err(panic) => {
                    let what = panic
                        .downcast_ref::<&str>()
                        .map(|s| s.to_string())
                        .or_else(|| panic.downcast_ref::<String>().cloned())
                        .unwrap_or_default();
                    Msg::Failed(exec_datafusion_err!("input partition {q} panicked: {what}"))
                }
            };
            let _ = tx.send(msg);
            finished[q].store(true, SeqCst);
            budget.blocked().remove(&q);
            done_tx.send_modify(|v| *v += 1);
        });
        inputs.push(Input { rx, queue, cur: None, done: false });
    }
    Ok(MergedStream {
        inputs,
        batch_size: ctx.session_config().batch_size().max(1),
        budget,
        _tasks: tasks,
    })
}

/// One partition's producer.
struct Producer<K> {
    q: usize,
    queue: Arc<Queue>,
    budget: Arc<Budget>,
    finished: Arc<Vec<AtomicBool>>,
    /// Changes when a partition finishes (the window moves).
    done: watch::Receiver<usize>,
    prep: Prep<K>,
    check: OrderCheck,
}

impl<K: MergeKeys> Producer<K> {
    /// Prepare, check and send every batch (until the merge is gone).
    async fn run(mut self, mut stream: SendableRecordBatchStream, tx: &mpsc::UnboundedSender<Msg<K>>) -> Result<()> {
        let check = self.check.clone();
        let mut last: Option<(Arc<K>, usize)> = None;
        while let Some(b) = stream.next().await {
            let Some(p) = (self.prep)(b?)? else { continue };
            if let Some((first, end)) = p.keys.bounds() {
                if let Some((lk, li)) = &last
                    && lk.cmp_at(*li, &p.keys, first) == Ordering::Greater
                {
                    return exec_err!(
                        "{} (a batch starts before the previous one ended){}",
                        check.message,
                        check.hint
                    );
                }
                if check.in_batch && !p.keys.is_sorted() {
                    return exec_err!("{}{}", check.message, check.hint);
                }
                last = Some((Arc::clone(&p.keys), end));
            }
            let size = p.batch.get_array_memory_size();
            let extra = self.reserve(size).await?;
            self.queue.held.fetch_add(1, SeqCst);
            if tx.send(Msg::Batch(p, size, extra)).is_err() {
                break;
            }
        }
        Ok(())
    }

    /// Reserve `size` bytes for the next batch, waiting while that doesn't
    /// fit; true if it is read-ahead budget.
    async fn reserve(&mut self, size: usize) -> Result<bool> {
        let (q, b) = (self.q, Arc::clone(&self.budget));
        loop {
            let held = self.queue.held.load(SeqCst);
            if held == 0 {
                // The merge needs a head from every partition: never wait.
                b.res.grow(size);
                return Ok(false);
            }
            let woken = self.queue.notify.notified();
            tokio::pin!(woken);
            woken.as_mut().enable();
            if held < FREE {
                if b.res.try_grow(size).is_ok() {
                    return Ok(false);
                }
            } else if self.finished[..q].iter().filter(|f| !f.load(SeqCst)).count() < b.window {
                if b.take(size) {
                    return Ok(true);
                }
                b.blocked().insert(q);
            }
            tokio::select! {
                _ = &mut woken => {}
                r = self.done.changed() => r.map_err(|_| exec_datafusion_err!("merge: producers gone"))?,
            }
            b.blocked().remove(&q);
        }
    }
}

impl<K: MergeKeys> MergedStream<K> {
    /// Head of input `s`: (keys, position).
    fn head(&self, s: usize) -> (&K, usize) {
        let (p, pos, _) = self.inputs[s].cur.as_ref().expect("live input");
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

    /// Input `s` is done with its current batch: free it.
    fn take_cur(&mut self, s: usize) -> Prepared<K> {
        let inp = &mut self.inputs[s];
        let (p, _, size) = inp.cur.take().expect("live input");
        self.budget.res.shrink(size);
        inp.queue.held.fetch_sub(1, SeqCst);
        inp.queue.notify.notify_one();
        p
    }

    /// Take a message from input `s`'s producer.
    fn accept(&mut self, s: usize, msg: Option<Msg<K>>) -> Result<()> {
        let inp = &mut self.inputs[s];
        match msg {
            Some(Msg::Batch(p, size, extra)) => {
                if extra {
                    self.budget.release(size);
                }
                inp.cur = Some((p, 0, size));
            }
            Some(Msg::End) => inp.done = true,
            Some(Msg::Failed(e)) => return Err(e),
            None => return internal_err!("input partition {s} stopped without finishing"),
        }
        Ok(())
    }

    /// The next merged batch, or `None` at the end.
    pub(crate) async fn next(&mut self) -> Result<Option<Merged<K>>> {
        for s in 0..self.inputs.len() {
            while self.inputs[s].cur.is_none() && !self.inputs[s].done {
                let inp = &mut self.inputs[s];
                let msg = match inp.rx.try_recv() {
                    Ok(m) => Some(m),
                    Err(mpsc::error::TryRecvError::Disconnected) => None,
                    Err(mpsc::error::TryRecvError::Empty) => {
                        // The merge has to wait for this partition.
                        self.budget.starved();
                        inp.rx.recv().await
                    }
                };
                self.accept(s, msg)?;
            }
        }
        let live: Vec<usize> = (0..self.inputs.len()).filter(|&s| self.inputs[s].cur.is_some()).collect();
        if live.is_empty() {
            return Ok(None);
        }
        let mut segs: Vec<(RecordBatch, usize, usize)> = Vec::new(); // (batch, start, len)
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
            let (p, cur, _) = self.inputs[s].cur.as_mut().expect("live input");
            segs.push((p.batch.clone(), pos, end - pos));
            total += end - pos;
            *cur = end;
            if end == n {
                // Whole remaining batch: unchanged when it starts at row 0.
                let p = self.take_cur(s);
                if segs.len() == 1 && pos == 0 {
                    return Ok(Some(Merged { batch: p.batch, keys: Some(p.keys) }));
                }
                break;
            }
            if total >= self.batch_size {
                break;
            }
        }
        if let [(b, start, len)] = segs.as_slice() {
            return Ok(Some(Merged { batch: b.slice(*start, *len), keys: None }));
        }
        let batches: Vec<&RecordBatch> = segs.iter().map(|s| &s.0).collect();
        let idx: Vec<(usize, usize)> =
            segs.iter().enumerate().flat_map(|(k, &(_, start, len))| (start..start + len).map(move |r| (k, r))).collect();
        Ok(Some(Merged { batch: interleave_record_batch(&batches, &idx)?, keys: None }))
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

/// Merge the (each sorted) partitions of an input into one ordered partition,
/// like `SortPreservingMergeExec`, with the batch-level fast path above.
#[derive(Debug)]
pub struct SortedMergeExec {
    input: Arc<dyn ExecutionPlan>,
    ordering: LexOrdering,
    read_ahead: usize,
    cache: Arc<PlanProperties>,
}

impl SortedMergeExec {
    pub fn new(input: Arc<dyn ExecutionPlan>, ordering: LexOrdering) -> Self {
        let mut eq = input.equivalence_properties().clone();
        eq.clear_per_partition_constants();
        eq.add_ordering(ordering.clone());
        let cache = PlanProperties::new(
            eq,
            Partitioning::UnknownPartitioning(1),
            input.pipeline_behavior(),
            input.boundedness(),
        );
        Self { input, ordering, read_ahead: usize::MAX, cache: Arc::new(cache) }
    }

    /// Bound the read-ahead to `bytes` (default: see the module docs).
    pub fn with_read_ahead_limit(mut self, bytes: usize) -> Self {
        self.read_ahead = bytes;
        self
    }
}

impl DisplayAs for SortedMergeExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "SortedMergeExec: [{}]", self.ordering)
    }
}

impl ExecutionPlan for SortedMergeExec {
    fn name(&self) -> &str {
        "SortedMergeExec"
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.cache
    }

    fn required_input_ordering(&self) -> Vec<Option<OrderingRequirements>> {
        vec![Some(OrderingRequirements::from(self.ordering.clone()))]
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

    fn with_new_children(self: Arc<Self>, mut children: Vec<Arc<dyn ExecutionPlan>>) -> Result<Arc<dyn ExecutionPlan>> {
        let (Some(input), true) = (children.pop(), children.is_empty()) else {
            return internal_err!("SortedMergeExec expects one child");
        };
        Ok(Arc::new(Self::new(input, self.ordering.clone()).with_read_ahead_limit(self.read_ahead)))
    }

    fn execute(&self, partition: usize, context: Arc<TaskContext>) -> Result<SendableRecordBatchStream> {
        if partition != 0 {
            return exec_err!("SortedMergeExec has one partition, got {partition}");
        }
        let prep = row_prep(&self.ordering, &self.input.schema())?;
        let check = OrderCheck {
            message: format!("input is not sorted by [{}]", self.ordering),
            in_batch: false,
            hint: String::new(),
        };
        let mut m = merged(&self.input, &context, prep, check, self.read_ahead)?;
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
