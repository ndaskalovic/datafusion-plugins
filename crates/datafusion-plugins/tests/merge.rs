//! `SortedMergeExec` (the merge inside the nodes) against a full sort, on
//! disjoint, overlapping, single and empty partitions, with ties; under
//! memory-limited pools and a tiny read-ahead limit; and with a failing
//! partition.

mod common;

use std::sync::Arc;

use arrow::array::{Array, Int64Array, RecordBatch};
use arrow::compute::SortOptions;
use arrow::datatypes::{DataType, Field, Schema};
use datafusion::execution::memory_pool::{
    FairSpillPool, GreedyMemoryPool, MemoryPool, PeakRecordingPool, UnboundedMemoryPool,
};
use datafusion::execution::runtime_env::RuntimeEnvBuilder;
use datafusion::physical_expr::expressions::Column;
use datafusion::physical_expr::{LexOrdering, PhysicalSortExpr};
use datafusion::physical_plan::collect;
use datafusion::prelude::{SessionConfig, SessionContext};
use datafusion_plugins::SortedMergeExec;

use common::Rng;

fn schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int64, true),
        Field::new("id", DataType::Int64, false),
    ]))
}

/// Partitions of (k, id) rows as batches of `bs` rows.
fn batches(parts: &[Vec<(Option<i64>, i64)>], bs: usize) -> Vec<Vec<RecordBatch>> {
    let s = schema();
    parts
        .iter()
        .map(|p| {
            p.chunks(bs.max(1))
                .map(|c| {
                    RecordBatch::try_new(
                        Arc::clone(&s),
                        vec![
                            Arc::new(Int64Array::from_iter(c.iter().map(|r| r.0))),
                            Arc::new(Int64Array::from_iter_values(c.iter().map(|r| r.1))),
                        ],
                    )
                    .unwrap()
                })
                .collect()
        })
        .collect()
}

async fn merge(parts: &[Vec<(Option<i64>, i64)>], bs: usize) -> Vec<(Option<i64>, i64)> {
    merge_in(parts, bs, Arc::new(UnboundedMemoryPool::default()), usize::MAX).await
}

/// Merge with `pool` as the memory pool and at most `read_ahead` bytes of
/// read-ahead.
async fn merge_in(
    parts: &[Vec<(Option<i64>, i64)>],
    bs: usize,
    pool: Arc<dyn MemoryPool>,
    read_ahead: usize,
) -> Vec<(Option<i64>, i64)> {
    let runtime = RuntimeEnvBuilder::new().with_memory_pool(pool).build_arc().unwrap();
    let ctx = SessionContext::new_with_config_rt(SessionConfig::new().with_batch_size(16), runtime);
    let m = Arc::new(SortedMergeExec::new(common::mem(&schema(), batches(parts, bs)), ordering()).with_read_ahead_limit(read_ahead));
    let mut out = Vec::new();
    for b in collect(m, ctx.task_ctx()).await.unwrap() {
        let k = b.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
        let id = b.column(1).as_any().downcast_ref::<Int64Array>().unwrap();
        for i in 0..b.num_rows() {
            out.push(((!k.is_null(i)).then(|| k.value(i)), id.value(i)));
        }
    }
    out
}

fn ordering() -> LexOrdering {
    let options = SortOptions { descending: false, nulls_first: true };
    LexOrdering::new(vec![PhysicalSortExpr { expr: Arc::new(Column::new("k", 0)), options }]).unwrap()
}

/// Expected: all rows sorted by k (nulls first); ties by partition, then
/// position (the merge is stable across partitions in index order).
fn expected(parts: &[Vec<(Option<i64>, i64)>]) -> Vec<(Option<i64>, i64)> {
    let mut all: Vec<(Option<i64>, usize, usize, i64)> = parts
        .iter()
        .enumerate()
        .flat_map(|(p, rows)| rows.iter().enumerate().map(move |(i, r)| (r.0, p, i, r.1)))
        .collect();
    all.sort();
    all.into_iter().map(|r| (r.0, r.3)).collect()
}

fn sorted_part(rng: &mut Rng, n: usize, lo: i64, span: u64, id0: i64, nulls: bool) -> Vec<(Option<i64>, i64)> {
    let mut v: Vec<Option<i64>> = (0..n)
        .map(|_| if nulls && rng.below(20) == 0 { None } else { Some(lo + rng.below(span) as i64) })
        .collect();
    v.sort();
    v.into_iter().enumerate().map(|(i, k)| (k, id0 + i as i64)).collect()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn disjoint_ranges() {
    let mut rng = Rng(7);
    let parts: Vec<_> = (0..5).map(|p| sorted_part(&mut rng, 300, p * 1000, 1000, p * 10_000, false)).collect();
    for bs in [1, 7, 64, 1000] {
        assert_eq!(merge(&parts, bs).await, expected(&parts), "bs={bs}");
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn overlapping_with_ties_and_nulls() {
    let mut rng = Rng(11);
    let parts: Vec<_> = (0..6).map(|p| sorted_part(&mut rng, 400, 0, 50, p * 10_000, true)).collect();
    for bs in [1, 5, 33, 400] {
        assert_eq!(merge(&parts, bs).await, expected(&parts), "bs={bs}");
    }
    // Round-robin of one sorted stream: maximal interleaving.
    let one = sorted_part(&mut rng, 2000, 0, 100_000, 0, false);
    let rr: Vec<Vec<_>> = (0..4).map(|p| one.iter().skip(p).step_by(4).cloned().collect()).collect();
    assert_eq!(merge(&rr, 16).await, expected(&rr));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn single_and_empty_streams() {
    let mut rng = Rng(13);
    let one = vec![sorted_part(&mut rng, 500, 0, 100, 0, true)];
    assert_eq!(merge(&one, 64).await, expected(&one));
    let with_empty = vec![Vec::new(), sorted_part(&mut rng, 100, 0, 30, 0, false), Vec::new(), sorted_part(&mut rng, 50, 10, 5, 1000, false)];
    assert_eq!(merge(&with_empty, 8).await, expected(&with_empty));
    let all_empty: Vec<Vec<(Option<i64>, i64)>> = vec![Vec::new(), Vec::new()];
    assert!(merge(&all_empty, 8).await.is_empty());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn memory_limited_pools() {
    // Pools far smaller than the input: read-ahead stops, results are
    // still right and nothing deadlocks.
    let mut rng = Rng(17);
    let disjoint: Vec<_> = (0..6).map(|p| sorted_part(&mut rng, 3000, p * 100_000, 100_000, p * 100_000, false)).collect();
    let overlap: Vec<_> = (0..6).map(|p| sorted_part(&mut rng, 2000, 0, 300, p * 100_000, true)).collect();
    for limit in [1usize, 16 << 10, 256 << 10] {
        let pools: [Arc<dyn MemoryPool>; 2] =
            [Arc::new(GreedyMemoryPool::new(limit)), Arc::new(FairSpillPool::new(limit))];
        for pool in pools {
            assert_eq!(merge_in(&disjoint, 64, Arc::clone(&pool), usize::MAX).await, expected(&disjoint), "limit {limit}");
            assert_eq!(merge_in(&overlap, 32, Arc::clone(&pool), usize::MAX).await, expected(&overlap), "limit {limit}");
            assert_eq!(pool.reserved(), 0, "limit {limit}: reservation not returned");
        }
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn reservation_grows_and_returns_to_zero() {
    let mut rng = Rng(19);
    let parts: Vec<_> = (0..5).map(|p| sorted_part(&mut rng, 5000, p * 100_000, 100_000, p * 100_000, false)).collect();
    let pool = Arc::new(PeakRecordingPool::new(Arc::new(UnboundedMemoryPool::default())));
    assert_eq!(merge_in(&parts, 64, Arc::clone(&pool) as _, usize::MAX).await, expected(&parts));
    assert!(pool.peak_reserved() > 0, "read-ahead was not accounted");
    assert_eq!(pool.reserved(), 0, "reservation not returned");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn tiny_read_ahead_limit() {
    // ~2 KiB of read-ahead: later ranges fill it while the first one drains;
    // the drained stream must still make progress (no deadlock), and under
    // an unbounded pool the peak stays within the limit plus the free
    // batches per partition.
    let limit = 2048;
    let mut rng = Rng(5);
    let parts: Vec<_> = (0..6).map(|p| sorted_part(&mut rng, 3000, p * 100_000, 100_000, p * 100_000, false)).collect();
    let pool = Arc::new(PeakRecordingPool::new(Arc::new(UnboundedMemoryPool::default())));
    assert_eq!(merge_in(&parts, 64, Arc::clone(&pool) as _, limit).await, expected(&parts));
    let one = 64 * 16 + 1024; // a 64-row (k, id) batch, with slack for buffers
    let bound = limit + parts.len() * 3 * one;
    assert!(pool.peak_reserved() <= bound, "peak {} > {bound}", pool.peak_reserved());
    let parts: Vec<_> = (0..6).map(|p| sorted_part(&mut rng, 2000, 0, 500, p * 100_000, true)).collect();
    assert_eq!(merge_in(&parts, 32, Arc::new(UnboundedMemoryPool::default()), limit).await, expected(&parts));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn failing_partition_is_an_error() {
    // A partition that panics or errors midway fails the merge (it used to
    // end it early, silently dropping rows).
    let mut rng = Rng(23);
    let parts: Vec<_> = (0..3).map(|p| sorted_part(&mut rng, 200, p * 1000, 1000, p * 10_000, false)).collect();
    let ctx = SessionContext::new_with_config(SessionConfig::new().with_batch_size(16));
    for error in [false, true] {
        let input = common::failing(&schema(), batches(&parts, 16), Some((1, 3, error)));
        let m = Arc::new(SortedMergeExec::new(input, ordering()));
        let err = collect(m, ctx.task_ctx()).await.unwrap_err().to_string();
        let want = if error { "upstream failed at batch 3" } else { "panicked: upstream panicked at batch 3" };
        assert!(err.contains(want), "{err}");
    }
}
