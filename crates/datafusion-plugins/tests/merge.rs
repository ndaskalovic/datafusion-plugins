//! `SortedMergeExec` (the merge inside the nodes) against a full sort, on
//! disjoint, overlapping, single and empty partitions, with ties.
//! (`merge_budget.rs` repeats some with a tiny read-ahead budget.)

use std::sync::Arc;

use arrow::array::{Array, Int64Array, RecordBatch};
use arrow::compute::SortOptions;
use arrow::datatypes::{DataType, Field, Schema};
use datafusion::datasource::memory::MemorySourceConfig;
use datafusion::execution::memory_pool::{
    FairSpillPool, GreedyMemoryPool, MemoryPool, PeakRecordingPool, UnboundedMemoryPool,
};
use datafusion::execution::runtime_env::RuntimeEnvBuilder;
use datafusion::physical_expr::expressions::Column;
use datafusion::physical_expr::{LexOrdering, PhysicalSortExpr};
use datafusion::physical_plan::{ExecutionPlan, collect};
use datafusion::prelude::{SessionConfig, SessionContext};
use datafusion_plugins::SortedMergeExec;

struct Rng(u64);
impl Rng {
    fn below(&mut self, n: u64) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0 % n.max(1)
    }
}

fn schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int64, true),
        Field::new("id", DataType::Int64, false),
    ]))
}

/// Partitions of (k, id) rows, each sorted by k, as batches of `bs` rows.
fn plan(parts: &[Vec<(Option<i64>, i64)>], bs: usize) -> Arc<dyn ExecutionPlan> {
    let s = schema();
    let data: Vec<Vec<RecordBatch>> = parts
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
        .collect();
    MemorySourceConfig::try_new_exec(&data, s, None).unwrap()
}

async fn merge(parts: &[Vec<(Option<i64>, i64)>], bs: usize) -> Vec<(Option<i64>, i64)> {
    merge_in(parts, bs, Arc::new(UnboundedMemoryPool::default())).await
}

/// Merge with `pool` as the memory pool.
async fn merge_in(parts: &[Vec<(Option<i64>, i64)>], bs: usize, pool: Arc<dyn MemoryPool>) -> Vec<(Option<i64>, i64)> {
    let ord = LexOrdering::new(vec![PhysicalSortExpr {
        expr: Arc::new(Column::new("k", 0)),
        options: SortOptions { descending: false, nulls_first: true },
    }])
    .unwrap();
    let runtime = RuntimeEnvBuilder::new().with_memory_pool(pool).build_arc().unwrap();
    let ctx = SessionContext::new_with_config_rt(SessionConfig::new().with_batch_size(16), runtime);
    let m = Arc::new(SortedMergeExec::new(plan(parts, bs), ord));
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
            assert_eq!(merge_in(&disjoint, 64, Arc::clone(&pool)).await, expected(&disjoint), "limit {limit}");
            assert_eq!(merge_in(&overlap, 32, Arc::clone(&pool)).await, expected(&overlap), "limit {limit}");
            assert_eq!(pool.reserved(), 0, "limit {limit}: reservation not returned");
        }
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn reservation_grows_and_returns_to_zero() {
    let mut rng = Rng(19);
    let parts: Vec<_> = (0..5).map(|p| sorted_part(&mut rng, 5000, p * 100_000, 100_000, p * 100_000, false)).collect();
    let pool = Arc::new(PeakRecordingPool::new(Arc::new(UnboundedMemoryPool::default())));
    assert_eq!(merge_in(&parts, 64, Arc::clone(&pool) as _).await, expected(&parts));
    assert!(pool.peak_reserved() > 0, "read-ahead was not accounted");
    assert_eq!(pool.reserved(), 0, "reservation not returned");
}
