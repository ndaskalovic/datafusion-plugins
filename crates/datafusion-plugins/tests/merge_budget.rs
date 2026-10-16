//! The merge with a tiny shared read-ahead budget (own test binary: the
//! budget is read from the environment).

use std::sync::Arc;

use arrow::array::{Array, Int64Array, RecordBatch};
use arrow::compute::SortOptions;
use arrow::datatypes::{DataType, Field, Schema};
use datafusion::datasource::memory::MemorySourceConfig;
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
    let ord = LexOrdering::new(vec![PhysicalSortExpr {
        expr: Arc::new(Column::new("k", 0)),
        options: SortOptions { descending: false, nulls_first: true },
    }])
    .unwrap();
    let ctx = SessionContext::new_with_config(SessionConfig::new().with_batch_size(16));
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
async fn tiny_budget_disjoint_and_overlapping() {
    // ~2 KiB of shared read-ahead: later ranges fill it while the first one
    // drains; the drained stream must still make progress (no deadlock).
    unsafe { std::env::set_var("DATAFUSION_PLUGINS_READAHEAD_MB", "0.002") };
    let mut rng = Rng(5);
    let parts: Vec<_> = (0..6).map(|p| sorted_part(&mut rng, 3000, p * 100_000, 100_000, p * 100_000, false)).collect();
    assert_eq!(merge(&parts, 64).await, expected(&parts));
    let parts: Vec<_> = (0..6).map(|p| sorted_part(&mut rng, 2000, 0, 500, p * 100_000, true)).collect();
    assert_eq!(merge(&parts, 32).await, expected(&parts));
}

#[allow(dead_code)]
async fn disjoint_ranges() {
    let mut rng = Rng(7);
    let parts: Vec<_> = (0..5).map(|p| sorted_part(&mut rng, 300, p * 1000, 1000, p * 10_000, false)).collect();
    for bs in [1, 7, 64, 1000] {
        assert_eq!(merge(&parts, bs).await, expected(&parts), "bs={bs}");
    }
}
