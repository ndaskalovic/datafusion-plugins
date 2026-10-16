//! `join_asof` over in-memory and host inputs: failing upstreams, NULL
//! placement, nested nodes, empty inputs, more key types, re-execution,
//! limits and memory limits.

mod common;

use std::sync::Arc;

use arrow::array::{ArrayRef, Date32Array, DictionaryArray, Float64Array, Int64Array, RecordBatch, UInt64Array};
use arrow::datatypes::{DataType, Field, Int32Type, Schema, SchemaRef};
use datafusion::catalog::TableProvider;
use datafusion::common::Result;
use datafusion::execution::memory_pool::GreedyMemoryPool;
use datafusion::execution::runtime_env::RuntimeEnvBuilder;
use datafusion::physical_plan::limit::GlobalLimitExec;
use datafusion::physical_plan::{ExecutionPlan, ExecutionPlanProperties, collect};
use datafusion::prelude::{SessionConfig, SessionContext};
use datafusion_plugins::{BatchReader, HostInputs, JoinAsof, Markout, NodeTable, PartitionSource, PreparedSource};

use common::{failing, ints, mem};

fn schema(key: DataType, payload: &str) -> SchemaRef {
    Arc::new(Schema::new(vec![Field::new("ts", key, true), Field::new(payload, DataType::Int64, false)]))
}

fn batch(s: &SchemaRef, ts: ArrayRef, x: &[i64]) -> RecordBatch {
    RecordBatch::try_new(Arc::clone(s), vec![ts, Arc::new(Int64Array::from(x.to_vec()))]).unwrap()
}

fn i64s(v: &[Option<i64>]) -> ArrayRef {
    Arc::new(Int64Array::from(v.to_vec()))
}

fn ctx() -> SessionContext {
    SessionContext::new_with_config(SessionConfig::new().with_target_partitions(2).with_batch_size(4))
}

async fn run(join: JoinAsof, left: Arc<dyn ExecutionPlan>, right: Arc<dyn ExecutionPlan>) -> Result<Vec<RecordBatch>> {
    let c = ctx();
    collect(join.build(left, right, c.state().config_options())?, c.task_ctx()).await
}

/// (id, v) per output row.
fn pairs(bs: &[RecordBatch]) -> Vec<(i64, Option<i64>)> {
    ints(bs, "id").into_iter().zip(ints(bs, "v")).map(|(id, v)| (id.unwrap(), v)).collect()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn failing_upstream_is_an_error() {
    let (ls, rs) = (schema(DataType::Int64, "id"), schema(DataType::Int64, "v"));
    let left = || {
        vec![
            vec![batch(&ls, i64s(&[Some(1), Some(2)]), &[1, 2]), batch(&ls, i64s(&[Some(3)]), &[3])],
            vec![
                batch(&ls, i64s(&[Some(10)]), &[10]),
                batch(&ls, i64s(&[Some(11)]), &[11]),
                batch(&ls, i64s(&[Some(12)]), &[12]),
            ],
        ]
    };
    let right = || vec![vec![batch(&rs, i64s(&[Some(0)]), &[100]), batch(&rs, i64s(&[Some(5)]), &[105])]];
    for error in [false, true] {
        for side in ["left", "right"] {
            let (lf, rf) = if side == "left" { (Some((1, 1, error)), None) } else { (None, Some((0, 1, error))) };
            let join = JoinAsof::new("ts").assume_sorted(true);
            let res = run(join, failing(&ls, left(), lf), failing(&rs, right(), rf)).await;
            let err = res.map(|b| pairs(&b)).expect_err("a failing partition must fail the query").to_string();
            assert!(err.contains("upstream") && err.contains("at batch 1"), "{side}: {err}");
        }
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn null_keys_first_or_last() {
    // Left rows with a NULL key never match and keep their place; inputs
    // sorted NULLS FIRST or LAST (SQL's default) both work, with one or
    // two partitions, checked or not.
    let (ls, rs) = (schema(DataType::Int64, "id"), schema(DataType::Int64, "v"));
    let right = vec![vec![
        batch(&rs, i64s(&[Some(0), Some(9)]), &[100, 109]),
        batch(&rs, i64s(&[None, Some(19), Some(49)]), &[0, 119, 149]),
    ]];
    let want = |id: i64| match id {
        1 => Some(100),
        2 => Some(149),
        4 => Some(109),
        5 => Some(119),
        _ => None, // 3, 6: NULL ts
    };
    // (ts, id), sorted NULLS FIRST and NULLS LAST.
    let first = [(None, 3), (None, 6), (Some(5), 1), (Some(10), 4), (Some(20), 5), (Some(50), 2)];
    let last = [(Some(5), 1), (Some(10), 4), (Some(20), 5), (Some(50), 2), (None, 3), (None, 6)];
    for rows in [first, last] {
        for parts in [1, 2] {
            for check in [true, false] {
                // Every `parts`-th row per partition (each stays sorted), two rows per batch.
                let left: Vec<Vec<RecordBatch>> = (0..parts)
                    .map(|p| {
                        let mine: Vec<_> = rows.iter().skip(p).step_by(parts).collect();
                        mine.chunks(2)
                            .map(|c| batch(&ls, i64s(&c.iter().map(|r| r.0).collect::<Vec<_>>()), &c.iter().map(|r| r.1).collect::<Vec<_>>()))
                            .collect()
                    })
                    .collect();
                let input: Vec<i64> = rows.iter().map(|r| r.1).collect();
                let plan = JoinAsof::new("ts")
                    .assume_sorted(true)
                    .check_sortedness(check)
                    .build(mem(&ls, left), mem(&rs, right.clone()), ctx().state().config_options())
                    .unwrap();
                // The NULL placement isn't known, so no order is declared.
                assert!(plan.output_ordering().is_none());
                let got = pairs(&collect(plan, ctx().task_ctx()).await.unwrap());
                let mut ids: Vec<i64> = got.iter().map(|p| p.0).collect();
                for (id, v) in &got {
                    assert_eq!(*v, want(*id), "id {id} in {got:?}");
                }
                if parts == 1 {
                    assert_eq!(ids, input, "one partition keeps its order");
                }
                // Matched rows come out in key order.
                let keys: Vec<i64> = got.iter().filter_map(|p| p.1).collect();
                assert!(keys.is_sorted(), "{got:?}");
                ids.sort();
                assert_eq!(ids, vec![1, 2, 3, 4, 5, 6]);
            }
        }
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn empty_inputs() {
    let (ls, rs) = (schema(DataType::Int64, "id"), schema(DataType::Int64, "v"));
    let left = vec![vec![batch(&ls, i64s(&[Some(1), Some(5)]), &[1, 2])]];
    let right = vec![vec![batch(&rs, i64s(&[Some(0)]), &[100])]];
    for assume in [true, false] {
        let j = || JoinAsof::new("ts").assume_sorted(assume);
        let got = run(j(), mem(&ls, left.clone()), mem(&rs, vec![vec![]])).await.unwrap();
        assert_eq!(pairs(&got), vec![(1, None), (2, None)]);
        let got = run(j(), mem(&ls, vec![vec![batch(&ls, i64s(&[]), &[])]]), mem(&rs, right.clone())).await.unwrap();
        assert!(pairs(&got).is_empty());
        let got = run(j(), mem(&ls, vec![vec![], vec![]]), mem(&rs, vec![vec![]])).await.unwrap();
        assert!(pairs(&got).is_empty());
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn more_key_types() {
    // `on`: Float64 (with -0.0), UInt64 (above i64::MAX), Date32.
    let ons: Vec<(DataType, ArrayRef, ArrayRef)> = vec![
        (
            DataType::Float64,
            Arc::new(Float64Array::from(vec![-1.5, 0.0, 2.5])),
            Arc::new(Float64Array::from(vec![-2.0, -0.0, 2.0])),
        ),
        (
            DataType::UInt64,
            Arc::new(UInt64Array::from(vec![1, u64::MAX - 5, u64::MAX])),
            Arc::new(UInt64Array::from(vec![0, 7, u64::MAX - 1])),
        ),
        (
            DataType::Date32,
            Arc::new(Date32Array::from(vec![-10, 0, 20])),
            Arc::new(Date32Array::from(vec![-11, -1, 19])),
        ),
    ];
    for (t, l, r) in ons {
        let (ls, rs) = (schema(t.clone(), "id"), schema(t.clone(), "v"));
        let got = run(
            JoinAsof::new("ts"),
            mem(&ls, vec![vec![batch(&ls, l, &[1, 2, 3])]]),
            mem(&rs, vec![vec![batch(&rs, r, &[10, 20, 30])]]),
        )
        .await
        .unwrap();
        assert_eq!(pairs(&got), vec![(1, Some(10)), (2, Some(20)), (3, Some(30))], "{t}");
    }
    // `by`: UInt64 (above i64::MAX) and a string dictionary.
    let bys: Vec<(ArrayRef, ArrayRef)> = vec![
        (Arc::new(UInt64Array::from(vec![3, u64::MAX, u64::MAX])), Arc::new(UInt64Array::from(vec![3, 3, u64::MAX]))),
        (
            Arc::new(DictionaryArray::<Int32Type>::from_iter(["a", "b", "b"])),
            Arc::new(DictionaryArray::<Int32Type>::from_iter(["a", "a", "b"])),
        ),
    ];
    for (lg, rg) in bys {
        let side = |g: &ArrayRef, payload: &str, x: &[i64]| {
            let s = Arc::new(Schema::new(vec![
                Field::new("g", g.data_type().clone(), true),
                Field::new("ts", DataType::Int64, true),
                Field::new(payload, DataType::Int64, false),
            ]));
            let ts = i64s(&[Some(1), Some(2), Some(3)]);
            let b = RecordBatch::try_new(Arc::clone(&s), vec![Arc::clone(g), ts, Arc::new(Int64Array::from(x.to_vec()))]);
            mem(&s, vec![vec![b.unwrap()]])
        };
        let got = run(JoinAsof::new("ts").by(["g"]), side(&lg, "id", &[1, 2, 3]), side(&rg, "v", &[10, 20, 30])).await;
        assert_eq!(pairs(&got.unwrap()), vec![(1, Some(10)), (2, None), (3, Some(30))], "{:?}", lg.data_type());
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn rerun_and_limit() {
    let (ls, rs) = (schema(DataType::Int64, "id"), schema(DataType::Int64, "v"));
    let ids: Vec<i64> = (0..1000).collect();
    let left = vec![vec![batch(&ls, i64s(&ids.iter().map(|i| Some(i * 3)).collect::<Vec<_>>()), &ids)]];
    let right = vec![vec![batch(&rs, i64s(&ids.iter().map(|i| Some(i * 7)).collect::<Vec<_>>()), &ids)]];
    let c = ctx();
    let plan = JoinAsof::new("ts").build(mem(&ls, left), mem(&rs, right), c.state().config_options()).unwrap();
    let first = pairs(&collect(Arc::clone(&plan), c.task_ctx()).await.unwrap());
    assert_eq!(first.len(), 1000);
    assert_eq!(pairs(&collect(Arc::clone(&plan), c.task_ctx()).await.unwrap()), first);
    let limited = Arc::new(GlobalLimitExec::new(plan, 0, Some(5)));
    assert_eq!(pairs(&collect(limited, c.task_ctx()).await.unwrap()), first[..5]);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn frontier_beyond_memory_limit() {
    // One left row that can reach every right row: the buffer can't fit a
    // 64 KiB pool, which is a clear error, not an overrun.
    let (ls, rs) = (schema(DataType::Int64, "id"), schema(DataType::Int64, "v"));
    let n: Vec<i64> = (0..20_000).collect();
    let right: Vec<Vec<RecordBatch>> =
        vec![n.chunks(512).map(|c| batch(&rs, i64s(&c.iter().map(|i| Some(*i)).collect::<Vec<_>>()), c)).collect()];
    let left = vec![vec![batch(&ls, i64s(&[Some(0), Some(1 << 40)]), &[1, 2])]];
    let runtime = RuntimeEnvBuilder::new().with_memory_pool(Arc::new(GreedyMemoryPool::new(64 << 10))).build_arc().unwrap();
    let c = SessionContext::new_with_config_rt(SessionConfig::new().with_batch_size(512), runtime);
    // Markouts at -1e12 and 0 s: one left batch spans all right rows.
    let join = JoinAsof::new("ts").markouts(vec![Markout::from(-1_000_000_000_000), Markout::from(0)]);
    let ts = DataType::Timestamp(arrow::datatypes::TimeUnit::Second, None);
    let cast = |s: &SchemaRef, parts: Vec<Vec<RecordBatch>>| {
        let s2 = Arc::new(Schema::new(vec![Field::new("ts", ts.clone(), true), s.field(1).clone()]));
        let parts = parts
            .into_iter()
            .map(|p| {
                p.into_iter()
                    .map(|b| {
                        let t = arrow::compute::cast(b.column(0), &ts).unwrap();
                        RecordBatch::try_new(Arc::clone(&s2), vec![t, Arc::clone(b.column(1))]).unwrap()
                    })
                    .collect()
            })
            .collect();
        mem(&s2, parts)
    };
    let plan = join.assume_sorted(true).build(cast(&ls, left), cast(&rs, right), c.state().config_options()).unwrap();
    let err = collect(plan, c.task_ctx()).await.unwrap_err().to_string();
    assert!(err.contains("exceed the memory limit"), "{err}");
}

/// A host input whose `open` runs the plan `inner` to completion, as
/// datafusion-python does for a DataFrame over another node.
#[derive(Debug)]
struct Nested {
    inner: Arc<dyn ExecutionPlan>,
}

impl PartitionSource for Nested {
    fn schema(&self) -> SchemaRef {
        self.inner.schema()
    }
    fn prepare(&self, _columns: &[String]) -> Result<Arc<dyn PreparedSource>> {
        Ok(Arc::new(Nested { inner: Arc::clone(&self.inner) }))
    }
}

impl PreparedSource for Nested {
    fn schema(&self) -> SchemaRef {
        self.inner.schema()
    }
    fn partition_count(&self) -> usize {
        1
    }
    fn open(&self) -> Result<Vec<BatchReader>> {
        // Blocks on the runtime: only allowed off its worker threads.
        let rt = tokio::runtime::Handle::current();
        let batches = rt.block_on(collect(Arc::clone(&self.inner), SessionContext::new().task_ctx()))?;
        Ok(vec![Box::new(batches.into_iter().map(Ok))])
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn nested_host_inputs() {
    // join_asof(join_asof(a, b), c) with the inner node a host input of the
    // outer: the host's blocking open must not run on a runtime worker.
    let (ls, bs, cs) = (schema(DataType::Int64, "id"), schema(DataType::Int64, "v"), schema(DataType::Int64, "w"));
    let a = mem(&ls, vec![vec![batch(&ls, i64s(&[Some(1), Some(5), Some(9)]), &[1, 2, 3])]]);
    let b = mem(&bs, vec![vec![batch(&bs, i64s(&[Some(0), Some(4), Some(8)]), &[10, 20, 30])]]);
    let c = mem(&cs, vec![vec![batch(&cs, i64s(&[Some(2), Some(6)]), &[7, 8])]]);
    let cfg = ctx().state().config_options().clone();
    let inner = JoinAsof::new("ts").build(a, b, &cfg).unwrap();
    let sources: Vec<Arc<dyn PartitionSource>> = vec![Arc::new(Nested { inner }), Arc::new(Nested { inner: c })];
    let table = NodeTable::try_new(Arc::new(HostInputs::new(sources)), Arc::new(JoinAsof::new("ts"))).unwrap();
    assert_eq!(table.schema().fields().len(), 4);
    for _ in 0..2 {
        // Planned again for every run.
        let plan = table.plan(&cfg, None).await.unwrap();
        let got = collect(plan, ctx().task_ctx()).await.unwrap();
        assert_eq!(ints(&got, "v"), vec![Some(10), Some(20), Some(30)]);
        assert_eq!(ints(&got, "w"), vec![None, Some(7), Some(8)]);
    }
    assert!(table.last_plan().unwrap().contains("HostInputExec"));
}

#[derive(Clone, Debug)]
struct Row {
    g: Option<i64>,
    ts: Option<i64>,
    id: i64,
}

fn fuzz_batch(s: &SchemaRef, rows: &[Row], string_by: bool) -> RecordBatch {
    let g: ArrayRef = if string_by {
        Arc::new(arrow::array::StringArray::from_iter(rows.iter().map(|r| r.g.map(|g| format!("k{:03}", g + 500)))))
    } else {
        Arc::new(Int64Array::from_iter(rows.iter().map(|r| r.g)))
    };
    let ts = Arc::new(arrow::array::TimestampSecondArray::from_iter(rows.iter().map(|r| r.ts)));
    RecordBatch::try_new(Arc::clone(s), vec![g, ts, Arc::new(Int64Array::from_iter_values(rows.iter().map(|r| r.id)))])
        .unwrap()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn fuzz_markouts_overlapping_partitions() {
    // Random sorted rows split over overlapping partitions in random
    // batches (some empty), vs brute force; the output keeps left order.
    use common::Rng;
    let gen_rows = |rng: &mut Rng, n: usize, nulls: bool, id0: i64, ungrouped: bool| {
        let mut v: Vec<Row> = (0..n)
            .map(|i| Row {
                g: if nulls && rng.below(15) == 0 { None } else { Some(rng.below(4) as i64 - 2) },
                ts: if nulls && rng.below(15) == 0 { None } else { Some(rng.below(60) as i64 - 30) },
                id: id0 + i as i64,
            })
            .collect();
        if ungrouped {
            v.sort_by_key(|r| r.ts);
        } else {
            v.sort_by_key(|r| (r.g, r.ts)); // NULLS FIRST
        }
        v
    };
    for seed in 1..400u64 {
        let mut rng = Rng(seed * 7919 + 13);
        let (string_by, ungrouped, strict) = (seed % 3 == 0, seed % 5 == 0, seed % 2 == 0);
        let n1 = 1 + rng.below(80) as usize;
        let left = gen_rows(&mut rng, n1, true, 0, ungrouped);
        let n2 = rng.below(40) as usize;
        let right = gen_rows(&mut rng, n2, seed % 4 == 0, 10_000, ungrouped);
        let schema = |payload: &str| {
            Arc::new(Schema::new(vec![
                Field::new("g", if string_by { DataType::Utf8 } else { DataType::Int64 }, true),
                Field::new("ts", DataType::Timestamp(arrow::datatypes::TimeUnit::Second, None), true),
                Field::new(payload, DataType::Int64, false),
            ]))
        };
        let (ls, rs) = (schema("id"), schema("rid"));
        let mut split = |rows: &[Row], np: usize, s: &SchemaRef| -> Vec<Vec<RecordBatch>> {
            let mut parts: Vec<Vec<Row>> = vec![Vec::new(); np];
            for r in rows {
                parts[rng.below(np as u64) as usize].push(r.clone());
            }
            parts
                .into_iter()
                .map(|p| {
                    let mut out = Vec::new();
                    let mut i = 0;
                    while i < p.len() {
                        let e = (i + 1 + rng.below(5) as usize).min(p.len());
                        out.push(fuzz_batch(s, &p[i..e], string_by));
                        i = e;
                    }
                    if rng.below(3) == 0 {
                        out.push(fuzz_batch(s, &[], string_by));
                    }
                    out
                })
                .collect()
        };
        let p1 = 1 + (seed as usize % 3);
        let lp = split(&left, p1, &ls);
        let p2 = 1 + (seed as usize % 4);
        let rp = split(&right, p2, &rs);
        let shifts: Vec<i64> = match seed % 4 {
            0 => vec![-7, 0, 3],
            1 => vec![25, -40],
            2 => vec![0],
            _ => vec![1, 2, 5, -1],
        };
        let mut node = JoinAsof::new("ts")
            .markouts(shifts.iter().map(|h| Markout::from(*h)).collect())
            .allow_exact_matches(!strict)
            .assume_sorted(seed % 7 != 0);
        if !ungrouped {
            node = node.by(["g"]);
        }
        let c = SessionContext::new_with_config(
            SessionConfig::new().with_target_partitions(1 + (seed % 3) as usize).with_batch_size(1 + (seed % 6) as usize),
        );
        let plan = node.build(failing(&ls, lp, None), failing(&rs, rp, None), c.state().config_options()).unwrap();
        let bs = collect(plan, c.task_ctx()).await.unwrap();
        let got_ids: Vec<i64> = ints(&bs, "id").into_iter().map(Option::unwrap).collect();
        let cols: Vec<Vec<Option<i64>>> = shifts.iter().map(|h| ints(&bs, &format!("rid_{h}s"))).collect();
        // Matched right ts per left row and markout (ties may pick any row).
        let rts = |id: Option<i64>| id.map(|id| right.iter().find(|r| r.id == id).unwrap().ts);
        let mut got: Vec<(i64, Vec<Option<Option<i64>>>)> =
            got_ids.iter().enumerate().map(|(i, id)| (*id, cols.iter().map(|c| rts(c[i])).collect())).collect();
        let mut want: Vec<(i64, Vec<Option<Option<i64>>>)> = left
            .iter()
            .map(|l| {
                let m = shifts.iter().map(|h| {
                    let (Some(lt), true) = (l.ts, ungrouped || l.g.is_some()) else { return None };
                    let ok = |r: &&Row| {
                        r.ts.is_some()
                            && (ungrouped || (r.g.is_some() && r.g == l.g))
                            && if strict { r.ts.unwrap() < lt + h } else { r.ts.unwrap() <= lt + h }
                    };
                    right.iter().filter(ok).map(|r| r.ts).max()
                });
                (l.id, m.collect())
            })
            .collect();
        // Rows with valid keys come out in left order.
        let key = |id: i64| left.iter().find(|r| r.id == id).map(|r| ((!ungrouped).then_some(r.g), r.ts)).unwrap();
        let valid: Vec<_> = got_ids.iter().map(|id| key(*id)).filter(|(g, t)| t.is_some() && g.is_none_or(|g| g.is_some())).collect();
        assert!(valid.is_sorted(), "seed {seed}: output not in left order {valid:?}");
        got.sort();
        want.sort();
        assert_eq!(got, want, "seed {seed} string_by={string_by} ungrouped={ungrouped} strict={strict}");
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn tight_memory_limit_with_many_batches_in_flight() {
    // Each left batch reaches 64 one-row right batches (the buffer peaks at
    // ~16 KiB), so the frontier fits a 20 KiB pool even while the pool is
    // full: what in-flight workers and outputs hold on top must not make it
    // an error.
    let (ls, rs) = (schema(DataType::Int64, "id"), schema(DataType::Int64, "v"));
    let n: Vec<i64> = (0..20_000).collect();
    let part = |s: &SchemaRef, size: usize| -> Vec<Vec<RecordBatch>> {
        vec![n.chunks(size).map(|c| batch(s, i64s(&c.iter().map(|i| Some(*i)).collect::<Vec<_>>()), c)).collect()]
    };
    let pool: Arc<dyn datafusion::execution::memory_pool::MemoryPool> = Arc::new(GreedyMemoryPool::new(20 << 10));
    // Another consumer holds the whole pool: every reservation is refused.
    let other = datafusion::execution::memory_pool::MemoryConsumer::new("other").register(&pool);
    other.grow(20 << 10);
    let runtime = RuntimeEnvBuilder::new().with_memory_pool(pool).build_arc().unwrap();
    let config = SessionConfig::new().with_target_partitions(8).with_batch_size(64);
    let c = SessionContext::new_with_config_rt(config, runtime);
    let join = JoinAsof::new("ts").markouts(vec![Markout::from(0), Markout::from(-1)]).assume_sorted(true);
    let ts = DataType::Timestamp(arrow::datatypes::TimeUnit::Second, None);
    let cast = |s: &SchemaRef, parts: Vec<Vec<RecordBatch>>| {
        let s2 = Arc::new(Schema::new(vec![Field::new("ts", ts.clone(), true), s.field(1).clone()]));
        let parts = parts
            .into_iter()
            .map(|p| {
                p.into_iter()
                    .map(|b| {
                        let t = arrow::compute::cast(b.column(0), &ts).unwrap();
                        RecordBatch::try_new(Arc::clone(&s2), vec![t, Arc::clone(b.column(1))]).unwrap()
                    })
                    .collect()
            })
            .collect();
        mem(&s2, parts)
    };
    let plan = join.build(cast(&ls, part(&ls, 64)), cast(&rs, part(&rs, 1)), c.state().config_options()).unwrap();
    let got = collect(plan, c.task_ctx()).await.unwrap();
    assert_eq!(ints(&got, "v_0s"), n.iter().map(|i| Some(*i)).collect::<Vec<_>>());
}
