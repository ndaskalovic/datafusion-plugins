//! Compare `join_asof` against a brute-force reference on small random
//! Parquet data, grouped and ungrouped, including ties, NULLs, groups without
//! right rows, and many (often empty) byte-range partitions. Each case runs
//!
//! * the builder over ordered `read_parquet` scans (range-parallel mode),
//! * the builder over a merged (so non-re-executable) left input, in
//!   execute-once mode against the right scan's range partitions,
//! * the builder over scans with no declared order, validated while
//!   streaming (execute-once, one right unit),
//! * the serialized-logical-plan provider (as used from Python).

use std::fs::File;
use std::sync::Arc;

use arrow::array::{
    Array, ArrayRef, Float64Array, Int32Array, Int64Array, RecordBatch, StringArray,
    TimestampNanosecondArray,
};
use arrow::datatypes::{DataType, Field, Schema, TimeUnit};
use datafusion::catalog::TableProvider;
use datafusion::parquet::arrow::ArrowWriter;
use datafusion::parquet::file::properties::WriterProperties;
use datafusion::physical_expr::LexOrdering;
use datafusion::physical_plan::sorts::sort_preserving_merge::SortPreservingMergeExec;
use datafusion::physical_plan::{ExecutionPlan, ExecutionPlanProperties, collect, displayable};
use datafusion::prelude::{ParquetReadOptions, SessionConfig, SessionContext, col};
use datafusion_plugins::{AsofOptions, AsofPlanTable, JoinAsof};

#[derive(Clone, Copy, Debug, PartialEq)]
enum By {
    None,
    I32,
    I64,
    Str,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Ts {
    I64,
    I32,
    TsNs,
}

#[derive(Clone, Copy, Debug)]
struct Case {
    seed: u64,
    by: By,
    ts: Ts,
    n_left: usize,
    n_right: usize,
    /// Left groups are drawn from [0, left_groups), right from [0, right_groups):
    /// left_groups > right_groups gives groups with no right rows.
    left_groups: i64,
    right_groups: i64,
    /// Range of ts values (small => many ties).
    ts_range: i64,
    null_per_mille: u64,
    strict: bool,
    row_group: usize,
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n.max(1)
    }
}

#[derive(Clone, Debug)]
struct Row {
    id: i64,
    g: Option<i64>,
    ts: Option<i64>,
}

fn gen_rows(rng: &mut Rng, n: usize, groups: i64, c: &Case, id_base: i64) -> Vec<Row> {
    (0..n)
        .map(|i| {
            let g = if c.by != By::None && rng.below(1000) < c.null_per_mille {
                None
            } else {
                Some(rng.below(groups as u64) as i64)
            };
            let ts = if rng.below(1000) < c.null_per_mille {
                None
            } else {
                Some(rng.below(c.ts_range as u64) as i64)
            };
            Row {
                id: id_base + i as i64,
                g,
                ts,
            }
        })
        .collect()
}

fn group_key(by: By, g: Option<i64>) -> Option<String> {
    // Sort key matching the column type's order.
    g.map(|g| match by {
        By::Str => format!("g{g}"),
        _ => format!("{:020}", g), // non-negative ints: zero-padded = numeric order
    })
}

/// Stable sort by `(g, ts)` (or `ts`), NULLS FIRST.
fn sort_rows(rows: &mut [Row], by: By) {
    match by {
        By::None => rows.sort_by_key(|r| r.ts),
        _ => rows.sort_by(|a, b| (group_key(by, a.g), a.ts).cmp(&(group_key(by, b.g), b.ts))),
    }
}

fn group_array(by: By, rows: &[Row]) -> Option<ArrayRef> {
    Some(match by {
        By::None => return None,
        By::I32 => Arc::new(Int32Array::from_iter(rows.iter().map(|r| r.g.map(|g| g as i32)))),
        By::I64 => Arc::new(Int64Array::from_iter(rows.iter().map(|r| r.g))),
        By::Str => Arc::new(StringArray::from_iter(
            rows.iter().map(|r| r.g.map(|g| format!("g{g}"))),
        )),
    })
}

fn ts_array(ts: Ts, rows: &[Row]) -> ArrayRef {
    match ts {
        Ts::I64 => Arc::new(Int64Array::from_iter(rows.iter().map(|r| r.ts))),
        Ts::I32 => Arc::new(Int32Array::from_iter(rows.iter().map(|r| r.ts.map(|t| t as i32)))),
        Ts::TsNs => Arc::new(TimestampNanosecondArray::from_iter(rows.iter().map(|r| r.ts))),
    }
}

fn group_type(by: By) -> DataType {
    match by {
        By::I32 => DataType::Int32,
        By::I64 => DataType::Int64,
        By::Str => DataType::Utf8,
        By::None => DataType::Null,
    }
}

fn ts_type(ts: Ts) -> DataType {
    match ts {
        Ts::I64 => DataType::Int64,
        Ts::I32 => DataType::Int32,
        Ts::TsNs => DataType::Timestamp(TimeUnit::Nanosecond, None),
    }
}

fn write_parquet(path: &std::path::Path, c: &Case, rows: &[Row], payload_name: &str) {
    let mut fields = vec![Field::new("id", DataType::Int64, false)];
    let mut cols: Vec<ArrayRef> =
        vec![Arc::new(Int64Array::from_iter_values(rows.iter().map(|r| r.id)))];
    if let Some(g) = group_array(c.by, rows) {
        fields.push(Field::new("g", group_type(c.by), true));
        cols.push(g);
    }
    fields.push(Field::new("ts", ts_type(c.ts), true));
    cols.push(ts_array(c.ts, rows));
    // payload: identifies the row
    fields.push(Field::new(payload_name, DataType::Float64, false));
    cols.push(Arc::new(Float64Array::from_iter_values(rows.iter().map(|r| r.id as f64))));
    let schema = Arc::new(Schema::new(fields));
    let batch = RecordBatch::try_new(Arc::clone(&schema), cols).unwrap();
    let props = WriterProperties::builder()
        .set_max_row_group_row_count(Some(c.row_group))
        .build();
    let mut w = ArrowWriter::try_new(File::create(path).unwrap(), schema, Some(props)).unwrap();
    w.write(&batch).unwrap();
    w.close().unwrap();
}

/// Brute force: last right row (file order) of the same group with the
/// greatest ts <= (or <) left ts.
fn reference(c: &Case, left: &[Row], right: &[Row]) -> Vec<(i64, Option<i64>)> {
    left.iter()
        .map(|l| {
            let ans = match (l.ts, c.by == By::None || l.g.is_some()) {
                (Some(lt), true) => {
                    let mut best: Option<(i64, i64)> = None; // (ts, id)
                    for r in right {
                        let (Some(rt), true) = (r.ts, c.by == By::None || r.g.is_some()) else {
                            continue;
                        };
                        if c.by != By::None && r.g != l.g {
                            continue;
                        }
                        let ok = if c.strict { rt < lt } else { rt <= lt };
                        if ok && best.is_none_or(|(bt, _)| rt >= bt) {
                            best = Some((rt, r.id));
                        }
                    }
                    best.map(|(_, id)| id)
                }
                _ => None,
            };
            (l.id, ans)
        })
        .collect()
}

fn options() -> AsofOptions {
    AsofOptions {
        right_split_factor: 8,
        max_sources: 3, // force frequent compaction
        // Many small left work units through the ordered output, with a
        // window narrower than the partition count.
        left_split_factor: 4,
        ordered_output: true,
        window_extra: 1,
        // FANOUT=1 forces the sparse-left fan-out on every partition.
        fanout_below: std::env::var("FANOUT").ok().map(|_| usize::MAX),
        ..AsofOptions::default()
    }
}

fn context() -> SessionContext {
    // Tiny files: force byte-range splitting and many small batches.
    let config = SessionConfig::new()
        .with_target_partitions(4)
        .with_batch_size(64)
        .set_usize("datafusion.optimizer.repartition_file_min_size", 1);
    SessionContext::new_with_config(config)
}

type Rows = Vec<(i64, Option<i64>)>;

fn rows_of(batches: &[RecordBatch]) -> Rows {
    let mut out = Vec::new();
    for b in batches {
        let id = b.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
        let rid = b.column(1).as_any().downcast_ref::<Float64Array>().unwrap();
        for i in 0..b.num_rows() {
            let v = if rid.is_null(i) { None } else { Some(rid.value(i) as i64) };
            out.push((id.value(i), v));
        }
    }
    out
}

fn by_cols(c: &Case) -> Vec<&'static str> {
    if c.by == By::None { vec![] } else { vec!["g"] }
}

fn sort_cols(c: &Case) -> Vec<&'static str> {
    if c.by == By::None { vec!["ts"] } else { vec!["g", "ts"] }
}

async fn read(ctx: &SessionContext, c: &Case, path: &std::path::Path, ordered: bool) -> datafusion::dataframe::DataFrame {
    let mut opts = ParquetReadOptions::default();
    if ordered {
        opts = opts.file_sort_order(vec![sort_cols(c).iter().map(|s| col(*s).sort(true, true)).collect()]);
    }
    ctx.read_parquet(path.to_str().unwrap(), opts).await.unwrap()
}

fn join(c: &Case) -> JoinAsof {
    JoinAsof::new("ts")
        .by(by_cols(c))
        .allow_exact_matches(!c.strict)
        .options(options())
}

fn select_id_rid(plan: Arc<dyn ExecutionPlan>) -> Arc<dyn ExecutionPlan> {
    use datafusion::physical_expr::expressions::Column;
    use datafusion::physical_plan::projection::ProjectionExec;
    let s = plan.schema();
    let exprs: Vec<(Arc<dyn datafusion::physical_expr::PhysicalExpr>, String)> = ["id", "rid"]
        .iter()
        .map(|n| (Arc::new(Column::new(n, s.index_of(n).unwrap())) as _, n.to_string()))
        .collect();
    Arc::new(ProjectionExec::try_new(exprs, plan).unwrap())
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum Input {
    /// Ordered scans: range-parallel mode.
    Scans,
    /// The left scan merged into one partition first (not a scan any more):
    /// execute-once with right range units.
    MergedLeft,
    /// No declared order: validated while streaming, execute-once.
    Unordered,
}

async fn run_builder(c: &Case, dir: &std::path::Path, input: Input) -> Result<(Rows, String), String> {
    let ctx = context();
    let ordered = input != Input::Unordered;
    let l = read(&ctx, c, &dir.join("left.parquet"), ordered).await;
    let r = read(&ctx, c, &dir.join("right.parquet"), ordered).await;
    let mut left = l.create_physical_plan().await.unwrap();
    let right = r.create_physical_plan().await.unwrap();
    if input == Input::MergedLeft {
        let ord: LexOrdering = left.output_ordering().unwrap().clone();
        left = Arc::new(SortPreservingMergeExec::new(ord, left));
    }
    let plan = join(c)
        .build(left, right, ctx.state().config_options())
        .map_err(|e| e.to_string())?;
    let plan = select_id_rid(plan);
    let plan_str = displayable(plan.as_ref()).indent(true).to_string();
    let batches = collect(plan, ctx.task_ctx()).await.map_err(|e| e.to_string())?;
    Ok((rows_of(&batches), plan_str))
}

/// The serialized-logical-plan provider.
async fn run_plan_bytes(c: &Case, dir: &std::path::Path) -> (Rows, String) {
    let ctx = context();
    let l = read(&ctx, c, &dir.join("left.parquet"), true).await;
    let r = read(&ctx, c, &dir.join("right.parquet"), true).await;
    let lb = datafusion_proto::bytes::logical_plan_to_bytes(l.logical_plan()).unwrap();
    let rb = datafusion_proto::bytes::logical_plan_to_bytes(r.logical_plan()).unwrap();
    let table = AsofPlanTable::try_new(lb.to_vec(), rb.to_vec(), join(c), None).unwrap();
    let schema = table.schema();
    let proj = vec![schema.index_of("id").unwrap(), schema.index_of("rid").unwrap()];
    let plan = table.plan(ctx.state().config_options(), Some(&proj)).await.unwrap();
    let plan_str = displayable(plan.as_ref()).indent(true).to_string();
    (rows_of(&collect(plan, ctx.task_ctx()).await.unwrap()), plan_str)
}

fn write_case(c: &Case, dir: &std::path::Path) -> (Vec<Row>, Vec<Row>) {
    let mut rng = Rng(c.seed.max(1));
    let mut left = gen_rows(&mut rng, c.n_left, c.left_groups, c, 0);
    let mut right = gen_rows(&mut rng, c.n_right, c.right_groups, c, 1_000_000);
    sort_rows(&mut left, c.by);
    sort_rows(&mut right, c.by);
    write_parquet(&dir.join("left.parquet"), c, &left, "lpx");
    write_parquet(&dir.join("right.parquet"), c, &right, "rid");
    (left, right)
}

async fn check(c: Case) {
    let dir = tempfile::tempdir().unwrap();
    let (left, right) = write_case(&c, dir.path());
    // Same rows, same order as the left input.
    let expect = reference(&c, &left, &right);
    let check_one = |name: &str, got: &[(i64, Option<i64>)], plan: &str, once: bool| {
        assert!(plan.contains("StreamingAsOfJoinExec"), "{name} {c:?}\n{plan}");
        assert_eq!(plan.contains("execute_once"), once, "{name} mode, {c:?}\n{plan}");
        assert!(!plan.contains("SortExec"), "{name}: unexpected sort\n{plan}");
        assert_eq!(got.len(), expect.len(), "{name} row count, {c:?}\n{plan}");
        if plan.contains("SortPreservingMergeExec") {
            // DataFusion's merge is not stable for equal keys across
            // partitions: the merged left stream reorders tied left rows, and
            // the join keeps that stream's order. Compare per left row.
            let want: std::collections::HashMap<i64, Option<i64>> = expect.iter().cloned().collect();
            for (id, rid) in got {
                assert_eq!(want.get(id), Some(rid), "{name}: left id {id}, case {c:?}\n{plan}");
            }
        } else {
            for (i, (g, e)) in got.iter().zip(&expect).enumerate() {
                assert_eq!(g, e, "{name}: row {i}, case {c:?}\n{plan}");
            }
        }
    };
    let (got, plan) = run_builder(&c, dir.path(), Input::Scans).await.unwrap();
    check_one("scans", &got, &plan, false);
    assert!(plan.contains("OrderedConcatExec"), "{c:?}\n{plan}");
    for bad in ["RepartitionExec", "SortPreservingMergeExec", "CoalescePartitionsExec"] {
        assert!(!plan.contains(bad), "unexpected {bad} in plan for {c:?}\n{plan}");
    }
    let (got, plan) = run_builder(&c, dir.path(), Input::MergedLeft).await.unwrap();
    check_one("merged left", &got, &plan, true);
    assert!(plan.contains("right units"), "{c:?}\n{plan}");
    let (got, plan) = run_builder(&c, dir.path(), Input::Unordered).await.unwrap();
    check_one("unordered", &got, &plan, true);
    let (got, plan) = run_plan_bytes(&c, dir.path()).await;
    check_one("plan bytes", &got, &plan, false);
}

fn base(seed: u64) -> Case {
    Case {
        seed,
        by: By::I32,
        ts: Ts::I64,
        n_left: 700,
        n_right: 2500,
        left_groups: 12,
        right_groups: 9,
        ts_range: 400,
        null_per_mille: 20,
        strict: false,
        row_group: 37,
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn grouped_i32() {
    for seed in 1..6 {
        check(base(seed)).await;
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn grouped_i64_ties_and_strict() {
    for seed in 10..14 {
        check(Case { by: By::I64, ts_range: 30, ..base(seed) }).await;
        check(Case { by: By::I64, strict: true, ts_range: 30, ..base(seed) }).await;
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn grouped_string_rows_path() {
    for seed in 20..23 {
        check(Case { by: By::Str, ..base(seed) }).await;
        check(Case { by: By::Str, strict: true, ts_range: 30, ..base(seed) }).await;
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn ungrouped() {
    for seed in 40..44 {
        check(Case { by: By::None, ..base(seed) }).await;
        check(Case { by: By::None, ts_range: 20, strict: true, ..base(seed) }).await;
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn other_ts_types() {
    check(Case { ts: Ts::I32, ..base(50) }).await;
    check(Case { ts: Ts::TsNs, ..base(51) }).await;
    check(Case { ts: Ts::I32, by: By::None, ..base(52) }).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn sparse_left_and_empty_sides() {
    // Sparse left (exercises partition jumps), dense right.
    check(Case { n_left: 15, n_right: 4000, ..base(60) }).await;
    check(Case { n_left: 15, n_right: 4000, by: By::None, ..base(64) }).await;
    // Right much smaller than left; groups mostly without right rows.
    check(Case { n_left: 1500, n_right: 40, right_groups: 3, ..base(61) }).await;
    // No nulls at all.
    check(Case { null_per_mille: 0, ..base(62) }).await;
    check(Case { null_per_mille: 0, by: By::None, ..base(63) }).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn unsorted_input_is_an_error() {
    let c = base(70);
    let dir = tempfile::tempdir().unwrap();
    let (mut left, right) = write_case(&c, dir.path());
    // Swap two far-apart valid rows: unsorted, not visible at boundaries only.
    let n = left.len();
    left.swap(n / 3, 2 * n / 3);
    write_parquet(&dir.path().join("left.parquet"), &c, &left, "lpx");
    let _ = right;
    let err = run_builder(&c, dir.path(), Input::Unordered).await.unwrap_err();
    assert!(err.contains("left input is not sorted by (g, ts)"), "{err}");
}
