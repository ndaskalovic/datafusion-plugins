//! `join_asof_markouts` against a brute-force reference: grouped and
//! ungrouped, ms/us/ns timestamps (with and without a time zone), negative,
//! zero, fractional and unsorted markout lists, ties, NULLs, groups without
//! right rows, sparse left, strict matches, horizons longer than a group's
//! span, timestamps near the i64 limits, and the error cases.

mod common;

use std::sync::Arc;

use arrow::array::{
    Array, ArrayRef, Float64Array, Int32Array, Int64Array, RecordBatch, StringArray, TimestampMicrosecondArray,
    TimestampMillisecondArray, TimestampNanosecondArray,
};
use arrow::datatypes::{DataType, Field, Schema, TimeUnit};
use datafusion::catalog::TableProvider;
use datafusion::physical_plan::{ExecutionPlan, collect, displayable};
use datafusion::prelude::{ParquetReadOptions, SessionContext, col};
use datafusion_plugins::{JoinAsof, Markout, NodeTable, PlanInputs};

use common::{Rng, write_parquet};

#[derive(Clone, Copy, Debug, PartialEq)]
enum By {
    None,
    I32,
    I64,
    /// Utf8 group ("g007": string order = numeric order).
    Str,
    /// Two columns: g (Int64) and h (Utf8, a function of g).
    Multi,
}

#[derive(Clone, Debug)]
struct Case {
    seed: u64,
    by: By,
    unit: TimeUnit,
    tz: bool,
    markouts: Vec<&'static str>,
    n_left: usize,
    n_right: usize,
    left_groups: i64,
    right_groups: i64,
    /// ts values are base + rng.below(ts_range) * step (in the unit).
    base: i64,
    step: i64,
    ts_range: i64,
    null_per_mille: u64,
    strict: bool,
    row_group: usize,
}

#[derive(Clone, Debug)]
struct Row {
    id: i64,
    g: Option<i64>,
    ts: Option<i64>,
}

fn gen_rows(rng: &mut Rng, n: usize, groups: i64, c: &Case, id_base: i64) -> Vec<Row> {
    let mut rows: Vec<Row> = (0..n)
        .map(|i| {
            let g = if c.by != By::None && rng.below(1000) < c.null_per_mille {
                None
            } else {
                Some(rng.below(groups as u64) as i64)
            };
            let ts = if rng.below(1000) < c.null_per_mille {
                None
            } else {
                Some(c.base.saturating_add((rng.below(c.ts_range as u64) as i64).saturating_mul(c.step)))
            };
            Row { id: id_base + i as i64, g, ts }
        })
        .collect();
    // Sorted by (g, ts) (or ts), NULLS FIRST, stable.
    if c.by == By::None {
        rows.sort_by_key(|r| r.ts);
    } else {
        rows.sort_by_key(|r| (r.g, r.ts));
    }
    rows
}

fn digits(u: TimeUnit) -> u32 {
    match u {
        TimeUnit::Second => 0,
        TimeUnit::Millisecond => 3,
        TimeUnit::Microsecond => 6,
        TimeUnit::Nanosecond => 9,
    }
}

fn ts_type(c: &Case) -> DataType {
    DataType::Timestamp(c.unit, c.tz.then(|| "UTC".into()))
}

fn ts_array(c: &Case, rows: &[Row]) -> ArrayRef {
    let v = rows.iter().map(|r| r.ts);
    let tz = c.tz.then_some("UTC");
    match c.unit {
        TimeUnit::Millisecond => Arc::new(TimestampMillisecondArray::from_iter(v).with_timezone_opt(tz)),
        TimeUnit::Microsecond => Arc::new(TimestampMicrosecondArray::from_iter(v).with_timezone_opt(tz)),
        _ => Arc::new(TimestampNanosecondArray::from_iter(v).with_timezone_opt(tz)),
    }
}

fn write_case_file(path: &std::path::Path, c: &Case, rows: &[Row], payload: &str) {
    let mut fields = vec![Field::new("id", DataType::Int64, false)];
    let mut cols: Vec<ArrayRef> = vec![Arc::new(Int64Array::from_iter_values(rows.iter().map(|r| r.id)))];
    match c.by {
        By::None => {}
        By::I32 => {
            fields.push(Field::new("g", DataType::Int32, true));
            cols.push(Arc::new(Int32Array::from_iter(rows.iter().map(|r| r.g.map(|g| g as i32)))));
        }
        By::I64 => {
            fields.push(Field::new("g", DataType::Int64, true));
            cols.push(Arc::new(Int64Array::from_iter(rows.iter().map(|r| r.g))));
        }
        By::Str => {
            fields.push(Field::new("g", DataType::Utf8, true));
            cols.push(Arc::new(StringArray::from_iter(rows.iter().map(|r| r.g.map(|g| format!("g{g:03}"))))));
        }
        By::Multi => {
            fields.push(Field::new("g", DataType::Int64, true));
            cols.push(Arc::new(Int64Array::from_iter(rows.iter().map(|r| r.g))));
            fields.push(Field::new("h", DataType::Utf8, true));
            cols.push(Arc::new(StringArray::from_iter(rows.iter().map(|r| r.g.map(|g| format!("h{}", g % 2))))));
        }
    }
    fields.push(Field::new("ts", ts_type(c), true));
    cols.push(ts_array(c, rows));
    fields.push(Field::new(payload, DataType::Float64, false));
    cols.push(Arc::new(Float64Array::from_iter_values(rows.iter().map(|r| r.id as f64))));
    let batch = RecordBatch::try_new(Arc::new(Schema::new(fields)), cols).unwrap();
    write_parquet(path, &batch, c.row_group);
}

/// Shift of markout `m` (decimal seconds) in the unit, exactly.
fn shift(m: &str, u: TimeUnit) -> i128 {
    let neg = m.starts_with('-');
    let m = m.trim_start_matches('-');
    let (i, f) = m.split_once('.').unwrap_or((m, ""));
    let d = digits(u) as usize;
    assert!(f.len() <= d);
    let s: i128 = format!("{i}{f}{}", "0".repeat(d - f.len())).parse().unwrap();
    if neg { -s } else { s }
}

/// Brute force, in i128 (no overflow): per left row, per markout, the last
/// right row (file order) of the same group with the greatest ts <= (<)
/// left ts + h.
fn reference(c: &Case, left: &[Row], right: &[Row]) -> Vec<Vec<Option<i64>>> {
    let shifts: Vec<i128> = c.markouts.iter().map(|m| shift(m, c.unit)).collect();
    left.iter()
        .map(|l| {
            shifts
                .iter()
                .map(|h| {
                    let (Some(lt), true) = (l.ts, c.by == By::None || l.g.is_some()) else {
                        return None;
                    };
                    let target = lt as i128 + h;
                    let mut best: Option<(i64, i64)> = None;
                    for r in right {
                        let (Some(rt), true) = (r.ts, c.by == By::None || r.g.is_some()) else {
                            continue;
                        };
                        if c.by != By::None && r.g != l.g {
                            continue;
                        }
                        let ok = if c.strict { (rt as i128) < target } else { (rt as i128) <= target };
                        if ok && best.is_none_or(|(bt, _)| rt >= bt) {
                            best = Some((rt, r.id));
                        }
                    }
                    best.map(|b| b.1)
                })
                .collect()
        })
        .collect()
}

fn context() -> SessionContext {
    common::context(4, 64)
}

fn node(c: &Case) -> JoinAsof {
    let ms = c.markouts.iter().map(|m| Markout::parse(m).unwrap()).collect();
    let n = JoinAsof::new("ts").markouts(ms).assume_sorted(true).allow_exact_matches(!c.strict);
    match c.by {
        By::None => n,
        By::Multi => n.by(["g", "h"]),
        _ => n.by(["g"]),
    }
}

fn label(m: &str) -> String {
    Markout::parse(m).unwrap().label()
}

/// Rows of (left id, [match id per markout]).
fn rows_of(c: &Case, batches: &[RecordBatch]) -> Vec<(i64, Vec<Option<i64>>)> {
    let mut out = Vec::new();
    for b in batches {
        let s = b.schema();
        let id = b.column(s.index_of("id").unwrap()).as_any().downcast_ref::<Int64Array>().unwrap();
        let cols: Vec<&Float64Array> = c
            .markouts
            .iter()
            .map(|m| {
                let i = s.index_of(&format!("rid_{}s", label(m))).unwrap();
                b.column(i).as_any().downcast_ref::<Float64Array>().unwrap()
            })
            .collect();
        for i in 0..b.num_rows() {
            let v = cols.iter().map(|a| (!a.is_null(i)).then(|| a.value(i) as i64)).collect();
            out.push((id.value(i), v));
        }
    }
    out
}

async fn read(ctx: &SessionContext, c: &Case, path: &std::path::Path, ordered: bool) -> datafusion::dataframe::DataFrame {
    let mut o = ParquetReadOptions::default();
    if ordered {
        let keys: Vec<&str> = match c.by {
            By::None => vec!["ts"],
            By::Multi => vec!["g", "h", "ts"],
            _ => vec!["g", "ts"],
        };
        o = o.file_sort_order(vec![keys.iter().map(|k| col(*k).sort(true, true)).collect()]);
    }
    ctx.read_parquet(path.to_str().unwrap(), o).await.unwrap()
}

async fn run(c: &Case, dir: &std::path::Path, ordered: bool) -> Result<(Vec<(i64, Vec<Option<i64>>)>, String), String> {
    let ctx = context();
    let l = read(&ctx, c, &dir.join("left.parquet"), ordered).await;
    let r = read(&ctx, c, &dir.join("right.parquet"), ordered).await;
    let left = l.create_physical_plan().await.unwrap();
    let right = r.create_physical_plan().await.unwrap();
    let plan = node(c).build(left, right, ctx.state().config_options()).map_err(|e| e.to_string())?;
    let s = displayable(plan.as_ref()).indent(true).to_string();
    let b = collect(plan, ctx.task_ctx()).await.map_err(|e| e.to_string())?;
    Ok((rows_of(c, &b), s))
}

async fn run_plan_bytes(c: &Case, dir: &std::path::Path) -> Vec<(i64, Vec<Option<i64>>)> {
    let ctx = context();
    let l = read(&ctx, c, &dir.join("left.parquet"), true).await;
    let r = read(&ctx, c, &dir.join("right.parquet"), true).await;
    let lb = datafusion_proto::bytes::logical_plan_to_bytes(l.logical_plan()).unwrap();
    let rb = datafusion_proto::bytes::logical_plan_to_bytes(r.logical_plan()).unwrap();
    let inputs = PlanInputs::try_new(vec![lb.to_vec(), rb.to_vec()]).unwrap();
    let t = NodeTable::try_new(Arc::new(inputs), Arc::new(node(c))).unwrap();
    // Only id and the payloads: exercises the projection.
    let s = t.schema();
    let proj: Vec<usize> = std::iter::once("id".to_string())
        .chain(c.markouts.iter().map(|m| format!("rid_{}s", label(m))))
        .map(|n| s.index_of(&n).unwrap())
        .collect();
    let plan: Arc<dyn ExecutionPlan> = t.plan(ctx.state().config_options(), Some(&proj)).await.unwrap();
    rows_of(c, &collect(plan, ctx.task_ctx()).await.unwrap())
}

async fn check(c: Case) {
    let dir = tempfile::tempdir().unwrap();
    let mut rng = Rng(c.seed.max(1));
    let left = gen_rows(&mut rng, c.n_left, c.left_groups, &c, 0);
    let right = gen_rows(&mut rng, c.n_right, c.right_groups, &c, 1_000_000);
    write_case_file(&dir.path().join("left.parquet"), &c, &left, "lpx");
    write_case_file(&dir.path().join("right.parquet"), &c, &right, "rid");
    let want = reference(&c, &left, &right);
    let cmp = |name: &str, got: &[(i64, Vec<Option<i64>>)], plan: &str| {
        assert_eq!(got.len(), left.len(), "{name} {c:?}");
        for (i, ((id, v), (l, w))) in got.iter().zip(left.iter().zip(&want)).enumerate() {
            assert_eq!(*id, l.id, "{name}: row order, row {i}, {c:?}");
            assert_eq!(v, w, "{name}: row {i} (left id {id}), {c:?}\n{plan}");
        }
    };
    for ordered in [true, false] {
        let (got, plan) = run(&c, dir.path(), ordered).await.unwrap();
        assert!(plan.contains("AsofExec") && !plan.contains("SortExec"), "{plan}");
        cmp(if ordered { "declared" } else { "assumed" }, &got, &plan);
    }
    cmp("plan bytes", &run_plan_bytes(&c, dir.path()).await, "");
}

fn base(seed: u64) -> Case {
    Case {
        seed,
        by: By::I32,
        unit: TimeUnit::Millisecond,
        tz: false,
        markouts: vec!["1", "5", "60"],
        n_left: 700,
        n_right: 2500,
        left_groups: 12,
        right_groups: 9,
        base: 1_700_000_000_000,
        step: 1000, // one second
        ts_range: 400,
        null_per_mille: 20,
        strict: false,
        row_group: 37,
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn grouped_units_and_markouts() {
    check(base(1)).await;
    check(Case { markouts: vec!["60", "1", "5"], ..base(2) }).await; // unsorted list
    check(Case { markouts: vec!["-5", "0", "0.5", "2.25"], ..base(3) }).await;
    check(Case { markouts: vec!["7"], ..base(4) }).await; // single markout
    check(Case { unit: TimeUnit::Microsecond, step: 1_000_000, base: 1_700_000_000_000_000, markouts: vec!["0.000001", "-1.5", "30"], ..base(5) }).await;
    check(Case { unit: TimeUnit::Nanosecond, step: 1_000_000_000, base: 1_700_000_000_000_000_000, tz: true, markouts: vec!["0.000000001", "10", "-10"], ..base(6) }).await;
    check(Case { by: By::I64, tz: true, ..base(7) }).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn string_and_multi_column_by() {
    check(Case { by: By::Str, ..base(8) }).await;
    check(Case { by: By::Str, ts_range: 30, strict: true, markouts: vec!["-3", "0", "2"], ..base(9) }).await;
    check(Case { by: By::Multi, ..base(12) }).await;
    check(Case { by: By::Multi, null_per_mille: 100, markouts: vec!["10000", "-10000"], ..base(13) }).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn ungrouped() {
    check(Case { by: By::None, ..base(10) }).await;
    check(Case { by: By::None, markouts: vec!["0", "-30", "300"], strict: true, ts_range: 40, ..base(11) }).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn ties_nulls_strict_and_sparse() {
    // Many ties (ts_range 30), strict and not.
    check(Case { ts_range: 30, markouts: vec!["0", "1", "3"], ..base(20) }).await;
    check(Case { ts_range: 30, markouts: vec!["0", "1", "3"], strict: true, ..base(21) }).await;
    // Heavy NULLs.
    check(Case { null_per_mille: 150, ..base(22) }).await;
    // Groups without right rows; sparse left; tiny right.
    check(Case { left_groups: 20, right_groups: 5, ..base(23) }).await;
    check(Case { n_left: 15, n_right: 4000, ..base(24) }).await;
    check(Case { n_left: 15, n_right: 4000, by: By::None, ..base(25) }).await;
    check(Case { n_left: 1500, n_right: 40, right_groups: 3, ..base(26) }).await;
    // Horizons longer than any group's span (400 s): everything / nothing.
    check(Case { markouts: vec!["-10000", "10000", "5"], ..base(27) }).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn near_i64_limits() {
    // ts within ~400 s of i64::MAX / i64::MIN (ns), markouts that overflow.
    let ns = |s: i64| s * 1_000_000_000;
    let hi = Case {
        unit: TimeUnit::Nanosecond,
        step: ns(1),
        base: i64::MAX - ns(401),
        markouts: vec!["-100", "0", "300", "100000"],
        ..base(30)
    };
    check(hi.clone()).await;
    check(Case { strict: true, by: By::I64, ..hi }).await;
    let lo = Case {
        unit: TimeUnit::Nanosecond,
        step: ns(1),
        base: i64::MIN + 1,
        markouts: vec!["-100000", "-1", "0", "100"],
        ..base(31)
    };
    check(lo.clone()).await;
    check(Case { strict: true, by: By::None, ..lo }).await;
}

#[test]
fn markout_parsing() {
    let p = |s: &str| Markout::parse(s).unwrap().label();
    assert_eq!(p("1"), "1");
    assert_eq!(p("1.0"), "1");
    assert_eq!(p("0.50"), "0.5");
    assert_eq!(p("-5"), "-5");
    assert_eq!(p("-0.001"), "-0.001");
    assert_eq!(p("0"), "0");
    assert!(Markout::parse("1e3").is_err());
    assert!(Markout::parse("abc").is_err());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn errors() {
    let dir = tempfile::tempdir().unwrap();
    let c = base(40);
    let mut rng = Rng(9);
    let rows = gen_rows(&mut rng, 50, 3, &c, 0);
    write_case_file(&dir.path().join("left.parquet"), &c, &rows, "lpx");
    write_case_file(&dir.path().join("right.parquet"), &c, &rows, "rid");
    let us = Case { unit: TimeUnit::Microsecond, ..base(41) };
    write_case_file(&dir.path().join("right_us.parquet"), &us, &rows, "rid");
    let ctx = context();
    let l = ctx.read_parquet(dir.path().join("left.parquet").to_str().unwrap(), ParquetReadOptions::default()).await.unwrap();
    let r = ctx.read_parquet(dir.path().join("right.parquet").to_str().unwrap(), ParquetReadOptions::default()).await.unwrap();
    let r_us = ctx.read_parquet(dir.path().join("right_us.parquet").to_str().unwrap(), ParquetReadOptions::default()).await.unwrap();
    let lp = l.create_physical_plan().await.unwrap();
    let rp = r.create_physical_plan().await.unwrap();
    let rp_us = r_us.create_physical_plan().await.unwrap();
    let cfg = ctx.state().config_options().clone();
    let err = |n: JoinAsof, r: &Arc<dyn ExecutionPlan>| n.build(Arc::clone(&lp), Arc::clone(r), &cfg).unwrap_err().to_string();
    let m = |v: &[&str]| v.iter().map(|s| Markout::parse(s).unwrap()).collect::<Vec<_>>();
    assert!(err(JoinAsof::new("id").markouts(m(&["1"])), &rp).contains("must be a Timestamp"));
    assert!(err(JoinAsof::new("ts").markouts(m(&["1"])), &rp_us).contains("same type"));
    assert!(err(JoinAsof::new("ts").markouts(m(&["0.0005"])), &rp).contains("not a whole number"));
    assert!(err(JoinAsof::new("ts").markouts(m(&["1", "1.0"])), &rp).contains("duplicate markout 1"));
    assert!(err(JoinAsof::new("ts").markouts(m(&[])), &rp).contains("at least one"));
}
