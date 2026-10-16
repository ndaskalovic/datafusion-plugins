"""Both nodes after general upstream plans and on any input kind, vs polars.

The data has unique (g, ts) keys on each side, so the sorted order (and so
the output row order) is fully determined whatever the upstream does.
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.compute as pc
import pytest
from datafusion import SessionConfig, SessionContext, col, lit, udf
from datafusion import functions as F

import datafusion_plugins as dp

MARKOUTS = [-5, 0, 1, 30]


def make_ctx() -> SessionContext:
    config = (
        SessionConfig()
        .with_target_partitions(4)
        .with_batch_size(64)
        .set("datafusion.optimizer.repartition_file_min_size", "1")
        .set("datafusion.execution.enable_file_stream_work_stealing", "false")
    )
    return SessionContext(config)


def gen(n: int, groups: int, seed: int, id_base: int, payload: str) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    df = pl.DataFrame(
        {
            "id": np.arange(n) + id_base,
            "g": rng.integers(0, groups, n).astype(np.int32),
            "ts": (rng.integers(0, 2000, n) + 1_700_000_000) * 1_000_000,
            payload: (np.arange(n) + id_base).astype(np.float64),
        }
    ).with_columns(pl.col("ts").cast(pl.Datetime("us")))
    return df.unique(["g", "ts"], keep="first", maintain_order=True).sort(["g", "ts"])


@pytest.fixture
def data(tmp_path: Path):
    left = gen(800, 12, 1, 0, "lpx")
    right = gen(3000, 9, 2, 1_000_000, "rid")
    lp, rp = str(tmp_path / "l.parquet"), str(tmp_path / "r.parquet")
    left.write_parquet(lp, row_group_size=37)
    right.write_parquet(rp, row_group_size=37)
    return left, right, lp, rp, tmp_path


ORDER = [[col("g").sort(nulls_first=True), col("ts").sort(nulls_first=True)]]


def asof_ref(left: pl.DataFrame, right: pl.DataFrame) -> pl.DataFrame:
    return left.join_asof(right, on="ts", by="g", strategy="backward")


def markouts_ref(left: pl.DataFrame, right: pl.DataFrame) -> pl.DataFrame:
    payload = [n for n in right.columns if n not in ("g", "ts")]
    out = left
    for h in MARKOUTS:
        j = left.select(pl.col("ts") + pl.duration(seconds=h, time_unit="us"), "g").join_asof(
            right, on="ts", by="g", strategy="backward"
        )
        out = out.hstack(j.select([pl.col(n).alias(f"{n}_{h}s") for n in payload]))
    return out


def both(ctx, lf, rf, left_pl, right_pl, assume_sorted: bool = False, **kw) -> None:
    """Run both nodes and compare with polars (row order included)."""
    kw["assume_sorted"] = assume_sorted
    got = dp.join_asof(lf, rf, on="ts", by="g", ctx=ctx, **kw).to_polars()
    assert got.equals(asof_ref(left_pl, right_pl)), (got, asof_ref(left_pl, right_pl))
    got = dp.join_asof_markouts(lf, rf, on="ts", markouts=MARKOUTS, by="g", ctx=ctx, **kw).to_polars()
    assert got.equals(markouts_ref(left_pl, right_pl))


def test_hash_join_then_sort(data) -> None:
    left, right, lp, rp, tmp = data
    dim = pl.DataFrame({"g2": pl.Series(range(12), dtype=pl.Int32), "w": [float(i) for i in range(12)]})
    dp_ = str(tmp / "dim.parquet")
    dim.write_parquet(dp_)
    ctx = make_ctx()
    lf = (
        ctx.read_parquet(lp)
        .join(ctx.read_parquet(dp_), left_on="g", right_on="g2", how="inner")
        .drop("g2")
        .sort(col("g"), col("ts"))
    )
    rf = ctx.read_parquet(rp, file_sort_order=ORDER)
    lj = left.join(dim, left_on="g", right_on="g2", how="inner").sort(["g", "ts"])
    both(ctx, lf, rf, lj, right)
    assert "HashJoinExec" in dp.last_plan()


def test_aggregate_then_sort(data) -> None:
    left, right, lp, rp, _ = data
    ctx = make_ctx()
    lf = (
        ctx.read_parquet(lp)
        .aggregate([col("g"), col("ts")], [F.max(col("lpx")).alias("mx")])
        .sort(col("g"), col("ts"))
    )
    rf = ctx.read_parquet(rp, file_sort_order=ORDER)
    la = left.group_by(["g", "ts"]).agg(pl.col("lpx").max().alias("mx")).sort(["g", "ts"])
    both(ctx, lf, rf, la, right)


def test_union_of_sorted_sources(data) -> None:
    left, right, lp, rp, tmp = data
    # Two sorted halves of the right side with overlapping key ranges.
    a, b = right.filter(pl.col("id") % 2 == 0), right.filter(pl.col("id") % 2 == 1)
    ap, bp = str(tmp / "a.parquet"), str(tmp / "b.parquet")
    a.write_parquet(ap, row_group_size=37)
    b.write_parquet(bp, row_group_size=37)
    ctx = make_ctx()
    rf = ctx.read_parquet(ap, file_sort_order=ORDER).union(ctx.read_parquet(bp, file_sort_order=ORDER))
    lf = ctx.read_parquet(lp, file_sort_order=ORDER)
    both(ctx, lf, rf, left, right)
    assert "UnionExec" in dp.last_plan() and "SortExec" not in dp.last_plan()


def test_round_robin_repartitioned(data) -> None:
    left, right, lp, rp, _ = data
    ctx = make_ctx()
    lf = ctx.read_parquet(lp, file_sort_order=ORDER).repartition(3)
    rf = ctx.read_parquet(rp, file_sort_order=ORDER).repartition(5)
    both(ctx, lf, rf, left, right)  # required order: DataFusion keeps or sorts
    both(ctx, lf, rf, left, right, assume_sorted=True)  # each partition is sorted


def test_computed_ts(data) -> None:
    left, right, lp, rp, _ = data
    ctx = make_ctx()
    lf = (
        ctx.read_parquet(lp, file_sort_order=ORDER)
        .filter(col("lpx") > lit(50.0))
        .with_column("ts", col("ts") + lit(timedelta(seconds=1)))
    )
    rf = ctx.read_parquet(rp, file_sort_order=ORDER)
    lc = left.filter(pl.col("lpx") > 50.0).with_columns(pl.col("ts") + timedelta(seconds=1))
    both(ctx, lf, rf, lc, right, assume_sorted=True)
    assert "SortExec" not in dp.last_plan()
    # Without assume_sorted, DataFusion can't prove ts + 1s sorted: it sorts.
    got = dp.join_asof(lf, rf, on="ts", by="g", ctx=ctx).to_polars()
    assert "SortExec" in dp.last_plan()
    assert got.equals(asof_ref(lc, right))


def test_unsorted_with_assume_sorted_raises(data) -> None:
    left, right, lp, rp, tmp = data
    sp = str(tmp / "shuffled.parquet")
    left.sample(fraction=1.0, shuffle=True, seed=3).write_parquet(sp, row_group_size=37)
    ctx = make_ctx()
    rf = ctx.read_parquet(rp, file_sort_order=ORDER)
    for f in (dp.join_asof, lambda *a, **k: dp.join_asof_markouts(*a, markouts=[1], **k)):
        with pytest.raises(Exception, match="not sorted"):
            f(ctx.read_parquet(sp), rf, on="ts", by="g", assume_sorted=True, ctx=ctx).collect()


# ---------------------------------------------------------------------------
# Inputs datafusion-python executes (streamed to the plugin)
# ---------------------------------------------------------------------------


def test_in_memory_inputs(data) -> None:
    left, right, _, _, _ = data
    ctx = make_ctx()
    for make in (
        lambda df: ctx.from_polars(df),
        lambda df: ctx.from_arrow(df.to_arrow()),
        lambda df: ctx.from_pydict(df.to_dict(as_series=False)),
    ):
        lf, rf = make(left), make(right)
        if "ts" in lf.schema().names and lf.schema().field("ts").type != pa.timestamp("us"):
            lf = lf.with_column("ts", col("ts").cast(pa.timestamp("us")))
            rf = rf.with_column("ts", col("ts").cast(pa.timestamp("us")))
        both(ctx, lf, rf, left, right)
        assert "HostInputExec" in dp.last_plan()


def test_multi_partition_in_memory(data) -> None:
    left, right, _, _, _ = data
    ctx = make_ctx()

    def parts(df: pl.DataFrame, n: int) -> list[list[pa.RecordBatch]]:
        # n consecutive sorted ranges, a few batches each.
        k = -(-df.height // n)
        return [df[i * k : (i + 1) * k].to_arrow().to_batches(max_chunksize=50) for i in range(n)]

    # As many partitions as target_partitions: kept as given, each sorted.
    lf, rf = ctx.create_dataframe(parts(left, 4)), ctx.create_dataframe(parts(right, 4))
    both(ctx, lf, rf, left, right, assume_sorted=True)
    both(ctx, lf, rf, left, right)  # order not visible to the plugin: it sorts
    # Fewer partitions: datafusion-python re-splits them by size, which loses
    # the order inside partitions; letting the node sort still works.
    lf, rf = ctx.create_dataframe(parts(left, 3)), ctx.create_dataframe(parts(right, 2))
    both(ctx, lf, rf, left, right)
    # Round-robin over one sorted in-memory partition: each partition sorted.
    both(ctx, ctx.from_polars(left).repartition(4), ctx.from_polars(right).repartition(3), left, right,
         assume_sorted=True)


def test_python_udf_input(data) -> None:
    left, right, lp, rp, _ = data
    ctx = make_ctx()
    double = udf(lambda a: pc.multiply(a, 2.0), [pa.float64()], pa.float64(), "stable", name="double")
    lf = ctx.read_parquet(lp, file_sort_order=ORDER).with_column("lpx2", double(col("lpx")))
    rf = ctx.read_parquet(rp, file_sort_order=ORDER)
    both(ctx, lf, rf, left.with_columns(lpx2=pl.col("lpx") * 2.0), right, assume_sorted=True)
    assert "HostInputExec" in dp.last_plan()


def test_csv_source(data) -> None:
    left, right, _, _, tmp = data
    lc, rc = str(tmp / "l.csv"), str(tmp / "r.csv")
    # CSV has no timestamp type: write epoch micros, cast after reading.
    left.with_columns(pl.col("ts").dt.epoch("us")).write_csv(lc)
    right.with_columns(pl.col("ts").dt.epoch("us")).write_csv(rc)
    ctx = make_ctx()
    cast = lambda df: df.with_column("ts", col("ts").cast(pa.timestamp("us")))  # noqa: E731
    both(ctx, cast(ctx.read_csv(lc)), cast(ctx.read_csv(rc)), left, right)


def test_routes_agree(data, monkeypatch) -> None:
    left, right, lp, rp, _ = data
    ctx = make_ctx()
    lf, rf = ctx.read_parquet(lp, file_sort_order=ORDER), ctx.read_parquet(rp, file_sort_order=ORDER)
    for route, marker in (("plan", False), ("stream", True)):
        monkeypatch.setenv("DATAFUSION_PLUGINS_ROUTE", route)
        both(ctx, lf, rf, left, right, assume_sorted=True)
        assert ("HostInputExec" in dp.last_plan()) == marker


def test_string_and_multi_column_by_markouts(data) -> None:
    left, right, _, _, _ = data
    ctx = make_ctx()
    s = lambda df: df.with_columns(gs=pl.format("g{}", pl.col("g").cast(pl.Utf8).str.zfill(3)), h=pl.col("g") % 2)  # noqa: E731
    ls_, rs_ = s(left), s(right)
    got = dp.join_asof_markouts(ctx.from_polars(ls_), ctx.from_polars(rs_.drop("g")), on="ts",
                                markouts=MARKOUTS, by_left=["gs", "h"], by_right=["gs", "h"], ctx=ctx).to_polars()
    payload = ["id", "rid"]
    want = ls_
    for h in MARKOUTS:
        j = ls_.select(pl.col("ts") + pl.duration(seconds=h, time_unit="us"), "gs", "h").join_asof(
            rs_.drop("g"), on="ts", by=["gs", "h"], strategy="backward")
        want = want.hstack(j.select([pl.col(n).alias(f"{n}_{h}s") for n in payload]))
    assert got.equals(want)


def test_plugin_memory_limit(data) -> None:
    left, right, lp, rp, _ = data
    ctx = make_ctx()
    lf, rf = ctx.read_parquet(lp, file_sort_order=ORDER), ctx.read_parquet(rp, file_sort_order=ORDER)
    try:
        for limit, pool in ((256 << 10, "fair"), (4 << 20, "greedy")):
            dp.set_memory_limit(limit, pool)
            both(ctx, lf, rf, left, right)
            both(ctx, ctx.from_polars(left), ctx.from_polars(right), left, right, assume_sorted=True)
    finally:
        dp.set_memory_limit(None)
