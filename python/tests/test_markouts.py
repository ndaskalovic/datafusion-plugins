"""join_asof_markouts vs polars: N shifted join_asofs side by side."""

from __future__ import annotations

import dataclasses
from datetime import timedelta
from decimal import Decimal
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from datafusion import SessionConfig, SessionContext, col, lit

import datafusion_plugins as dp

UNIT_PER_S = {"ms": 1_000, "us": 1_000_000, "ns": 1_000_000_000}


def make_ctx() -> SessionContext:
    config = (
        SessionConfig()
        .with_target_partitions(4)
        .with_batch_size(64)
        .set("datafusion.optimizer.repartition_file_min_size", "1")
    )
    return SessionContext(config)


@dataclasses.dataclass(frozen=True)
class Case:
    seed: int
    markouts: tuple = (1, 5, 60)
    by: str | None = "i32"
    unit: str = "us"
    tz: str | None = None
    n_left: int = 700
    n_right: int = 2500
    left_groups: int = 12
    right_groups: int = 9
    span_s: int = 400  # ts drawn from [0, span_s) seconds (whole seconds: ties)
    null_frac: float = 0.0
    exact: bool = True
    row_group: int = 37


def gen(c: Case, n: int, groups: int, seed: int, id_base: int, payload: str) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    cols = {"id": pl.Series("id", np.arange(n) + id_base)}
    if c.by:
        g = pl.Series("g", rng.integers(0, groups, n)).cast(pl.Int32 if c.by == "i32" else pl.Int64)
        if c.null_frac:
            g = g.scatter(np.flatnonzero(rng.random(n) < c.null_frac), None)
        cols["g"] = g
    ts = pl.Series("ts", rng.integers(0, c.span_s, n) * UNIT_PER_S[c.unit] + 1_700_000_000 * UNIT_PER_S[c.unit])
    if c.null_frac:
        ts = ts.scatter(np.flatnonzero(rng.random(n) < c.null_frac), None)
    cols["ts"] = ts.cast(pl.Datetime(c.unit, c.tz))
    cols[payload] = pl.Series(payload, (np.arange(n) + id_base).astype(np.float64))
    cols["px"] = pl.Series("px", rng.random(n))
    df = pl.DataFrame(list(cols.values()))
    keys = ["g", "ts"] if c.by else ["ts"]
    return df.sort(keys, nulls_last=False, maintain_order=True)


def write(c: Case, tmp: Path):
    left = gen(c, c.n_left, c.left_groups, c.seed, 0, "lpx")
    right = gen(c, c.n_right, c.right_groups, c.seed + 1000, 1_000_000, "rid")
    lp, rp = str(tmp / "left.parquet"), str(tmp / "right.parquet")
    left.write_parquet(lp, row_group_size=c.row_group)
    right.write_parquet(rp, row_group_size=c.row_group)
    return left, right, lp, rp


def label(h) -> str:
    d = Decimal(h) if isinstance(h, int) else Decimal(repr(h))
    return format(d.normalize(), "f") if d != 0 else "0"


def polars_markouts(c: Case, left: pl.DataFrame, right: pl.DataFrame) -> pl.DataFrame:
    """N shifted polars join_asofs, concatenated horizontally."""
    by = ["g"] if c.by else None
    payload = [n for n in right.columns if n not in ("ts", "g")]
    kw = {"ms": "milliseconds", "us": "microseconds", "ns": "nanoseconds"}[c.unit]
    out = left
    for h in c.markouts:
        k = Decimal(h) if isinstance(h, int) else Decimal(repr(h))
        shift = pl.duration(**{kw: int(k * UNIT_PER_S[c.unit])}, time_unit=c.unit)
        shifted = left.select(pl.col("ts") + shift, *([pl.col("g")] if by else []))
        joined = shifted.join_asof(
            right, on="ts", by=by, strategy="backward", allow_exact_matches=c.exact
        )
        out = out.hstack(joined.select([pl.col(n).alias(f"{n}_{label(h)}s") for n in payload]))
    return out


def run(c: Case, ctx: SessionContext, lf, rf):
    return dp.join_asof_markouts(
        lf, rf, on="ts", markouts=list(c.markouts), by="g" if c.by else None,
        allow_exact_matches=c.exact, ctx=ctx,
    )


def read_sorted(ctx: SessionContext, path: str, c: Case):
    keys = ["g", "ts"] if c.by else ["ts"]
    return ctx.read_parquet(path, file_sort_order=[[col(k).sort(ascending=True, nulls_first=True) for k in keys]])


CASES = [
    Case(seed=1),
    Case(seed=2, markouts=(60, 1, 5)),  # unsorted list
    Case(seed=3, markouts=(-5, 0, 0.5, 2.25)),  # negative, zero, fractional
    Case(seed=4, markouts=(7,)),  # single markout
    Case(seed=5, unit="ms", markouts=(0.001, -1.5, 30)),
    Case(seed=6, unit="ns", tz="UTC", markouts=(0.000000001, 10, -10)),
    Case(seed=7, unit="us", tz="Europe/London", by="i64"),
    Case(seed=8, by=None),
    Case(seed=9, by=None, markouts=(0, -30, 300), exact=False, span_s=40),
    Case(seed=10, span_s=30, markouts=(0, 1, 3)),  # ties
    Case(seed=11, span_s=30, markouts=(0, 1, 3), exact=False),
    Case(seed=12, null_frac=0.05),
    Case(seed=13, left_groups=20, right_groups=5),  # groups with no right rows
    Case(seed=14, n_left=15, n_right=4000),  # sparse left
    Case(seed=15, markouts=(-10000, 10000, 5)),  # longer than any group's span
]


def case_id(c: Case) -> str:
    return f"s{c.seed}-{c.by}-{c.unit}-{c.tz}-{'_'.join(map(str, c.markouts))}-{'ge' if c.exact else 'gt'}-n{c.null_frac}"


@pytest.mark.parametrize("declared", [True, False], ids=["declared", "undeclared"])
@pytest.mark.parametrize("c", CASES, ids=case_id)
def test_matches_polars(c: Case, declared: bool, tmp_path: Path) -> None:
    left, right, lp, rp = write(c, tmp_path)
    ctx = make_ctx()
    lf, rf = (read_sorted(ctx, lp, c), read_sorted(ctx, rp, c)) if declared else (ctx.read_parquet(lp), ctx.read_parquet(rp))
    got = run(c, ctx, lf, rf).to_polars()
    plan = dp.last_plan()
    assert "MarkoutsExec" in plan and "SortExec" not in plan, plan
    want = polars_markouts(c, left, right)
    assert got.columns == want.columns
    assert got.equals(want), (got, want)


def test_single_markout_equals_join_asof(tmp_path: Path) -> None:
    c = Case(seed=20, markouts=(5,))
    left, right, lp, rp = write(c, tmp_path)
    ctx = make_ctx()
    m = run(c, ctx, read_sorted(ctx, lp, c), read_sorted(ctx, rp, c)).to_polars()
    shifted = ctx.read_parquet(lp, file_sort_order=[[col("g").sort(nulls_first=True), col("ts").sort(nulls_first=True)]])
    shifted = shifted.with_column("ts", col("ts") + lit(timedelta(seconds=5)))
    a = dp.join_asof(shifted, read_sorted(ctx, rp, c), on="ts", by="g", ctx=ctx).to_polars()
    assert m["rid_5s"].to_list() == a["rid"].to_list()
    assert m["px_5s"].to_list() == a["px_right"].to_list()


def test_headline_pipeline(tmp_path: Path) -> None:
    c = Case(seed=21, n_left=3000, n_right=8000, markouts=(1, 10, 60))
    left, right, lp, rp = write(c, tmp_path)
    ctx = make_ctx()
    order = [[col("g").sort(), col("ts").sort()]]
    trades = ctx.read_parquet(lp, file_sort_order=order).filter(col("lpx") > lit(100.0))
    quotes = ctx.read_parquet(rp, file_sort_order=order)
    got = dp.join_asof_markouts(trades, quotes, on="ts", markouts=[1, 10, 60], by="g", ctx=ctx).to_polars()
    plan = dp.last_plan()
    assert "MarkoutsExec" in plan and "OrderedConcatExec" in plan and "SortExec" not in plan, plan
    assert "predicate=lpx@" in plan and "pruning_predicate" in plan, plan
    want = polars_markouts(c, left.filter(pl.col("lpx") > 100.0), right)
    assert got.equals(want)


def test_errors(tmp_path: Path) -> None:
    c = Case(seed=22)
    left, right, lp, rp = write(c, tmp_path)
    ctx = make_ctx()
    lf, rf = read_sorted(ctx, lp, c), read_sorted(ctx, rp, c)
    with pytest.raises(ValueError, match="must be a Timestamp"):
        dp.join_asof_markouts(lf, rf, on="id", markouts=[1], ctx=ctx)
    other = right.with_columns(pl.col("ts").cast(pl.Datetime("ns")))
    op = str(tmp_path / "other.parquet")
    other.write_parquet(op)
    with pytest.raises(ValueError, match="same type"):
        dp.join_asof_markouts(lf, ctx.read_parquet(op), on="ts", markouts=[1], by="g", ctx=ctx)
    with pytest.raises(ValueError, match="not a whole number"):
        dp.join_asof_markouts(lf, rf, on="ts", markouts=[0.0000001], by="g", ctx=ctx)
    with pytest.raises(ValueError, match="duplicate markout 1"):
        dp.join_asof_markouts(lf, rf, on="ts", markouts=[1, 1.0], by="g", ctx=ctx)
    clash = left.with_columns(pl.col("lpx").alias("rid_1s"))
    cp = str(tmp_path / "clash.parquet")
    clash.write_parquet(cp)
    with pytest.raises(ValueError, match="rid_1s"):
        dp.join_asof_markouts(ctx.read_parquet(cp), rf, on="ts", markouts=[1], by="g", ctx=ctx)
    with pytest.raises(TypeError):
        dp.join_asof_markouts(lf, rf, on="ts", markouts=["1"], by="g", ctx=ctx)
    with pytest.raises(ValueError, match="by_left"):
        dp.join_asof_markouts(lf, rf, on="ts", markouts=[1], by_left="g", ctx=ctx)
    joined = lf.alias("a").join(lf.select("id").alias("b"), left_on="id", right_on="id")
    with pytest.raises(Exception, match="scan of one Parquet file"):
        dp.join_asof_markouts(joined, rf, on="ts", markouts=[1], by="g", ctx=ctx).collect()
