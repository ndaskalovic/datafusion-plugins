"""Differential tests: datafusion_plugins vs polars ``join_asof`` and a
brute-force reference, on random Parquet files with small row groups (so the
scans split into many byte ranges and batches are small)."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from datafusion import SessionContext, col, lit
from datafusion import functions as F

import datafusion_plugins as dp



@dataclasses.dataclass(frozen=True)
class Case:
    seed: int
    by: str | None = "i32"  # None, "i32", "i64"
    ts: str = "i64"  # "i64", "ts_ns"
    n_left: int = 700
    n_right: int = 2500
    left_groups: int = 12
    right_groups: int = 9  # < left_groups: groups with no right rows
    ts_range: int = 400  # small => many ties
    null_frac: float = 0.0
    strict: bool = False  # allow_exact_matches=False
    row_group: int = 37


def gen(c: Case, n: int, groups: int, seed: int, id_base: int, payload: str) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    ts = rng.integers(0, c.ts_range, n)
    cols: dict[str, pl.Series] = {"id": pl.Series("id", np.arange(n) + id_base)}
    if c.by is not None:
        g = rng.integers(0, groups, n)
        dtype = pl.Int32 if c.by == "i32" else pl.Int64
        gs = pl.Series("g", g).cast(dtype)
        if c.null_frac:
            gs = gs.scatter(np.flatnonzero(rng.random(n) < c.null_frac), None)
        cols["g"] = gs
    tss = pl.Series("ts", ts, dtype=pl.Int64)
    if c.null_frac:
        tss = tss.scatter(np.flatnonzero(rng.random(n) < c.null_frac), None)
    if c.ts == "ts_ns":
        tss = tss.cast(pl.Datetime("ns"))
    cols["ts"] = tss
    cols[payload] = pl.Series(payload, (np.arange(n) + id_base).astype(np.float64))
    df = pl.DataFrame(list(cols.values()))
    keys = ["g", "ts"] if c.by is not None else ["ts"]
    # Stable, nulls first (the declared file order).
    return df.sort(keys, nulls_last=False, maintain_order=True)


def write(c: Case, tmp: Path) -> tuple[pl.DataFrame, pl.DataFrame, str, str]:
    left = gen(c, c.n_left, c.left_groups, c.seed, 0, "lpx")
    right = gen(c, c.n_right, c.right_groups, c.seed + 1000, 1_000_000, "rid")
    lp, rp = str(tmp / "left.parquet"), str(tmp / "right.parquet")
    left.write_parquet(lp, row_group_size=c.row_group)
    right.write_parquet(rp, row_group_size=c.row_group)
    return left, right, lp, rp


def by_list(c: Case) -> list[str]:
    return [] if c.by is None else ["g"]


def reference(c: Case, left: pl.DataFrame, right: pl.DataFrame) -> list[float | None]:
    """Brute force: the last right row (file order) of the same group with the
    greatest ts <= (< when strict) left ts. NULL keys never match."""
    def as_int(s: pl.Series) -> np.ndarray:
        return s.cast(pl.Int64).to_numpy() if s.dtype != pl.Int64 else s.to_numpy()

    r = right.with_row_index("pos")
    keys = ["ts", *by_list(c)]
    r = r.drop_nulls(keys)
    groups: dict[object, tuple[np.ndarray, np.ndarray]] = {}
    gcol = r["g"].to_list() if c.by else [None] * r.height
    rts = as_int(r["ts"])
    rid = r["rid"].to_numpy()
    for k in set(gcol):
        m = np.array([x == k for x in gcol], dtype=bool)
        groups[k] = (rts[m], rid[m])  # file order = ts order within a group
    out: list[float | None] = []
    lg = left["g"].to_list() if c.by else [None] * left.height
    lts = left["ts"].cast(pl.Int64).to_list()
    for g, t in zip(lg, lts):
        if t is None or (c.by and g is None) or g not in groups:
            out.append(None)
            continue
        ts_arr, ids = groups[g]
        i = np.searchsorted(ts_arr, t, side="left" if c.strict else "right") - 1
        out.append(float(ids[i]) if i >= 0 else None)
    return out


def polars_ref(c: Case, left: pl.DataFrame, right: pl.DataFrame) -> pl.DataFrame:
    by = by_list(c)
    return left.join_asof(
        right.select(["ts", *by, "rid"]),
        on="ts",
        by=by or None,
        strategy="backward",
        allow_exact_matches=not c.strict,
    )


def check_result(c: Case, left: pl.DataFrame, right: pl.DataFrame, got: pl.DataFrame) -> None:
    # Row order equals the left input order.
    assert got["id"].to_list() == left["id"].to_list()
    assert got["rid"].to_list() == reference(c, left, right)
    # polars agrees on NULLs too: NULL ts or by never matches.
    want = polars_ref(c, left, right).select(got.columns)
    assert got.equals(want), (c, got, want)


CASES = [
    # Grouped and ungrouped, Int32/Int64 by, Int64/timestamp on.
    *[Case(seed=s) for s in (1, 2)],
    Case(seed=3, by="i64"),
    Case(seed=6, by=None),
    Case(seed=7, ts="ts_ns"),
    Case(seed=8, ts="ts_ns", by=None),
    # Ties, allow_exact_matches False / True.
    Case(seed=9, ts_range=30),
    Case(seed=10, ts_range=30, strict=True),
    Case(seed=12, by=None, ts_range=20, strict=True),
    # NULL keys.
    Case(seed=13, null_frac=0.03),
    Case(seed=14, null_frac=0.03, strict=True),
    Case(seed=15, null_frac=0.03, by=None, strict=True),
    # Groups with no right rows; sparse left; tiny right.
    Case(seed=16, left_groups=20, right_groups=5),
    Case(seed=17, n_left=15, n_right=4000),
    Case(seed=18, n_left=15, n_right=4000, by=None),
    Case(seed=19, n_left=1500, n_right=40, right_groups=3),
]


def case_id(c: Case) -> str:
    return (
        f"s{c.seed}-{c.by}-{c.ts}-L{c.n_left}-R{c.n_right}"
        f"-tr{c.ts_range}-n{c.null_frac}-{'gt' if c.strict else 'ge'}"
    )


def read_sorted(ctx: SessionContext, path: str, c: Case, nulls_first: bool = True):
    keys = ["g", "ts"] if c.by is not None else ["ts"]
    order = [[col(k).sort(ascending=True, nulls_first=nulls_first) for k in keys]]
    return ctx.read_parquet(path, file_sort_order=order)


def run(c: Case, ctx: SessionContext, lf, rf, **kw):
    by = by_list(c) or None
    out = dp.join_asof(lf, rf, on="ts", by=by, allow_exact_matches=not c.strict, ctx=ctx, **kw)
    return out


# Declared order (nothing sorted), no order (DataFusion sorts: tied left
# rows may be reordered), or no order with assume_sorted (trusted, checked).
@pytest.mark.parametrize("mode", ["declared", "sorted", "assumed"])
@pytest.mark.parametrize("c", CASES, ids=case_id)
def test_matches_polars(c: Case, mode: str, tmp_path: Path, ctx) -> None:
    left, right, lp, rp = write(c, tmp_path)
    if mode == "declared":
        lf, rf = read_sorted(ctx, lp, c), read_sorted(ctx, rp, c)
    else:
        lf, rf = ctx.read_parquet(lp), ctx.read_parquet(rp)
    out = run(c, ctx, lf, rf, assume_sorted=mode == "assumed")
    got = out.select("id", "rid").to_polars()
    plan = dp.last_plan(out)
    assert ("SortExec" in plan) == (mode == "sorted"), plan
    if mode == "sorted":
        want = dict(zip(left["id"].to_list(), reference(c, left, right)))
        assert dict(zip(got["id"].to_list(), got["rid"].to_list())) == want
        assert got.height == left.height
    else:
        check_result(c, left, right, got)


def test_headline_pipeline(tmp_path: Path, ctx) -> None:
    """read_parquet with a declared order -> filter -> with_column ->
    join_asof: no sort, range-parallel mode, filter pushed into the scan, and
    the result chains on."""
    c = Case(seed=50, n_left=3000, n_right=8000)
    left, right, lp, rp = write(c, tmp_path)
    order = [[col("g").sort(), col("ts").sort()]]  # NULLS LAST is fine too
    trades = (
        ctx.read_parquet(lp, file_sort_order=order)
        .filter(col("lpx") > lit(100.0))
        .with_column("notional", col("lpx") * lit(2))
    )
    quotes = ctx.read_parquet(rp, file_sort_order=order)
    out = dp.join_asof(trades, quotes, on="ts", by="g", ctx=ctx)
    res = (
        out.filter(col("rid").is_not_null())
        .aggregate([col("g")], [F.count(col("rid")).alias("n")])
        .sort(col("g").sort())
        .to_polars()
    )
    plan = dp.last_plan(out)
    assert "SortExec" not in plan and "HostInputExec" not in plan, plan
    assert "predicate=lpx@" in plan and "pruning_predicate" in plan, plan
    lf = left.filter(pl.col("lpx") > 100.0).with_columns(notional=pl.col("lpx") * 2)
    want = polars_ref(c, lf, right)
    full = out.select("id", "notional", "rid").to_polars()
    assert full.equals(want.select("id", "notional", "rid"))
    want_agg = (
        want.filter(pl.col("rid").is_not_null())
        .group_by("g")
        .agg(pl.col("rid").count().alias("n"))
        .sort("g")
    )
    assert res.equals(want_agg.with_columns(pl.col("n").cast(res["n"].dtype)))


def test_input_is_a_join(tmp_path: Path, ctx) -> None:
    c = Case(seed=52)
    left, right, lp, rp = write(c, tmp_path)
    dim = pl.DataFrame({"g": pl.Series(range(12), dtype=pl.Int32), "w": [float(i) for i in range(12)]})
    dp_path = str(tmp_path / "dim.parquet")
    dim.write_parquet(dp_path)
    lf = read_sorted(ctx, lp, c).alias("t").join(
        ctx.read_parquet(dp_path).alias("d"), on="g", how="inner"
    )
    rf = read_sorted(ctx, rp, c)
    out = dp.join_asof(lf, rf, on="ts", by="g", ctx=ctx)
    got = out.select("id", "w", "rid").to_polars()
    plan = dp.last_plan(out)
    assert "HashJoinExec" in plan, plan
    lj = left.join(dim, on="g", how="inner")
    want = polars_ref(c, lj, right).select("id", "w", "rid")
    assert got.sort("id").equals(want.sort("id"))


def test_not_sorted_raises(tmp_path: Path, ctx) -> None:
    c = Case(seed=54)
    left, right, lp, rp = write(c, tmp_path)
    shuffled = str(tmp_path / "left_shuffled.parquet")
    left.sample(fraction=1.0, shuffle=True, seed=2).write_parquet(shuffled, row_group_size=37)
    out = dp.join_asof(ctx.read_parquet(shuffled), ctx.read_parquet(rp), on="ts", by="g", assume_sorted=True, ctx=ctx)
    with pytest.raises(Exception, match=r"left input is not sorted by \(g, ts\)"):
        out.collect()
    shuffled_r = str(tmp_path / "right_shuffled.parquet")
    right.sample(fraction=1.0, shuffle=True, seed=3).write_parquet(shuffled_r, row_group_size=37)
    by_ts = str(tmp_path / "left_by_ts.parquet")
    left.sort("ts", maintain_order=True).write_parquet(by_ts, row_group_size=37)
    out = dp.join_asof(ctx.read_parquet(by_ts), ctx.read_parquet(shuffled_r), on="ts", assume_sorted=True, ctx=ctx)
    with pytest.raises(Exception, match=r"right input is not sorted by \(ts\)"):
        out.collect()


def test_check_sortedness_false_trusts(tmp_path: Path, ctx) -> None:
    c = Case(seed=55)
    left, right, lp, rp = write(c, tmp_path)
    out = dp.join_asof(
        ctx.read_parquet(lp), ctx.read_parquet(rp), on="ts", by="g",
        assume_sorted=True, check_sortedness=False, ctx=ctx,
    )
    check_result(c, left, right, out.select("id", "rid").to_polars())


SCHEMA_ARGS = [
    {"on": "ts"},
    {"on": "ts", "by": "g"},
    {"on": "ts", "by": "g", "suffix": "_r"},
    {"left_on": "ts", "right_on": "rts"},
    {"left_on": "ts", "right_on": "rts", "by_left": "g", "by_right": "rg"},
    {"left_on": "ts", "right_on": "rts", "by_left": ["g"], "by_right": ["rg"], "suffix": "_x"},
]


@pytest.mark.parametrize("kw", SCHEMA_ARGS, ids=lambda k: "-".join(f"{a}={b}" for a, b in k.items()))
def test_schema_matches_polars(kw: dict, tmp_path: Path, ctx) -> None:
    rename = "right_on" in kw
    left = pl.DataFrame(
        {"id": [1, 2, 3], "g": [1, 1, 2], "ts": [5, 9, 4], "x": [1.0, 2.0, 3.0], "y_right": [0, 0, 0],
         "rts": [7, 7, 7]}
    ).with_columns(pl.col("g").cast(pl.Int32))
    right = pl.DataFrame(
        {"g": [1, 1, 2], "ts": [3, 8, 1], "x": [2.0, 4.0, 6.0], "y": [9, 8, 7], "id": [7, 8, 9]}
    ).with_columns(pl.col("g").cast(pl.Int32))
    if rename:
        right = right.rename({"ts": "rts", "g": "rg"})
    by = kw.get("by") or kw.get("by_left")
    keys = (["g"] if by else []) + ["ts"]
    rkeys = (["rg" if rename else "g"] if by else []) + ["rts" if rename else "ts"]
    left, right = left.sort(keys), right.sort(rkeys)
    lp, rp = str(tmp_path / "l.parquet"), str(tmp_path / "r.parquet")
    left.write_parquet(lp)
    right.write_parquet(rp)
    want = left.join_asof(right, strategy="backward", **kw)
    got = dp.join_asof(ctx.read_parquet(lp), ctx.read_parquet(rp), ctx=ctx, **kw).to_polars()
    assert got.columns == want.columns
    assert got.equals(want), (got, want)


def test_suffix_clash_raises(tmp_path: Path, ctx) -> None:
    left = pl.DataFrame({"ts": [1], "y": [0], "y_right": [3]})
    right = pl.DataFrame({"ts": [1], "y": [1]})
    with pytest.raises(Exception, match="already exists"):
        left.join_asof(right, on="ts")
    lp, rp = str(tmp_path / "l.parquet"), str(tmp_path / "r.parquet")
    left.write_parquet(lp)
    right.write_parquet(rp)
    with pytest.raises(ValueError, match="already exists"):
        dp.join_asof(ctx.read_parquet(lp), ctx.read_parquet(rp), on="ts", ctx=ctx)


def test_argument_validation(tmp_path: Path, ctx) -> None:
    c = Case(seed=56)
    _, _, lp, rp = write(c, tmp_path)
    lf, rf = read_sorted(ctx, lp, c), read_sorted(ctx, rp, c)
    with pytest.raises(NotImplementedError):
        dp.join_asof(lf, rf, on="ts", strategy="forward", ctx=ctx)
    with pytest.raises(ValueError, match="either 'on'"):
        dp.join_asof(lf, rf, ctx=ctx)
    with pytest.raises(ValueError, match="not both"):
        dp.join_asof(lf, rf, on="ts", left_on="ts", right_on="ts", ctx=ctx)
    with pytest.raises(ValueError, match="left_on"):
        dp.join_asof(lf, rf, left_on="ts", ctx=ctx)
    with pytest.raises(ValueError, match="not both"):
        dp.join_asof(lf, rf, on="ts", by="g", by_left="g", by_right="g", ctx=ctx)
    with pytest.raises(ValueError, match="by_left"):
        dp.join_asof(lf, rf, on="ts", by_left="g", ctx=ctx)
    with pytest.raises(ValueError, match="not found"):
        dp.join_asof(lf, rf, on="nope", ctx=ctx)
