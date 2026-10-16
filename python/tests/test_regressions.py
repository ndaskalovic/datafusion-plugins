"""Nested nodes, failing upstreams, NULLS LAST inputs, awkward column names,
empty inputs, more key types, re-running and limits; on both input routes."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest
from datafusion import col, udf

import datafusion_plugins as dp

ROUTES = ["plan", "stream"]


def frames(ctx, tmp: Path, route: str, **tables: pa.Table):
    """`tables` as DataFrames: parquet files (plan route) or in memory."""
    out = []
    for name, t in tables.items():
        if route == "plan":
            p = str(tmp / f"{name}.parquet")
            pq.write_table(t, p)
            out.append(ctx.read_parquet(p))
        else:
            out.append(ctx.from_arrow(t))
    return out


def q(name: str):
    """Column `name`, quoted (a bare string is parsed as SQL)."""
    return col('"' + name.replace('"', '""') + '"')


A = pa.table({"ts": pa.array([1, 5, 9], pa.int64()), "id": [1, 2, 3]})
B = pa.table({"ts": pa.array([0, 4, 8], pa.int64()), "v": [10, 20, 30]})
C = pa.table({"ts": pa.array([2, 6], pa.int64()), "w": [7, 8]})


@pytest.mark.parametrize("route", ROUTES)
def test_nested_nodes(ctx, tmp_path: Path, route: str) -> None:
    a, b, c = frames(ctx, tmp_path, route, a=A, b=B, c=C)
    inner = dp.join_asof(a, b, on="ts", ctx=ctx)
    outer = dp.join_asof(inner, c, on="ts", ctx=ctx)
    want = {"ts": [1, 5, 9], "id": [1, 2, 3], "v": [10, 20, 30], "w": [None, 7, 8]}
    assert outer.to_arrow_table().to_pydict() == want
    assert outer.count() == 3
    assert "HostInputExec" in dp.last_plan(outer)  # the inner node is streamed in
    # Markouts inside join_asof, and the other way round.
    ts = lambda df: df.with_column("ts", col("ts").cast(pa.timestamp("s")))  # noqa: E731
    mk = dp.join_asof_markouts(ts(a), ts(b), on="ts", markouts=[0, 2], ctx=ctx)
    got = dp.join_asof(mk, ts(c), on="ts", ctx=ctx).to_arrow_table().to_pydict()
    assert got["v_0s"] == [10, 20, 30] and got["v_2s"] == [10, 20, 30] and got["w"] == [None, 7, 8]
    got = dp.join_asof_markouts(dp.join_asof(ts(a), ts(c), on="ts", ctx=ctx), ts(b), on="ts", markouts=[-1], ctx=ctx)
    assert got.to_arrow_table().to_pydict()["v_-1s"] == [10, 20, 30]


def test_failing_upstream_raises(ctx) -> None:
    def boom(a: pa.Array) -> pa.Array:
        if pc.max(a).as_py() > 1500:
            raise RuntimeError("boom in the UDF")
        return a

    f = udf(boom, [pa.int64()], pa.int64(), "volatile", name="boom")
    n = 3000
    left = ctx.from_arrow(pa.table({"ts": pa.array(range(n), pa.int64()), "id": range(n)})).repartition(3)
    right = ctx.from_arrow(pa.table({"ts": pa.array(range(0, n, 7), pa.int64()), "v": range(0, n, 7)}))
    out = dp.join_asof(left.with_column("ts", f(col("ts"))), right, on="ts", ctx=ctx)
    with pytest.raises(Exception, match="boom in the UDF"):
        out.collect()


@pytest.mark.parametrize("parts", [1, 2])
@pytest.mark.parametrize("nulls_first", [True, False])
def test_nulls_first_or_last(ctx, parts: int, nulls_first: bool) -> None:
    ts = pa.array([5, None, 1, 9, None, 3], pa.timestamp("s"))
    left = pa.record_batch({"ts": ts, "id": [1, 2, 3, 4, 5, 6]})
    right = pa.record_batch({"ts": pa.array([0, 4, None, 8], pa.timestamp("s")), "v": [10, 20, 0, 30]})
    ctx.register_record_batches("l", [[left]])
    ctx.register_record_batches("r", [[right]])
    order = "NULLS FIRST" if nulls_first else "NULLS LAST"
    lf = ctx.sql(f"select * from l order by ts {order}")
    rf = ctx.sql(f"select * from r order by ts {order}")
    if parts == 2:
        # Two partitions, each sorted the same way.
        t = lf.to_arrow_table()
        halves = [t.filter(pc.equal(pc.bit_wise_and(t["id"], 1), k)).to_batches() for k in (0, 1)]
        lf = ctx.create_dataframe(halves)
    want = {1: 20, 2: None, 3: 10, 4: 30, 5: None, 6: 10}
    for node in ("asof", "markouts"):
        for check in (True, False):
            if node == "asof":
                out = dp.join_asof(lf, rf, on="ts", assume_sorted=True, check_sortedness=check, ctx=ctx)
                v = "v"
            else:
                out = dp.join_asof_markouts(lf, rf, on="ts", markouts=[0], check_sortedness=check, ctx=ctx)
                v = "v_0s"
            got = out.to_arrow_table().to_pydict()
            assert dict(zip(got["id"], got[v])) == want, got
            matched = [t for t, x in zip(got["ts"], got[v]) if t is not None]
            assert matched == sorted(matched), got
            if parts == 1:
                assert got["id"] == lf.to_arrow_table().to_pydict()["id"]
    # Letting the node sort works whatever the placement.
    got = dp.join_asof(lf, rf, on="ts", ctx=ctx).to_arrow_table().to_pydict()
    assert dict(zip(got["id"], got["v"])) == want


@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize("name", ["Price", "a.b", "my col", 'q"x'])
def test_awkward_column_names(ctx, tmp_path: Path, route: str, name: str) -> None:
    left = pa.table({name: pa.array([1, 5], pa.int64()), "g": [1, 1], "id": [1, 2], "junk": [0, 0]})
    right = pa.table({name: pa.array([0, 4], pa.int64()), "g": [1, 1], "x" + name: [10, 20], "junk2": [0, 0]})
    lf, rf = frames(ctx, tmp_path, route, l=left, r=right)
    out = dp.join_asof(lf, rf, on=name, by="g", ctx=ctx)
    assert out.to_arrow_table().to_pydict()["x" + name] == [10, 20]
    pruned = out.select(q("x" + name), col("id"))
    assert pruned.to_arrow_table().to_pydict() == {"x" + name: [10, 20], "id": [1, 2]}
    ts = lambda df: df.with_column(name, q(name).cast(pa.timestamp("s")))  # noqa: E731
    mk = dp.join_asof_markouts(ts(lf), ts(rf), on=name, markouts=[1], by="g", ctx=ctx)
    assert mk.select(q(f"x{name}_1s"), col("id")).to_arrow_table().to_pydict() == {f"x{name}_1s": [10, 20], "id": [1, 2]}


@pytest.mark.parametrize("route", ROUTES)
def test_empty_inputs(ctx, tmp_path: Path, route: str) -> None:
    empty = A.slice(0, 0)
    a, b, e = frames(ctx, tmp_path, route, a=A, b=B, e=empty)
    eb = frames(ctx, tmp_path, route, eb=B.slice(0, 0))[0]
    assert dp.join_asof(a, eb, on="ts", ctx=ctx).to_arrow_table().to_pydict()["v"] == [None] * 3
    assert dp.join_asof(e, b, on="ts", ctx=ctx).count() == 0
    assert dp.join_asof(e, eb, on="ts", assume_sorted=True, ctx=ctx).count() == 0


@pytest.mark.parametrize("route", ROUTES)
def test_more_key_types(ctx, tmp_path: Path, route: str) -> None:
    ons = [
        (pa.float64(), [-1.5, 0.0, 2.5], [-2.0, -0.0, 2.0]),
        (pa.uint64(), [1, 2**64 - 6, 2**64 - 1], [0, 7, 2**64 - 2]),
        (pa.date32(), [-10, 0, 20], [-11, -1, 19]),
    ]
    arr = lambda v, t: pa.array(v, pa.int32()).cast(t) if t == pa.date32() else pa.array(v, t)  # noqa: E731
    for t, lv, rv in ons:
        left = pa.table({"ts": arr(lv, t), "id": [1, 2, 3]})
        right = pa.table({"ts": arr(rv, t), "v": [10, 20, 30]})
        lf, rf = frames(ctx, tmp_path, route, l=left, r=right)
        got = dp.join_asof(lf, rf, on="ts", ctx=ctx).to_polars()
        want = pl.from_arrow(left).join_asof(pl.from_arrow(right), on="ts")
        assert got.equals(want), (t, got, want)
    bys = [
        (pa.array([3, 2**64 - 1, 2**64 - 1], pa.uint64()), pa.array([3, 3, 2**64 - 1], pa.uint64())),
        (pa.array(["a", "b", "b"]).dictionary_encode(), pa.array(["a", "a", "b"]).dictionary_encode()),
    ]
    for lg, rg in bys:
        left = pa.table({"g": lg, "ts": pa.array([1, 2, 3], pa.int64()), "id": [1, 2, 3]})
        right = pa.table({"g": rg, "ts": pa.array([1, 2, 3], pa.int64()), "v": [10, 20, 30]})
        lf, rf = frames(ctx, tmp_path, route, l=left, r=right)
        got = dp.join_asof(lf, rf, on="ts", by="g", ctx=ctx).to_arrow_table().to_pydict()
        assert got["v"] == [10, None, 30], (lg.type, got)


@pytest.mark.parametrize("route", ROUTES)
def test_rerun_and_limit(ctx, tmp_path: Path, route: str) -> None:
    n = 5000
    left = pa.table({"ts": pa.array(range(0, 3 * n, 3), pa.int64()), "id": range(n)})
    right = pa.table({"ts": pa.array(range(0, 3 * n, 7), pa.int64()), "v": range(len(range(0, 3 * n, 7)))})
    lf, rf = frames(ctx, tmp_path, route, l=left, r=right)
    out = dp.join_asof(lf, rf, on="ts", ctx=ctx, _route=route)
    first = out.to_arrow_table()
    assert first.num_rows == n
    assert out.to_arrow_table().equals(first)  # planned and executed again
    assert out.limit(3).to_arrow_table().equals(first.slice(0, 3))
    assert out.limit(2, offset=n - 2).to_arrow_table().equals(first.slice(n - 2))


def test_errors_are_not_swallowed_by_route_fallback(ctx, tmp_path: Path) -> None:
    for route in ROUTES:
        a, b = frames(ctx, tmp_path, route, a=A, b=B)
        with pytest.raises(ValueError, match="column 'nope' not found"):
            dp.join_asof(a, b, on="nope", ctx=ctx)
    a, b = frames(ctx, tmp_path, "stream", a=A, b=B)
    with pytest.raises(dp._internal.PlanRouteUnavailable):
        dp.join_asof(a, b, on="ts", ctx=ctx, _route="plan")


def test_memory_limit_set_after_creating_the_node(ctx) -> None:
    out = dp.join_asof(ctx.from_arrow(A), ctx.from_arrow(B), on="ts", ctx=ctx)
    try:
        dp.set_memory_limit(1, "greedy")  # applies when the query runs
        with pytest.raises(Exception, match="(?i)memory"):
            out.collect()
    finally:
        dp.set_memory_limit(None)
    assert out.count() == 3
