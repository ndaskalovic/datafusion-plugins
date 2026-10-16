"""Custom DataFusion operators for datafusion-python 55.

* :func:`join_asof`: streaming ASOF join with polars semantics.
* :func:`join_asof_markouts`: ASOF joins at several time offsets in one pass.

Each takes :class:`datafusion.DataFrame` inputs and returns a
:class:`datafusion.DataFrame`, so you keep chaining ``.filter(...)``,
``.aggregate(...)``, ``.to_polars()`` and so on. :func:`join_asof`'s
arguments, output columns and contract follow polars ``DataFrame.join_asof``:

* backward search: for every left row, the last right row with equal ``by``
  values and ``right.on <= left.on`` (``<`` with
  ``allow_exact_matches=False``); every left row appears once, in left order;
  right columns are null without a match;
* inputs are sorted by ``(by..., on)`` (or ``on`` without ``by``): either
  required (DataFusion sorts only what it can't prove sorted) or, with
  ``assume_sorted``, trusted and checked while streaming.

The operator is a native DataFusion ``ExecutionPlan`` loaded through
datafusion-python's FFI layer.
"""

from __future__ import annotations

import math
import weakref
from collections.abc import Sequence
from decimal import Decimal
from typing import Any, Literal

from datafusion import DataFrame, SessionContext

from ._internal import NodeTable, PlanRouteUnavailable, __version__, set_memory_limit

__all__ = ["__version__", "join_asof", "join_asof_markouts", "last_plan", "set_memory_limit"]

Route = Literal["auto", "plan", "stream"]

# Each node result's table, and the latest one, for `last_plan`.
_tables: weakref.WeakKeyDictionary[DataFrame, NodeTable] = weakref.WeakKeyDictionary()
_latest: list[NodeTable] = []


def _names(arg: str | Sequence[str] | None) -> list[str] | None:
    if arg is None:
        return None
    if isinstance(arg, str):
        return [arg]
    return list(arg)


def _by_args(
    by: str | Sequence[str] | None,
    by_left: str | Sequence[str] | None,
    by_right: str | Sequence[str] | None,
) -> tuple[list[str], list[str]]:
    """Resolve ``by`` / ``by_left`` + ``by_right`` as polars does."""
    by_l, by_r = _names(by_left), _names(by_right)
    if by is not None:
        if by_l is not None or by_r is not None:
            msg = "give either 'by' or both 'by_left' and 'by_right', not both"
            raise ValueError(msg)
        by_l = by_r = _names(by)
    elif (by_l is None) != (by_r is None):
        msg = "expected both 'by_left' and 'by_right' to be set"
        raise ValueError(msg)
    by_l, by_r = by_l or [], by_r or []
    if len(by_l) != len(by_r):
        msg = "'by_left' and 'by_right' must have the same length"
        raise ValueError(msg)
    return by_l, by_r


def _node(
    name: str, inputs: Sequence[DataFrame], args: dict[str, Any], ctx: SessionContext | None, route: Route
) -> DataFrame:
    """Run node ``name`` over ``inputs`` and return the result as a DataFrame.

    Inputs whose plans serialize and read only local files are planned by
    the plugin itself (its own parallel scans, pushdown and orderings);
    anything else (in-memory tables, Python UDFs, object stores, ...) is
    executed by datafusion-python and streamed to the plugin, partition by
    partition. Both routes give the same results; ``route`` forces one.
    """
    ctx = ctx if ctx is not None else SessionContext.global_ctx()
    table = None
    if route != "stream":
        try:
            plans = [df.logical_plan().to_bytes(ctx) for df in inputs]
        except Exception as e:  # not serializable (in-memory tables, Python UDFs, ...)
            if route == "plan":
                raise PlanRouteUnavailable(str(e)) from e
            plans = None
        if plans is not None:
            try:
                table = NodeTable.from_plans(name, plans, args)
            except PlanRouteUnavailable:
                if route == "plan":
                    raise
    if table is None:
        table = NodeTable.from_frames(name, list(inputs), args)
    out = ctx.read_table(table)
    _tables[out] = table
    _latest[:] = [table]
    return out


def join_asof(
    left: DataFrame,
    right: DataFrame,
    on: str | None = None,
    *,
    left_on: str | None = None,
    right_on: str | None = None,
    by: str | Sequence[str] | None = None,
    by_left: str | Sequence[str] | None = None,
    by_right: str | Sequence[str] | None = None,
    strategy: Literal["backward", "forward", "nearest"] = "backward",
    allow_exact_matches: bool = True,
    suffix: str = "_right",
    assume_sorted: bool = False,
    check_sortedness: bool = True,
    ctx: SessionContext | None = None,
    _route: Route = "auto",
) -> DataFrame:
    """ASOF join of two DataFrames, like polars ``DataFrame.join_asof``.

    For each row of ``left``, take the last row of ``right`` with equal
    ``by`` columns and ``right[on] <= left[on]``.

    Args:
        left: Left input. Its rows are kept, once each, in order.
        right: Right input.
        on: The join column of both inputs (any integer, float, date, time,
            timestamp or duration type; the same type on both sides). Use
            either ``on`` or both ``left_on`` and ``right_on``.
        left_on: The left join column.
        right_on: The right join column.
        by: Equality column(s), the same names on both sides. Use either
            ``by`` or both ``by_left`` and ``by_right``.
        by_left: Left equality column(s).
        by_right: Right equality column(s).
        strategy: Only ``"backward"`` is supported.
        allow_exact_matches: If False, require ``right[on] < left[on]``.
        suffix: Appended to right column names that clash with left ones.
        assume_sorted: Trust that each input partition arrives sorted by
            ``(by..., on)`` (or ``on``) instead of requiring it. With False,
            the node requires that order and DataFusion sorts only where it
            can't prove it (a declared ``file_sort_order`` or your own
            ``.sort()`` cost nothing). With True nothing is sorted and the
            order is checked while streaming.
        check_sortedness: With ``assume_sorted``, also check the order inside
            every batch (vectorised), not only at batch and partition
            boundaries. An unsorted input raises ``left/right input is not
            sorted by (...)``.
        ctx: The SessionContext the inputs belong to; the result is a
            DataFrame of this context, and its configuration
            (``target_partitions``, ...) is used to plan the inputs. Defaults
            to the global context.

    Returns:
        polars' columns: every left column, then the right columns except
        the ``by`` keys and the ``on`` column (kept when ``right_on`` differs
        from ``left_on``). Right names already present get ``suffix``.

    Raises:
        NotImplementedError: ``strategy`` is not ``"backward"``.
        ValueError: Invalid argument combinations, unknown columns or types.

    Any input works: files, in-memory tables, Python UDFs, joins,
    aggregates, unions. Inputs that read local files are planned by the
    plugin; the others are executed by datafusion-python and streamed in.
    """
    if strategy != "backward":
        msg = f"join_asof: strategy={strategy!r} is not supported, only 'backward'"
        raise NotImplementedError(msg)
    if on is not None:
        if left_on is not None or right_on is not None:
            msg = "join_asof: give either 'on' or both 'left_on' and 'right_on', not both"
            raise ValueError(msg)
        left_on = right_on = on
    elif left_on is None or right_on is None:
        msg = "join_asof: give either 'on' or both 'left_on' and 'right_on'"
        raise ValueError(msg)
    try:
        by_l, by_r = _by_args(by, by_left, by_right)
    except ValueError as e:
        raise ValueError(f"join_asof: {e}") from None
    args = {
        "left_on": left_on,
        "right_on": right_on,
        "by_left": by_l,
        "by_right": by_r,
        "allow_exact_matches": allow_exact_matches,
        "suffix": suffix,
        "assume_sorted": assume_sorted,
        "check_sortedness": check_sortedness,
    }
    return _node("join_asof", [left, right], args, ctx, _route)


def _markout_str(h: object) -> str:
    """An int or float markout as an exact plain decimal string."""
    if isinstance(h, bool) or not isinstance(h, (int, float)):
        msg = f"join_asof_markouts: markouts must be ints or floats, got {h!r}"
        raise TypeError(msg)
    if isinstance(h, float) and not math.isfinite(h):
        msg = f"join_asof_markouts: markouts must be finite, got {h!r}"
        raise ValueError(msg)
    d = Decimal(h) if isinstance(h, int) else Decimal(repr(h))
    return format(d, "f")


def join_asof_markouts(
    left: DataFrame,
    right: DataFrame,
    on: str,
    markouts: Sequence[int | float],
    *,
    by: str | Sequence[str] | None = None,
    by_left: str | Sequence[str] | None = None,
    by_right: str | Sequence[str] | None = None,
    allow_exact_matches: bool = True,
    assume_sorted: bool = True,
    check_sortedness: bool = True,
    ctx: SessionContext | None = None,
    _route: Route = "auto",
) -> DataFrame:
    """Backward ASOF joins at several time offsets, in one pass.

    For each left row and each markout ``h`` (seconds), the last right row of
    the same ``by`` group with ``right[on] <= left[on] + h`` (``<`` with
    ``allow_exact_matches=False``): what polars gives for
    ``left.with_columns(pl.col(on) + timedelta(seconds=h)).join_asof(right,
    on=on, by=by, strategy="backward")``, for every ``h`` side by side.

    Args:
        left: Left input. Its rows are kept, once each, in order.
        right: Right input.
        on: A Timestamp column (any unit, with or without a time zone) of the
            same type in both inputs.
        markouts: Offsets in seconds (ints or floats; negative, zero and
            fractional allowed; no duplicates). Each must be a whole number
            of the column's unit.
        by: Equality column(s), the same names on both sides, or
            ``by_left`` and ``by_right``.
        allow_exact_matches: If False, require ``right[on] < left[on] + h``.
        assume_sorted: Trust that each input partition arrives sorted by
            ``(by..., on)`` (or ``on``) instead of requiring it. With False,
            the node requires that order and DataFusion sorts only where it
            can't prove it (a declared ``file_sort_order`` or your own
            ``.sort()`` cost nothing). With True nothing is sorted and the
            order is checked while streaming.
        check_sortedness: With ``assume_sorted``, also check the order inside
            every batch (vectorised), not only at batch and partition
            boundaries. An unsorted input raises ``left/right input is not
            sorted by (...)``.
            Unlike ``join_asof``, ``assume_sorted`` defaults to True: inputs
            are expected sorted, as the node's typical use (trades vs quotes)
            has them.
        ctx: The SessionContext the inputs belong to (default: global).

    Returns:
        All left columns, then for each markout in the given order every
        right column except ``on`` and ``by``, suffixed ``_{h}s`` (``_1s``,
        ``_0.5s``, ``_-5s``).

    Any input works, as for :func:`join_asof`.
    """
    try:
        by_l, by_r = _by_args(by, by_left, by_right)
    except ValueError as e:
        raise ValueError(f"join_asof_markouts: {e}") from None
    args = {
        "left_on": on,
        "right_on": on,
        "markouts": [_markout_str(h) for h in markouts],
        "by_left": by_l,
        "by_right": by_r,
        "allow_exact_matches": allow_exact_matches,
        "assume_sorted": assume_sorted,
        "check_sortedness": check_sortedness,
    }
    return _node("join_asof", [left, right], args, ctx, _route)


def last_plan(df: DataFrame | None = None) -> str | None:
    """The physical plan of node result ``df`` (default: of the latest node).

    Planning happens when a result is executed (``collect``, ``to_polars``,
    ...) or explained; ``None`` until then. Use it to check whether a sort
    was added, which input route ran (``HostInputExec`` = streamed from
    datafusion-python), and that filters reached the scans.
    """
    table = _tables.get(df) if df is not None else (_latest[0] if _latest else None)
    return table.last_plan() if table is not None else None
