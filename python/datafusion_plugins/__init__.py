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
* inputs must already be sorted by ``(by..., on)`` (or ``on`` without
  ``by``). Nothing is sorted for you: sort it yourself, as with polars.

The operator is a native DataFusion ``ExecutionPlan`` loaded through
datafusion-python's FFI layer.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from decimal import Decimal
from typing import Any, Literal

from datafusion import DataFrame, SessionContext

from ._internal import PlanTable, __version__, ffi_version
from ._internal import last_plan as _last_plan

__all__ = ["__version__", "join_asof", "join_asof_markouts", "last_plan"]


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
    name: str, inputs: Sequence[DataFrame], args: dict[str, Any], ctx: SessionContext | None
) -> DataFrame:
    """Run node ``name`` over ``inputs``: their logical plans go to the plugin,
    which plans them and returns an FFI table provider."""
    ctx = ctx if ctx is not None else SessionContext.global_ctx()
    try:
        plans = [df.logical_plan().to_bytes(ctx) for df in inputs]
    except Exception as e:
        msg = (
            f"{name}: an input DataFrame cannot be serialized ({e}). On datafusion "
            "55 inputs must be file-sourced; in-memory tables and Python UDFs are "
            "not supported."
        )
        raise ValueError(msg) from e
    return ctx.read_table(PlanTable(name, plans, args))


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
    check_sortedness: bool = True,
    ctx: SessionContext | None = None,
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
        check_sortedness: Validate the order of an input whose order
            DataFusion cannot prove, and raise ``left/right input is not
            sorted by (...)`` if it is not sorted. Inputs with a proven order
            (``read_parquet(..., file_sort_order=...)``, then filters and
            projections) are only checked at batch and partition boundaries.
            False trusts the data.
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
        ValueError: Invalid argument combinations, unknown columns or types,
            or an input that cannot be serialized (see Notes).

    Notes:
        The inputs travel as serialized logical plans (datafusion-proto) and
        are re-planned by the plugin, so they must be file-sourced (Parquet,
        CSV, ...) under standard operators: filters, projections, joins,
        aggregates. In-memory DataFrames (``from_pydict``, ``from_polars``,
        ...) and Python UDFs cannot be serialized on DataFusion 55. Files are
        read from the local filesystem.
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
        "check_sortedness": check_sortedness,
    }
    return _node("join_asof", [left, right], args, ctx)


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
    ctx: SessionContext | None = None,
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
        by: Equality column, the same name on both sides (one integer-like
            column), or ``by_left`` and ``by_right``.
        allow_exact_matches: If False, require ``right[on] < left[on] + h``.
        ctx: The SessionContext the inputs belong to (default: global).

    Returns:
        All left columns, then for each markout in the given order every
        right column except ``on`` and ``by``, suffixed ``_{h}s`` (``_1s``,
        ``_0.5s``, ``_-5s``).

    Inputs are assumed sorted by ``(by..., on)`` (or ``on``) and are not
    validated beyond batch and partition boundaries. They must be scans of
    one Parquet file each, optionally filtered or projected.
    """
    try:
        by_l, by_r = _by_args(by, by_left, by_right)
    except ValueError as e:
        raise ValueError(f"join_asof_markouts: {e}") from None
    args = {
        "on": on,
        "markouts": [_markout_str(h) for h in markouts],
        "by_left": by_l,
        "by_right": by_r,
        "allow_exact_matches": allow_exact_matches,
    }
    return _node("join_asof_markouts", [left, right], args, ctx)


def last_plan() -> str | None:
    """The physical plan most recently built by a node in this process.

    Planning happens when a result is executed (``collect``, ``to_polars``,
    ...) or explained. Use it to check which mode ran (range-parallel, or
    ``execute_once``) and that the filters reached the scans.
    """
    return _last_plan()


def _check_version() -> None:
    import datafusion

    major = int(datafusion.__version__.split(".")[0])
    if major != ffi_version():
        msg = (
            f"datafusion-plugins {__version__} was built for datafusion "
            f"{ffi_version()}.*, but datafusion {datafusion.__version__} is installed"
        )
        raise ImportError(msg)


_check_version()
