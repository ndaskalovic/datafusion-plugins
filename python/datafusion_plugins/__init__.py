"""Fast streaming ``join_asof`` for datafusion-python 55, with polars semantics.

:func:`join_asof` takes two :class:`datafusion.DataFrame` inputs and returns a
:class:`datafusion.DataFrame`, so you keep chaining ``.filter(...)``,
``.aggregate(...)``, ``.to_polars()`` and so on. Its arguments, output
columns and contract follow polars ``DataFrame.join_asof``:

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

from collections.abc import Sequence
from typing import Literal

from datafusion import DataFrame, SessionContext

from ._internal import AsofJoinTable, __version__, ffi_version
from ._internal import last_plan as _last_plan

__all__ = ["__version__", "join_asof", "last_plan"]


def _names(arg: str | Sequence[str] | None) -> list[str] | None:
    if arg is None:
        return None
    if isinstance(arg, str):
        return [arg]
    return list(arg)


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
    by_l, by_r = _names(by_left), _names(by_right)
    if by is not None:
        if by_l is not None or by_r is not None:
            msg = "join_asof: give either 'by' or both 'by_left' and 'by_right', not both"
            raise ValueError(msg)
        by_l = by_r = _names(by)
    elif (by_l is None) != (by_r is None):
        msg = "join_asof: expected both 'by_left' and 'by_right' to be set"
        raise ValueError(msg)
    by_l, by_r = by_l or [], by_r or []
    if len(by_l) != len(by_r):
        msg = "join_asof: 'by_left' and 'by_right' must have the same length"
        raise ValueError(msg)

    ctx = ctx if ctx is not None else SessionContext.global_ctx()
    try:
        lb = left.logical_plan().to_bytes(ctx)
        rb = right.logical_plan().to_bytes(ctx)
    except Exception as e:
        msg = (
            "join_asof: an input DataFrame cannot be serialized "
            f"({e}). On datafusion 55 inputs must be file-sourced; "
            "in-memory tables and Python UDFs are not supported."
        )
        raise ValueError(msg) from e
    table = AsofJoinTable(
        lb,
        rb,
        left_on,
        right_on,
        by_l,
        by_r,
        allow_exact_matches=allow_exact_matches,
        suffix=suffix,
        check_sortedness=check_sortedness,
    )
    return ctx.read_table(table)


def last_plan() -> str | None:
    """The physical plan of the most recent join planned in this process.

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
