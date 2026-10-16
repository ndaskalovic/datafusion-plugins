# datafusion-plugins

A fast, streaming `join_asof` for [Apache DataFusion](https://datafusion.apache.org)
55, with the arguments, output and contract of polars
[`DataFrame.join_asof`](https://docs.pola.rs/api/python/stable/reference/dataframe/api/polars.DataFrame.join_asof.html).
It is a native `ExecutionPlan`, usable from Rust and from Python through
datafusion-python's FFI layer. No fork of DataFusion is needed.

It is faster than polars on large joins and uses 6–9x less memory. Memory is
O(groups + batch size), not O(right side), and there is no broadcast and no
sort.

```python
from datafusion import SessionContext, col, functions as F
import datafusion_plugins as dp

ctx = SessionContext()
order = [[col("security_id").sort(), col("ts").sort()]]
trades = (
    ctx.read_parquet("trades.parquet", file_sort_order=order)
    .filter(col("qty") > 10)
    .with_column("notional", col("qty") * col("price"))
)
quotes = ctx.read_parquet("quotes.parquet", file_sort_order=order)

out = dp.join_asof(trades, quotes, on="ts", by="security_id", ctx=ctx)
summary = (
    out.filter(col("bid").is_not_null())
    .aggregate([col("security_id")], [F.avg(col("ask") - col("bid")).alias("spread")])
    .to_polars()
)
```

## Install

From source; needs a Rust toolchain (1.94 or newer):

```sh
pip install git+https://github.com/ndaskalovic/datafusion-plugins
# or, in a checkout:
uv venv .venv && uv pip install --python .venv/bin/python "datafusion==55.*" maturin
source .venv/bin/activate && maturin develop --release
```

The package depends on `datafusion==55.*`. The DataFusion FFI ABI is tied to
DataFusion's major version, so this build loads only into datafusion-python
55.x, and fails at import with a clear message on any other version.

## Python API

```python
datafusion_plugins.join_asof(
    left, right, on=None, *, left_on=None, right_on=None,
    by=None, by_left=None, by_right=None,
    strategy="backward", allow_exact_matches=True, suffix="_right",
    check_sortedness=True, ctx=None,
) -> datafusion.DataFrame
```

The arguments mean what they mean in polars:

- For every left row, the join takes the last right row with equal `by`
  values and `right[on] <= left[on]`. With `allow_exact_matches=False` it
  requires `<` instead.
- Every left row appears once, in left order. Right columns are null when
  nothing matches.
- `on` and `left_on`/`right_on` are mutually exclusive, and so are `by` and
  `by_left`/`by_right`. These are checked as in polars.
- `strategy` other than `"backward"` raises `NotImplementedError`.
- The output columns are the same as polars':
  - every left column first;
  - then the right columns, in order, except the `by` keys, and except the
    `on` column when it has the left `on` name (a differently named
    `right_on` is kept);
  - a right name that already exists gets `suffix`, and if that is taken too
    the call raises (polars: "column with name ... already exists").
- `on` may be any integer, float, date, time, timestamp or duration column,
  of the same type on both sides. `by` may be integers, strings, or several
  columns, but not floats.
- `ctx` is the `SessionContext` the inputs came from. datafusion-python 55
  has no way to get it from a DataFrame, so it is a parameter. It defaults to
  the global context. Its configuration (`target_partitions`, ...) is used to
  plan the inputs.

`datafusion_plugins.last_plan()` returns the physical plan of the most recent
join. Use it to see which mode ran and that your filters reached the scans.

## Contract: sort it yourself, like polars

With `by`, each input must be sorted by `(by..., on)`; without `by`, by
`on`. The plugin never sorts. If your data isn't sorted, sort it first,
ideally once and on disk.

- **Proven order.** DataFusion proves the order when you declare it with
  `read_parquet(..., file_sort_order=...)` or `register_parquet`, and keep it
  through filters and projections. Then the join only checks batch and
  partition boundaries, which is O(1) per batch.
- **Order not proven, `check_sortedness=True` (the default).** Every batch is
  validated as it streams, plus the boundaries. An unsorted input raises
  `left input is not sorted by (security_id, ts)` (or `right input ...`).
- **Order not proven, `check_sortedness=False`.** The data is trusted.
  Boundary violations still raise, but unsorted data elsewhere gives wrong
  results.
- **NULLs.** A left row with a NULL `on` or `by` never matches, and right
  rows with NULL keys are ignored, as in polars. In a declared order,
  `nulls_first` may be either.

## Rust API

The operators and the planning logic live in the `datafusion-plugins` crate,
which has no Python dependency. Until it is published, use a git dependency:

```toml
[dependencies]
datafusion-plugins = { git = "https://github.com/ndaskalovic/datafusion-plugins" }
datafusion = "55"
```

```rust
use datafusion::prelude::*;
use datafusion_plugins::JoinAsof;

let ctx = SessionContext::new();
let order = vec![vec![col("security_id").sort(true, true), col("ts").sort(true, true)]];
let opts = || ParquetReadOptions::default().file_sort_order(order.clone());
let trades = ctx.read_parquet("trades.parquet", opts()).await?.filter(col("qty").gt(lit(10)))?;
let quotes = ctx.read_parquet("quotes.parquet", opts()).await?;

let plan = JoinAsof::new("ts")            // or JoinAsof::new_with("ts", "quote_ts")
    .by(["security_id"])                  // or .by_left_right([..], [..])
    .allow_exact_matches(true)
    .build(
        trades.create_physical_plan().await?,
        quotes.create_physical_plan().await?,
        ctx.state().config_options(),
    )?;
let batches = datafusion::physical_plan::collect(plan, ctx.task_ctx()).await?;
```

The public API is:

- `JoinAsof`: the builder, with `output_columns`, `output_schema`, `build`
  and `build_with_outputs`.
- `AsofPlanTable`: the join of two datafusion-proto serialized logical
  plans, as a `TableProvider`. This is what the Python package uses.
- The operators: `StreamingAsOfJoinExec`, `OrderedConcatExec`, and
  `AsofJoinScanExec`, a leaf wrapper for plans that pass through another
  session's optimizer.
- `AsofOptions` (tuning knobs) and `last_plan`.

`AsofOptions::from_env()` reads these `DFASOF_*` variables:

| knob | default | meaning |
|---|---|---|
| `DFASOF_RIGHT_SPLIT_FACTOR` | 32 | right ranges = `target_partitions` × this |
| `DFASOF_LEFT_SPLIT_FACTOR` | 4 | left ranges (work units) = `target_partitions` × this |
| `DFASOF_WINDOW_EXTRA` | 2 | units in flight = `target_partitions` + this |
| `DFASOF_MAX_SOURCES` | 64 | right batches kept before compaction |
| `DFASOF_FANOUT_BELOW` | `target_partitions` | fan sparse left ranges out below this many non-empty ranges |
| `DFASOF_VALIDATE=1` | off | validate every batch, even with a proven order |
| `DFASOF_ORDERED=0` | on | disable the ordered output (and so the fast mode) |

The Python package runs its plans on its own tokio runtime, with
`DATAFUSION_PLUGINS_THREADS` worker threads (default: the number of CPUs).

## How it works

**One kernel.** Each batch's keys become one order-preserving i128 per row:
the group in the high 64 bits and `on` in the low 64 bits. Without `by` the
group is 0. Multi-column or string `by` keys use the Arrow row format
instead, through the same loop.

For each left row, the kernel gallops the right cursor to the last right
row `<=` the left key (`<` when strict) and takes it if it is in the same
group. One candidate row is carried along; there is no hash table. Right
batches that the candidates still point into are compacted, so memory is
O(groups + batch size).

**Fast mode** runs when both inputs are re-executable scans of one Parquet
file each, with a proven order, optionally under filters and projections:

- Both scans are split into ordered byte ranges: right into
  `target_partitions × 32` ranges, left into `× 4`.
- A short prologue peeks at the first key of every left range and probes the
  first key of every right range.
- Each left range is a work unit. It binary-searches the right range that
  holds its first row's match, then merges, skipping whole right ranges when
  the left jumps ahead.
- A sparse left side (fewer non-empty ranges than threads) fans each range
  out over right sub-ranges.
- `OrderedConcatExec` runs a sliding window of `target_partitions + 2` units
  and emits their outputs in order as one partition.

**Execute-once mode** covers every other input, such as joins, aggregates or
undeclared orders. Every input partition runs exactly once:

- **Left side.** It is one stream: its partitions are read in order when they
  are consecutive key ranges (a split scan, possibly under a broadcast
  join's probe side), and merged otherwise.
- **Right side.**
  - With consecutive, proven ranges, each partition is peeked once for its
    first key and becomes a unit.
  - Otherwise the right side is a single unit.
- **Distributor.** A serial distributor slices the left batches at unit
  boundaries by galloping on the keys. A left row's match is always inside
  its unit.
- **Workers and output.** The workers run the same kernel. Their outputs are
  concatenated in unit order, with at most `target_partitions + 2` units
  active.
- **Memory** stays bounded. Each active unit has bounded channels of left
  slices and output batches, and each unit not yet started holds one peeked
  batch.

## How DataFrames get across (datafusion-python 55)

datafusion-python 55 cannot export a DataFrame as an FFI table provider, so
`join_asof`:

1. serializes each input's logical plan (`df.logical_plan().to_bytes(ctx)`,
   datafusion-proto),
2. hands the bytes to the plugin, and
3. wraps the FFI table provider it returns with `ctx.read_table(...)`.

When the result is scanned, the plugin decodes the plans in its own
`SessionContext`, configured from the scanning session's options, and plans
them with DataFusion's optimizer. Filter and projection pushdown into the
input scans therefore happen as usual, including Parquet pruning predicates,
and the declared orders are visible to the join. The join plan goes back to
the host as one leaf node, so the host's optimizer leaves it alone.
Execution runs on the plugin's own threads.

Another route also exists: reaching the host's tables through the FFI
session's catalog list. It was not used, because running host plans on the
plugin's threads would need the host's tokio runtime, and a tokio handle
cannot be shared across the two libraries.

## Limitations

- **DataFusion 55 only** (`datafusion==55.*`), because of the FFI ABI.
- **No sorting.** Unsorted inputs are an error (with `check_sortedness`) or
  wrong results (without it). This matches polars.
- **Inputs must be file-sourced**: Parquet, CSV or JSON files under standard
  operators. datafusion-proto cannot serialize in-memory DataFrames
  (`from_pydict`, `from_polars`, ...) or Python UDFs on DataFusion 55, and
  `join_asof` raises a `ValueError` for them. They need a native route in
  DataFusion 56.
- **Local filesystem only.** The plugin's session has no object stores (no
  S3, GCS or Azure).
- `strategy="backward"` only.
- Float `by` keys are not supported.
- Fast mode needs a proven order and one Parquet file per side. Anything
  else runs in execute-once mode, which is serial when the right side is one
  unit.
- A computed `on` column, like `with_column("ts", col("ts") + 1)`, loses its
  proven order in DataFusion 55. That input is validated instead.
- Native `ASOF JOIN` SQL syntax arrives with DataFusion 56. Hooking it to this
  operator through a query-planner extension is future work.

## Benchmarks

### Prototype, DataFusion `main`, CLI

These numbers were measured with the prototype of this operator on DataFusion
`main` (rev b645de03), using a `datafusion-cli` look-alike: LTO, 5 threads,
300M right rows (3000 securities), median of 7 runs (3 for grouped_ts).
Outputs are identical to polars, including row order.

| case | polars 2.0 | this operator | DataFusion AsOfJoinExec |
|---|---|---|---|
| grouped (sec,ts), 30M left | 1.63 s / 1.22 GB | 1.62 s / 0.81 GB | 44.6 s / 8.4 GB |
| grouped, 300k left (sparse) | 1.62 s / 11.57 GB | 1.11 s / 0.26 GB | 32.9 s / 8.4 GB |
| ungrouped (ts), 30M left | 1.29 s / 1.01 GB | 1.66 s / 0.82 GB | 31.4 s / 7.0 GB |
| grouped on ts-only data | 63.2 s / 15.3 GB | 2.61 s / 1.52 GB | 86.0 s / 18.8 GB (1 run) |
| grouped, 4 GB cap (1 run) | 2.42 s / 1.03 GB | 2.27 s / 0.80 GB | fails: broadcast can't spill |

### This package, datafusion-python 55 (smoke benchmark)

`python bench/bench.py`: Parquet files sorted by `(security_id, ts)`, 5
threads (`target_partitions=5`, `DATAFUSION_PLUGINS_THREADS=5`,
`POLARS_MAX_THREADS=5`), materialized results, median after one warm-up run,
peak RSS of the process. M5 MacBook Air (fanless; measured after a
cooldown), datafusion 55.0.0, polars 2.0.0, 2026-10-15.

The cases:

- `polars`: `scan_parquet(l).join_asof(scan_parquet(r), on="ts", by="security_id").collect()`
- `df-frames`: `join_asof` on `read_parquet(..., file_sort_order=...)` DataFrames
- `*-pipe`: the README example, with `.filter(qty > 10)` and a computed
  `notional` column on the left first (polars: `scan_parquet().filter().with_columns().join_asof()`)
- `df-once`: no declared order, so execute-once mode with every batch
  validated

20M right rows, 2M left, 1000 securities, 5 runs:

| case | median | peak RSS |
|---|---|---|
| polars | 0.169 s | 1.77 GB |
| df-frames | 0.214 s | 0.32 GB |
| polars-pipe | 0.174 s | 1.78 GB |
| df-pipe | 0.223 s | 0.32 GB |
| df-once | 0.446 s | 0.22 GB |

100M right rows, 10M left, 3000 securities, 3 runs:

| case | median | peak RSS |
|---|---|---|
| polars | 0.814 s | 6.42 GB |
| df-frames | 0.691 s | 0.65 GB |
| polars-pipe | 0.862 s | 6.52 GB |
| df-pipe | 0.721 s | 0.70 GB |

On the 20M files, the prototype's CLI takes 0.21–0.23 s in the same session,
so the Python/FFI route adds nothing measurable. At this size fixed costs
(range probes, planning) are a larger share, and polars is about 25% faster.
At 100M rows the plugin is about 15% faster. It always uses 6–9x less
memory.

## Development

```sh
CARGO_BUILD_JOBS=5 cargo test -p datafusion-plugins          # Rust differential tests
FANOUT=1 cargo test -p datafusion-plugins --test compare     # force the sparse-left fan-out
maturin develop --release && pytest                          # Python tests
python bench/bench.py                                        # smoke benchmark
```

The Rust tests compare against a brute-force reference, on four paths:
ordered scans (fast mode), a merged left input (execute-once with right
units), undeclared orders (execute-once, validated), and the plan-bytes
provider. They also check the not-sorted error.

The Python tests compare against polars `join_asof`, including the output
schema for each argument combination, and check that the output row order
equals the left input order.

## License

Apache-2.0, the same as DataFusion.
