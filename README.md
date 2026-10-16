# datafusion-plugins

Custom operators for [Apache DataFusion](https://datafusion.apache.org), usable from
`datafusion-python` and from Rust.

```bash
pip install git+https://github.com/ndaskalovic/datafusion-plugins   # needs a Rust toolchain
```

Requires `datafusion==55.*`. Inputs can be any DataFrame: files, in-memory tables, Python UDFs, joins,
aggregates.

## Available

### `join_asof`

Streaming asof join with polars-style arguments. Inputs are sorted by `(by..., on)`, or by `on` when
`by` is omitted: DataFusion sorts only what it can't prove sorted, or pass `assume_sorted=True` to
trust the order (checked while streaming).

```python
from datafusion import SessionContext, col
import datafusion_plugins as dp

ctx = SessionContext()
order = [[col("security_id").sort(), col("ts").sort()]]
trades = ctx.read_parquet("trades.parquet", file_sort_order=order).filter(col("qty") > 10)
quotes = ctx.read_parquet("quotes.parquet", file_sort_order=order)

out = dp.join_asof(trades, quotes, on="ts", by="security_id", ctx=ctx)  # a DataFrame
```

### `join_asof_markouts`

Asof joins at several time offsets (markouts, in seconds) in one pass over the right side. `on` must
be a timestamp. Same ordering rules as `join_asof`, but `assume_sorted` defaults to `True`. Right
columns come out suffixed `_{h}s`.

```python
out = dp.join_asof_markouts(trades, quotes, on="ts", markouts=[1, 5, 60], by="security_id", ctx=ctx)
# columns: trades..., bid_1s, ask_1s, bid_5s, ask_5s, bid_60s, ask_60s
```

## License

Apache-2.0
