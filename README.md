# datafusion-plugins

Custom operators for [Apache DataFusion](https://datafusion.apache.org), usable from
`datafusion-python` and from Rust.

```bash
pip install git+https://github.com/ndaskalovic/datafusion-plugins   # needs a Rust toolchain
```

Requires `datafusion==55.*`. Inputs must be file-backed; in-memory DataFrames are not supported yet.

## Available

### `join_asof`

Streaming asof join with polars-style arguments. Inputs must be sorted by `(by..., on)`, or by `on`
when `by` is omitted.

```python
from datafusion import SessionContext, col
import datafusion_plugins as dp

ctx = SessionContext()
order = [[col("security_id").sort(), col("ts").sort()]]
trades = ctx.read_parquet("trades.parquet", file_sort_order=order).filter(col("qty") > 10)
quotes = ctx.read_parquet("quotes.parquet", file_sort_order=order)

out = dp.join_asof(trades, quotes, on="ts", by="security_id", ctx=ctx)  # a DataFrame
```

## License

Apache-2.0
