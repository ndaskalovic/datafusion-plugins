"""Smoke benchmark: datafusion_plugins vs polars join_asof (not run in CI).

    python bench/bench.py [--right 20_000_000] [--left 2_000_000] [--groups 1000]
                          [--threads 5] [--reps 5]

Generates Parquet files sorted by (security_id, ts) under bench/data/ (once),
then times each case in its own subprocess (median of --reps after one
warm-up run; peak RSS of that subprocess):

* polars:       scan_parquet(l).join_asof(scan_parquet(r), by=...).collect()
* polars-pipe:  scan_parquet(l).filter(qty > 10).with_columns(notional=...)
                .join_asof(scan_parquet(r), ...).collect()
* df-frames:    datafusion_plugins.join_asof on read_parquet DataFrames
                (declared file_sort_order, no other ops)
* df-pipe:      same, after .filter(qty > 10).with_column("notional", ...)
* df-once:      (opt-in) no declared order: execute-once mode, each batch
                validated (check_sortedness=True)

Threads: SessionConfig().with_target_partitions(N), DATAFUSION_PLUGINS_THREADS=N,
POLARS_MAX_THREADS=N. Results are materialized (polars DataFrame / Arrow
batches).
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"


def generate(n_right: int, n_left: int, groups: int) -> tuple[Path, Path]:
    import numpy as np
    import polars as pl

    DATA.mkdir(exist_ok=True)
    rp = DATA / f"quotes_{n_right}_{groups}.parquet"
    lp = DATA / f"trades_{n_left}_{groups}.parquet"
    if rp.exists() and lp.exists():
        return lp, rp
    rng = np.random.default_rng(42)
    span = 6 * 3600 * 10**9  # six hours of ns timestamps

    def frame(n: int, payload: dict[str, np.ndarray]) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "security_id": rng.integers(0, groups, n).astype(np.int32),
                "ts": rng.integers(0, span, n).astype(np.int64),
                **payload,
            }
        ).sort(["security_id", "ts"])

    quotes = frame(
        n_right,
        {"bid": rng.random(n_right) * 100, "ask": rng.random(n_right) * 100 + 0.01},
    )
    quotes.write_parquet(rp)
    del quotes
    trades = frame(
        n_left,
        {"qty": rng.integers(1, 100, n_left).astype(np.int64), "price": rng.random(n_left) * 100},
    )
    trades.write_parquet(lp)
    return lp, rp


def run_case(case: str, lp: str, rp: str, threads: int, reps: int) -> dict:
    """Runs inside the subprocess."""
    times = []
    rows = 0
    if case.startswith("polars"):
        import polars as pl

        def once() -> int:
            left = pl.scan_parquet(lp)
            if case == "polars-pipe":
                left = left.filter(pl.col("qty") > 10).with_columns(
                    notional=pl.col("qty") * pl.col("price")
                )
            out = left.join_asof(
                pl.scan_parquet(rp), on="ts", by="security_id", strategy="backward"
            ).collect()
            return out.height
    else:
        from datafusion import SessionConfig, SessionContext, col

        import datafusion_plugins as dp

        ctx = SessionContext(SessionConfig().with_target_partitions(threads))
        order = [[col("security_id").sort(), col("ts").sort()]]

        def once() -> int:
            if case == "df-once":
                left, right = ctx.read_parquet(lp), ctx.read_parquet(rp)
            else:
                left = ctx.read_parquet(lp, file_sort_order=order)
                if case == "df-pipe":
                    left = left.filter(col("qty") > 10).with_column(
                        "notional", col("qty") * col("price")
                    )
                right = ctx.read_parquet(rp, file_sort_order=order)
            df = dp.join_asof(left, right, on="ts", by="security_id", ctx=ctx)
            return sum(b.num_rows for b in df.collect())

    once()  # warm-up (page cache, metadata)
    for _ in range(reps):
        t0 = time.perf_counter()
        rows = once()
        times.append(time.perf_counter() - t0)
    return {"case": case, "median_s": statistics.median(times), "min_s": min(times), "rows": rows}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--right", type=int, default=20_000_000)
    ap.add_argument("--left", type=int, default=2_000_000)
    ap.add_argument("--groups", type=int, default=1000)
    ap.add_argument("--threads", type=int, default=5)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--cases", default="polars,df-frames,polars-pipe,df-pipe")
    ap.add_argument("--child", nargs=3, metavar=("CASE", "LEFT", "RIGHT"))
    args = ap.parse_args()

    if args.child:
        case, lp, rp = args.child
        print(json.dumps(run_case(case, lp, rp, args.threads, args.reps)))
        return

    lp, rp = generate(args.right, args.left, args.groups)
    env = dict(
        os.environ,
        POLARS_MAX_THREADS=str(args.threads),
        DATAFUSION_PLUGINS_THREADS=str(args.threads),
    )
    print(
        f"right={args.right:,} left={args.left:,} groups={args.groups} "
        f"threads={args.threads} reps={args.reps} (median, after 1 warm-up)"
    )
    print(f"{'case':<12} {'median':>9} {'min':>9} {'peak RSS':>10} {'rows':>12}")
    for case in args.cases.split(","):
        proc = subprocess.Popen(
            [sys.executable, __file__, "--threads", str(args.threads), "--reps", str(args.reps),
             "--child", case, str(lp), str(rp)],
            env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        stdout, stderr = proc.stdout.read(), proc.stderr.read()
        _, status, usage = os.wait4(proc.pid, 0)
        proc.returncode = os.waitstatus_to_exitcode(status)
        if proc.returncode != 0:
            print(f"{case:<12} failed:\n{stderr}")
            continue
        r = json.loads(stdout.strip().splitlines()[-1])
        # ru_maxrss: bytes on macOS, KiB on Linux.
        scale = 1 if sys.platform == "darwin" else 1024
        rss = f"{usage.ru_maxrss * scale / 2**30:.2f} GB"
        print(f"{case:<12} {r['median_s']:>8.3f}s {r['min_s']:>8.3f}s {rss:>10} {r['rows']:>12,}")

if __name__ == "__main__":
    main()
