"""持倉量 × 價格的擴大參數表：python -m research.orderflow_oi_check --tf 4h（或 2h）

orderflow_backtest.py 只測回看 1/3/6 根。這裡把回看拉到 3/6/12/18 根、門檻 1.5~3σ，
並加上「只看價格、不看持倉量」的對照組（price_only），確認持倉量有沒有多貢獻。
輸出 output/orderflow/oi_check_{tf}.csv 和每筆交易。
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

from research import orderflow_backtest as ob
from grid_trading import signals
from grid_trading.orderflow import add_features, flow_bars
from grid_trading.signals.evaluate import trades_for

CONFIGS = {
    f"oi n{n} k{k} {mode}{' +trend' if trend else ''}": {"n": n, "k": k, "mode": mode, "trend_filter": trend}
    for n in (3, 6, 12, 18) for k in (1.5, 2.0, 2.5, 3.0)
    for mode in ("build", "price_only", "unwind_follow") for trend in (False, True)
}


def run(job) -> list[dict]:
    coin, tf = job
    df = flow_bars(coin, tf)
    if len(df) < 300:
        return []
    df = add_features(df, windows=(1, 3, 6, 12, 18))
    rows = []
    for label, params in CONFIGS.items():
        for t in trades_for(signals.get("of_oi", **params), df, coin, tf, ob.FEE, ob.MAX_BARS[tf]):
            t.pop("strict", None)
            rows.append(dict(t, coin=coin, variant=label))
    return rows


def main() -> int:
    p = argparse.ArgumentParser(description="持倉量 × 價格的擴大參數表")
    p.add_argument("--tf", default="4h", choices=["4h", "2h"])
    p.add_argument("--out", default="output/orderflow")
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(os.cpu_count() or 2) as ex:
        trades = pd.DataFrame([r for part in ex.map(run, [(c, args.tf) for c in ob.coins()]) for r in part])
    trades.to_csv(out / f"oi_check_{args.tf}_trades.csv.gz", index=False)
    rows = []
    for label, g in trades.groupby("variant"):
        is_, oos = g[g["time"] < ob.SPLIT], g[g["time"] >= ob.SPLIT]
        rows.append({"variant": label, "is_n": len(is_), "is_exp": is_["r"].mean(), "oos_n": len(oos),
                     "oos_exp": oos["r"].mean(), "oos_se": oos["r"].std() / math.sqrt(len(oos)) if len(oos) > 1 else None,
                     "all_exp": g["r"].mean()})
    summary = pd.DataFrame(rows)
    summary.to_csv(out / f"oi_check_{args.tf}.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_rows", 300):
        print(summary.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
