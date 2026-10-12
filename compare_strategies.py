"""1H 策略比較：python compare_strategies.py --group crypto（或 rwa）

在同一批幣上用訊號網頁的出場規則（tracker：TP1/2/3 各平 1/3、TP1 後保本）回測多個策略版本，
比較期望值、前後半段、多空、油價資料開始後（2026-03-20 起）的表現。只讀已下載的 K 線快取。
"""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

import grid_trading.signals.vegas_breakout as vb
from grid_trading import signals
from grid_trading.signals.evaluate import split_summary, trades_for
from grid_trading.universe import RWA, RWA_COMMODITIES, perp_symbols

VARIANTS = {
    "維加斯趨勢(原版)": ("vegas_trend", {}),
    "突破(群組版,只做多)": ("vegas_breakout", {"trend": "basic", "allow_short": False}),
    "結合(只做多)": ("vegas_breakout", {"allow_short": False}),
    "結合(多空)": ("vegas_breakout", {}),
    "結合+避開RSI背離": ("vegas_breakout", {"divergence": "rsi"}),
    "結合+避開MACD背離": ("vegas_breakout", {"divergence": "macd"}),
    "結合+避開任一背離": ("vegas_breakout", {"divergence": "any"}),
    "結合+油價同向": ("vegas_breakout", {"oil": "with"}),
    "結合+油價反向": ("vegas_breakout", {"oil": "against"}),
    "結合+完整Vegas均線組": ("vegas_breakout", {"ribbon": True}),
    "結合+200MA慢線": ("vegas_breakout", {"slow_ma": 200}),
    "結合+HMA55快線": ("vegas_breakout", {"hma": 55}),
    "結合+PSAR": ("vegas_breakout", {"psar": True}),
    "結合+均線組+200MA+HMA+PSAR": ("vegas_breakout", {"ribbon": True, "slow_ma": 200, "hma": 55, "psar": True}),
}
OIL_START = pd.Timestamp("2026-03-20", tz="UTC")


def cached(sym: str, interval: str = "60M", days: int = 420, refresh: bool = False) -> pd.DataFrame:
    path = Path(f"data/{sym}_{interval}.csv.gz")
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True)
    return df


vb.fetch_history = cached  # 回測只用快取，避免多個程序同時重抓同一份資料


def run(job):
    label, sym, tier, fee = job
    name, params = VARIANTS[label]
    df = cached(sym)
    strat = signals.get(name, **params)
    if len(df) < strat.min_bars + 50:
        return []
    return [dict(t, variant=label, tier=tier) for t in trades_for(strat, df, sym, "60M", fee)]


def main() -> int:
    p = argparse.ArgumentParser(description="1H 策略比較")
    p.add_argument("--group", choices=["crypto", "rwa"], default="crypto")
    p.add_argument("--fee", type=float, default=0.0005)
    p.add_argument("--out", default="output")
    args = p.parse_args()
    if args.group == "rwa":
        syms = perp_symbols(RWA)
        tier = {s: ("商品" if s.split("_")[0] in RWA_COMMODITIES else "股票") for s in syms}
    else:
        syms = perp_symbols()
        tier = {s: ("市值前20" if i < 20 else "市值21-100") for i, s in enumerate(syms)}
    jobs = [(v, s, tier[s], args.fee) for v in VARIANTS for s in syms]
    rows = []
    with ProcessPoolExecutor(os.cpu_count() or 2) as ex:
        for r in ex.map(run, jobs, chunksize=4):
            rows += r
    trades = pd.DataFrame(rows)
    out = []
    for v in VARIANTS:
        t = trades[trades["variant"] == v]
        if t.empty:
            continue
        s = split_summary(t)
        a = s["all"]
        row = {"variant": v, "trades": a["trades"], "win_rate": a["win_rate"], "exp_r": a["expectancy_r"],
               "first_half_r": s["first_half"].get("expectancy_r"), "second_half_r": s["second_half"].get("expectancy_r"),
               "long_r": t.loc[t["side"] == "LONG", "r"].mean(), "short_r": t.loc[t["side"] == "SHORT", "r"].mean(),
               "since_oil_trades": int((t["time"] >= OIL_START).sum()),
               "since_oil_r": t.loc[t["time"] >= OIL_START, "r"].mean()}
        for k in sorted(set(tier.values())):
            row[f"{k}_r"] = t.loc[t["tier"] == k, "r"].mean()
        out.append(row)
    summary = pd.DataFrame(out)
    Path(args.out).mkdir(exist_ok=True)
    summary.to_csv(Path(args.out) / f"compare_{args.group}.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(summary.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
