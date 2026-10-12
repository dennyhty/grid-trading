"""多週期比較：python compare_timeframes.py --group crypto（或 rwa）

同一批幣、同一套出場規則（tracker：TP1/2/3 各平 1/3、TP1 後保本），比較週線到 1 小時線的表現。
只讀 data/ 裡已下載的 K 線快取（先用 grid_trading.history.fetch_history 抓 60M、4H、1D）。

派網只提供到日線，而且最早只到 2023-01；1 小時線只能往回查約 14 個月。所以：
- 1H、4H、1D 用派網原生 K 線；2H 由 1H 合成；2D、3D、週線（週一開盤）由日線合成。
- 2H 和 1H 一樣只有約 14 個月。另外算一欄「共同期間」（COMMON_START 之後），各週期在同一段時間比較。
- 結合策略的大週期 = 進場週期 × 4（1H 看 4H、日線看 4 日線）。停損距離的上下限照週期放寬：
  1H 是 3%~12%，其他週期乘上 √(週期小時數)，上限分別封頂在 10% 和 40%。
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

from grid_trading import signals
from grid_trading.history import resample
from grid_trading.signals.evaluate import split_summary, trades_for
from grid_trading.universe import RWA, RWA_COMMODITIES, perp_symbols

# 週期 → (快取的派網週期, 合成規則, 小時數)
TIMEFRAMES = {
    "1W": ("1D", "168h", 168),
    "3D": ("1D", "72h", 72),
    "2D": ("1D", "48h", 48),
    "1D": ("1D", None, 24),
    "4H": ("4H", None, 4),
    "2H": ("60M", "2h", 2),
    "1H": ("60M", None, 1),
}
SOURCE_HOURS = {"60M": 1, "4H": 4, "1D": 24}
VARIANTS = {
    "維加斯趨勢(原版)": ("vegas_trend", {}),
    "突破(群組版,只做多)": ("vegas_breakout", {"trend": "basic", "allow_short": False}),
    "結合(只做多)": ("vegas_breakout", {"allow_short": False}),
    "結合(多空)": ("vegas_breakout", {}),
    "結合(多空,同週期趨勢)": ("vegas_breakout", {"trend_mult": 1}),
}
COMMON_START = pd.Timestamp("2025-09-01", tz="UTC")  # 1H 快取從 2025-08 下旬開始


def cached(sym: str, interval: str) -> pd.DataFrame:
    path = Path(f"data/{sym}_{interval}.csv.gz")
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True)
    return df


def bars(sym: str, tf: str) -> pd.DataFrame:
    """回傳該週期已收盤的 K 線。合成時最後一組如果還沒湊滿就丟掉。"""
    src, rule, hours = TIMEFRAMES[tf]
    df = cached(sym, src)
    if df.empty or rule is None:
        return df
    # 週線從週一 00:00 UTC 起算（2023-01-02 是週一），其他從第一天 00:00 起算
    out = resample(df, rule, pd.Timestamp("2023-01-02", tz="UTC") if rule == "168h" else "start_day")
    last_close = df["time"].iloc[-1] + pd.Timedelta(hours=SOURCE_HOURS[src])
    out = out[out["time"] + pd.Timedelta(hours=hours) <= last_close]
    return out.reset_index(drop=True)


def strategy_for(label: str, tf: str):
    name, params = VARIANTS[label]
    params = dict(params)
    if name == "vegas_breakout":
        h = TIMEFRAMES[tf][2]
        params.setdefault("min_risk", min(0.03 * math.sqrt(h), 0.10))
        params.setdefault("max_risk", min(0.12 * math.sqrt(h), 0.40))
    return signals.get(name, **params)


def run(job):
    tf, label, sym, tier, fee = job
    df = bars(sym, tf)
    strat = strategy_for(label, tf)
    if len(df) < strat.min_bars + 20:
        return []
    return [dict(t, tf=tf, variant=label, tier=tier) for t in trades_for(strat, df, sym, tf, fee)]


def summarize(t: pd.DataFrame, tiers: list[str]) -> dict:
    s = split_summary(t)
    a = s["all"]
    common = t[t["time"] >= COMMON_START]
    row = {"trades": a["trades"], "symbols": s["symbols"], "start": t["time"].min().date(),
           "win_rate": a["win_rate"], "exp_r": a["expectancy_r"], "se": t["r"].std() / math.sqrt(len(t)),
           "first_half_r": s["first_half"].get("expectancy_r"), "second_half_r": s["second_half"].get("expectancy_r"),
           "long_r": t.loc[t["side"] == "LONG", "r"].mean(), "short_r": t.loc[t["side"] == "SHORT", "r"].mean(),
           "common_trades": len(common), "common_r": common["r"].mean(),
           "max_dd_pct": a["max_drawdown_pct"],
           "avg_hold_days": (t["exit_time"] - t["time"]).dt.total_seconds().mean() / 86400}
    for k in tiers:
        row[f"{k}_r"] = t.loc[t["tier"] == k, "r"].mean()
    return row


def main() -> int:
    p = argparse.ArgumentParser(description="多週期策略比較")
    p.add_argument("--group", choices=["crypto", "rwa"], default="crypto")
    p.add_argument("--fee", type=float, default=0.0005)
    p.add_argument("--timeframes", default=",".join(TIMEFRAMES))
    p.add_argument("--out", default="output")
    args = p.parse_args()
    if args.group == "rwa":
        syms = perp_symbols(RWA)
        tier = {s: ("商品" if s.split("_")[0] in RWA_COMMODITIES else "股票") for s in syms}
    else:
        syms = perp_symbols()
        tier = {s: ("市值前20" if i < 20 else "市值21-100") for i, s in enumerate(syms)}
    tfs = args.timeframes.split(",")
    jobs = [(tf, v, s, tier[s], args.fee) for tf in tfs for v in VARIANTS for s in syms]
    rows = []
    with ProcessPoolExecutor(os.cpu_count() or 2) as ex:
        for r in ex.map(run, jobs, chunksize=2):
            rows += r
    trades = pd.DataFrame(rows)
    tiers = sorted(set(tier.values()))
    out = []
    for tf in tfs:
        for v in VARIANTS:
            t = trades[(trades["tf"] == tf) & (trades["variant"] == v)] if not trades.empty else trades
            out.append({"tf": tf, "variant": v, **(summarize(t, tiers) if len(t) else {"trades": 0})})
    summary = pd.DataFrame(out)
    Path(args.out).mkdir(exist_ok=True)
    summary.to_csv(Path(args.out) / f"timeframes_{args.group}.csv", index=False)
    trades.to_csv(Path(args.out) / f"timeframes_{args.group}_trades.csv.gz", index=False)
    with pd.option_context("display.width", 300, "display.max_columns", 30):
        print(summary.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
