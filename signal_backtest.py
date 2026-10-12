"""回測從群組交易訊號回推的順勢突破規則：python signal_backtest.py

1. 先確認規則在三則真實訊號的時間點會不會觸發
2. 再用一年多的 1H K 線，在派網流動性夠的永續合約上跑一遍，看長期期望值

出場：TP1 平一半並把停損移到進場價，剩下一半在 TP2 或停損出場，最多持有 7 天。
同一根 K 線同時碰到停損和停利時，保守當作先停損。
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import pandas as pd

from grid_trading.breakout import BreakoutConfig, add_indicators, signal_at
from grid_trading.history import fetch_history
from grid_trading.pionex import PionexPublic

# 截圖中的三則訊號（時間已從台北時間換成 UTC）
KNOWN = [
    ("AERO_USDT_PERP", "2026-10-10 10:26"),
    ("S_USDT_PERP", "2026-10-11 13:58"),
    ("ZRO_USDT_PERP", "2026-10-12 04:35"),
]


def simulate(df: pd.DataFrame, cfg: BreakoutConfig, cost: float, max_hold: int = 168) -> list[dict]:
    """cost 是單邊手續費加滑價，結果以 R（停損距離）和 1 倍槓桿報酬率表示。"""
    trades = []
    # 先用向量化粗篩，只在候選 K 線上跑完整判斷
    cand = (df["close"] > df["prior_high"]) & (df["rsi"].between(cfg.rsi_min, cfg.rsi_max))
    candidates = list(df.index[cand])
    i, n = 0, len(df)
    while i < n - 1:
        candidates = [c for c in candidates if c >= i]
        if not candidates:
            break
        i = candidates.pop(0)
        if i >= n - 1:
            break
        sig = signal_at(df.iloc[i], cfg)
        if sig is None:
            i += 1
            continue
        entry, stop, risk = sig["entry"], sig["stop"], sig["entry"] - sig["stop"]
        half_done, pnl, j = False, 0.0, i + 1
        exit_reason = "time"
        while j < n and j <= i + max_hold:
            bar = df.iloc[j]
            if bar["low"] <= stop:
                pnl += (0.5 if half_done else 1.0) * (stop - entry)
                exit_reason = "be" if half_done else "stop"
                break
            if not half_done and bar["high"] >= sig["tp1"]:
                pnl += 0.5 * (sig["tp1"] - entry)
                half_done, stop = True, entry
            if half_done and bar["high"] >= sig["tp2"]:
                pnl += 0.5 * (sig["tp2"] - entry)
                exit_reason = "tp2"
                break
            j += 1
        else:
            j = min(j, n - 1)
            pnl += (0.5 if half_done else 1.0) * (df["close"].iloc[j] - entry)
        fee = 2 * cost * entry
        trades.append({
            "time": df["time"].iloc[i], "entry": entry, "risk_pct": sig["risk_pct"],
            "exit": exit_reason, "hit_tp1": half_done, "bars": j - i,
            "r": (pnl - fee) / risk, "ret": (pnl - fee) / entry,
        })
        i = j + 1  # 同一幣種一次只開一單
    return trades


def check_known(cfg: BreakoutConfig) -> None:
    print("三則真實訊號當下，規則是否觸發（訊號前最後兩根已收盤 1H）：")
    for sym, ts in KNOWN:
        t = pd.Timestamp(ts, tz="UTC")
        df = add_indicators(fetch_history(sym, "60M", 120), cfg)
        done = df[df["bar_close"] <= t].tail(2)
        hits = [signal_at(r, cfg) is not None for _, r in done.iterrows()]
        r = done.iloc[-1]
        print(f"  {sym:16s} 觸發={any(hits)}  收盤={r.close:.4g} 前20高={r.prior_high:.4g} "
              f"RSI={r.rsi:.1f} 4H趨勢={r.h4_trend} 4H MACD={r.h4_macd}")


def main() -> int:
    p = argparse.ArgumentParser(description="順勢突破訊號回測")
    p.add_argument("--days", type=int, default=420)
    p.add_argument("--min-volume", type=float, default=5e6)
    p.add_argument("--cost", type=float, default=0.001, help="單邊手續費+滑價，預設 0.1%%")
    p.add_argument("--symbols", nargs="*")
    p.add_argument("--out", default="output")
    args = p.parse_args()

    check_known(BreakoutConfig())

    if args.symbols:
        symbols = args.symbols
    else:
        api = PionexPublic()
        tickers = api.perp_tickers()
        liquid = set(tickers.loc[tickers["amount"] >= args.min_volume, "symbol"])
        symbols = sorted(s for s in api.perp_symbols() if s in liquid)

    data = {}
    for sym in symbols:
        df = fetch_history(sym, "60M", args.days)
        if len(df) > 24 * 60:
            data[sym] = df
    print(f"\n{len(data)} 個幣種，1H 資料", file=sys.stderr)

    configs = [
        BreakoutConfig(rsi_max=rmax, use_4h_filter=f, tp2_r=tp2)
        for rmax, f, tp2 in itertools.product([75.0, 80.0], [True, False], [2.0, 3.0])
    ]
    rows, summary = [], []
    for cfg in configs:
        trades = []
        for sym, df in data.items():
            for t in simulate(add_indicators(df, cfg), cfg, args.cost):
                trades.append({**t, "symbol": sym, "variant": cfg.label()})
        res = pd.DataFrame(trades)
        if res.empty:
            continue
        rows.append(res)
        mid = res["time"].min() + (res["time"].max() - res["time"].min()) / 2
        for period, part in (("全部", res), ("前半", res[res["time"] < mid]), ("後半", res[res["time"] >= mid])):
            summary.append({
                "variant": cfg.label(), "period": period, "trades": len(part),
                "tp1_rate": part["hit_tp1"].mean(), "stop_rate": (part["exit"] == "stop").mean(),
                "avg_r": part["r"].mean(), "total_r": part["r"].sum(),
                "avg_ret_pct": 100 * part["ret"].mean(),
            })
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    summ = pd.DataFrame(summary)
    summ.to_csv(out / "signal_backtest_summary.csv", index=False)
    pd.concat(rows).to_csv(out / "signal_backtest_trades.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_rows", 100):
        print(summ.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
