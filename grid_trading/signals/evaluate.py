"""訊號策略回測：把策略在歷史 K 線上逐根掃一遍，每個訊號用 tracker 跑完，統計勝率和期望值。

同一個幣同一時間只持有一張單：前一張還沒結束時出現的新訊號略過（和實際照訊號下單一致）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .strategy import Strategy
from .tracker import track


def trades_for(strategy: Strategy, df: pd.DataFrame, symbol: str, interval: str,
               fee: float = 0.0005, max_bars: int = 240) -> list[dict]:
    df = df.reset_index(drop=True)
    busy_until = None
    out = []
    for sig in strategy.scan(df, symbol, interval):
        if busy_until is not None and sig.time < busy_until:
            continue
        res = track(sig, df, fee, max_bars)
        if res.status != "CLOSED":
            continue  # 回測尾端還沒結束的不算
        exit_time = res.events[-1].time
        busy_until = exit_time
        out.append({
            "symbol": symbol, "side": sig.side, "time": sig.time, "exit_time": exit_time,
            "entry": sig.price, "r": res.realized_r, "pct": res.realized_pct, "tps_hit": res.tps_hit,
            "exit": res.events[-1].kind, "bars": res.bars,
        })
    return out


def summarize(trades: pd.DataFrame, risk_per_trade: float = 0.01) -> dict:
    """勝率、平均 R、期望值等。risk_per_trade：每張單願意賠總資金的比例（1R），用來換算資金曲線。"""
    if trades.empty:
        return {"trades": 0}
    r = trades["r"]
    trades = trades.sort_values("exit_time")
    equity = (1 + risk_per_trade * trades["r"]).cumprod()
    dd = (equity / equity.cummax() - 1).min()
    wins = r[r > 0]
    losses = r[r <= 0]
    return {
        "trades": int(len(r)),
        "win_rate": float((r > 0).mean()),
        "tp1_rate": float((trades["tps_hit"] >= 1).mean()),
        "avg_win_r": float(wins.mean()) if len(wins) else 0.0,
        "avg_loss_r": float(losses.mean()) if len(losses) else 0.0,
        "expectancy_r": float(r.mean()),
        "profit_factor": float(wins.sum() / -losses.sum()) if len(losses) and losses.sum() < 0 else float("inf"),
        "total_return_pct": float((equity.iloc[-1] - 1) * 100),
        "max_drawdown_pct": float(dd * 100),
        "risk_per_trade_pct": risk_per_trade * 100,
        "long_trades": int((trades["side"] == "LONG").sum()),
        "short_trades": int((trades["side"] == "SHORT").sum()),
        "avg_bars": float(np.mean(trades["bars"])),
        "max_concurrent": max_concurrent(trades),
    }


def max_concurrent(trades: pd.DataFrame) -> int:
    """回測期間最多同時有幾筆單持倉（不同幣常常一起出訊號）。"""
    events = sorted([(t, 1) for t in trades["time"]] + [(t, -1) for t in trades["exit_time"]],
                    key=lambda x: (x[0], x[1]))
    cur = best = 0
    for _, d in events:
        cur += d
        best = max(best, cur)
    return int(best)


def split_summary(trades: pd.DataFrame, risk_per_trade: float = 0.01) -> dict:
    """全期 / 前半（訓練）/ 後半（驗證）。"""
    if trades.empty:
        return {"all": {"trades": 0}, "first_half": {"trades": 0}, "second_half": {"trades": 0}}
    mid = trades["time"].min() + (trades["time"].max() - trades["time"].min()) / 2
    return {
        "all": summarize(trades, risk_per_trade),
        "first_half": summarize(trades[trades["time"] < mid], risk_per_trade),
        "second_half": summarize(trades[trades["time"] >= mid], risk_per_trade),
        "start": int(trades["time"].min().timestamp() * 1000),
        "end": int(trades["exit_time"].max().timestamp() * 1000),
        "symbols": int(trades["symbol"].nunique()),
    }

