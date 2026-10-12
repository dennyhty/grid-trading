"""合約網格模擬器（等差網格）。

模擬派網合約網格的三種模式：
  做多：開單時把現價上方的格子先買進，價格下跌逐格買、上漲逐格賣。
  做空：開單時把現價下方的格子先放空，價格上漲逐格空、下跌逐格回補。
  中性：開單時不持倉，現價下方的格子做多、上方的格子做空。
觸及停損價就全部市價平倉並結束。

K 線內部路徑假設：陽線 開→低→高→收，陰線 開→高→低→收。
報酬以「投入資金、1 倍槓桿」計；槓桿 L 倍時報酬與回撤約乘以 L。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

LONG, SHORT, NEUTRAL = "做多", "做空", "中性"


@dataclass
class GridResult:
    ret: float            # 期末報酬（佔投入資金比例）
    grid_profit: float    # 已實現的網格套利收益（不含手續費）
    fees: float
    max_drawdown: float   # 期間最大回撤（佔投入資金比例，正數）
    trades: int           # 完成的一買一賣次數
    stopped: bool
    stop_time: pd.Timestamp | None


def simulate(
    bars: pd.DataFrame,
    lower: float,
    upper: float,
    grids: int,
    mode: str,
    stop_low: float | None = None,
    stop_high: float | None = None,
    fee: float = 0.0005,
) -> GridResult:
    levels = np.linspace(lower, upper, grids + 1)
    buy_px, sell_px = levels[:-1], levels[1:]
    n = grids
    # 每格投入相同資金（1/n），以該格的買價換算數量
    qty = (1.0 / n) / buy_px
    p0 = float(bars["open"].iloc[0])

    # 格子類型：+1 做多格（低買高賣）、-1 做空格（高空低補）
    if mode == LONG:
        kind = np.ones(n)
    elif mode == SHORT:
        kind = -np.ones(n)
    else:
        kind = np.where(sell_px <= p0, 1.0, np.where(buy_px >= p0, -1.0, 0.0))  # 跨現價那格不用

    pos = np.zeros(n)      # 持倉方向 +1/-1/0
    entry = np.zeros(n)
    realized = 0.0
    grid_profit = 0.0
    fees = 0.0
    trades = 0

    # 開單時的底倉
    if mode == LONG:
        init = buy_px >= p0
    elif mode == SHORT:
        init = sell_px <= p0
    else:
        init = np.zeros(n, dtype=bool)
    pos[init] = kind[init]
    entry[init] = p0
    fees += float((qty[init] * p0).sum() * fee)

    def unrealized(price: float) -> float:
        return float((pos * (price - entry) * qty).sum())

    peak, max_dd = 0.0, 0.0
    stopped, stop_time = False, None

    for t, o, h, l, c in zip(bars["time"], bars["open"], bars["high"], bars["low"], bars["close"]):
        path = (o, l, h, c) if c >= o else (o, h, l, c)
        for a, b in zip(path[:-1], path[1:]):
            if b < a:  # 下跌段
                floor = b if stop_low is None else max(b, stop_low)
                # 做多格在買價開倉
                hit = (kind == 1) & (pos == 0) & (buy_px < a) & (buy_px >= floor)
                pos[hit], entry[hit] = 1, buy_px[hit]
                fees += float((qty[hit] * buy_px[hit]).sum() * fee)
                # 做空格在買價回補
                hit = (kind == -1) & (pos == -1) & (buy_px < a) & (buy_px >= floor)
                gp = float(((entry[hit] - buy_px[hit]) * qty[hit]).sum())
                realized += gp
                grid_profit += gp
                fees += float((qty[hit] * buy_px[hit]).sum() * fee)
                trades += int(hit.sum())
                pos[hit] = 0
                if stop_low is not None and b <= stop_low:
                    realized, fees = _close_all(pos, entry, qty, stop_low, realized, fees, fee)
                    stopped, stop_time = True, t
            elif b > a:  # 上漲段
                ceil = b if stop_high is None else min(b, stop_high)
                # 做多格在賣價平倉
                hit = (kind == 1) & (pos == 1) & (sell_px > a) & (sell_px <= ceil)
                gp = float(((sell_px[hit] - entry[hit]) * qty[hit]).sum())
                realized += gp
                grid_profit += gp
                fees += float((qty[hit] * sell_px[hit]).sum() * fee)
                trades += int(hit.sum())
                pos[hit] = 0
                # 做空格在賣價開倉
                hit = (kind == -1) & (pos == 0) & (sell_px > a) & (sell_px <= ceil)
                pos[hit], entry[hit] = -1, sell_px[hit]
                fees += float((qty[hit] * sell_px[hit]).sum() * fee)
                if stop_high is not None and b >= stop_high:
                    realized, fees = _close_all(pos, entry, qty, stop_high, realized, fees, fee)
                    stopped, stop_time = True, t
            eq = realized - fees + (0.0 if stopped else unrealized(b))
            peak = max(peak, eq)
            max_dd = max(max_dd, peak - eq)
            if stopped:
                break
        if stopped:
            break

    if not stopped:
        last = float(bars["close"].iloc[-1])
        realized, fees = _close_all(pos, entry, qty, last, realized, fees, fee)

    return GridResult(realized - fees, grid_profit, fees, max_dd, trades, stopped, stop_time)


def _close_all(pos, entry, qty, price, realized, fees, fee):
    open_ = pos != 0
    realized += float((pos[open_] * (price - entry[open_]) * qty[open_]).sum())
    fees += float((qty[open_] * price).sum() * fee)
    pos[:] = 0
    return realized, fees
