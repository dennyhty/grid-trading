"""強勢突破追多（突破觀察組，1H，只做多加密貨幣）。

由外部訊號群的突破追多回推（grid_trading/breakout.py），再用 breakout_research.py 測過濾條件
（reports/report_breakout_filters_20261014.md）：只做「夠強」的突破才有優勢。
- 1H 收盤突破前 20 根 1H 的最高價，而且站上 1H EMA20/50/144/169；
- 最近一根已收盤的 4H（由 1H 合成）收在維加斯通道（EMA144/169）上方；
- 1H RSI(14) ≥ 75（外部訊號群用 60~80，剛好切掉最好的一段）；
- 這根 1H 的成交量 ≥ 前 20 根平均的 2 倍。
- 停損：最近一個已確認的 1H 波段低點（左右各 3 根都比它高）下方 0.3%；波段低點已經在價格上方時改用近 24 根最低點。
  停損距離要在 2%~15% 之間。
- 出場：TP1 = 1R 平一半、止損移到保本；TP2 = 2R 平剩下（和研究、外部訊號群相同）。

幣安 2020-01~2026-09、83 個幣：2020-23 每筆 +0.127R（勝率 56%），2024 年後 +0.105R（54.5%）。
但同時最多 5 張時 2024 年後只剩約 +0.06R，2022 熊市是虧的，所以只當觀察組（網站顯示、不推播）。

即時掃描時 1H 只有 1000 根（約 250 根 4H），4H 的 EMA169 和用完整歷史算的差一點：
抽樣 1500 個時點有 1.5% 的「是否在通道上方」判斷不同，都是價格貼著通道的時候。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import indicators as ind
from ..history import resample
from .strategy import LONG, Strategy, register


@register
class BreakoutMomentum(Strategy):
    name = "breakout_momentum"
    title = "強勢突破追多（突破觀察組）"
    description = ("1H 收盤突破前 20 根高點、站上 1H EMA20/50/144/169，4H 在維加斯通道上方；"
                   "突破那根 RSI ≥ 75、成交量 ≥ 前 20 根平均 2 倍才做（只做多）。"
                   "停損在最近的 1H 波段低點下方 0.3%；TP1 = 1R 平一半、止損移到保本，TP2 = 2R 平剩下。")
    tag = "突破觀察組"
    intervals = ("60M",)
    markets = ("crypto",)
    chart_emas = {"EMA20": 20, "EMA144": 144, "EMA169": 169}
    min_bars = 200

    def __init__(self, lookback: int = 20, rsi_min: float = 75.0, vol_mult: float = 2.0, pivot_k: int = 3,
                 stop_buffer: float = 0.003, min_risk: float = 0.02, max_risk: float = 0.15,
                 tp_r: tuple[float, float] = (1.0, 2.0)):
        self.lookback = lookback
        self.rsi_min = rsi_min
        self.vol_mult = vol_mult
        self.pivot_k = pivot_k
        self.stop_buffer = stop_buffer
        self.min_risk = min_risk
        self.max_risk = max_risk
        self.tp_r = tuple(tp_r)

    def exit_plan(self, interval: str) -> dict | None:
        return {"tp_r": list(self.tp_r), "portions": [0.5, 0.5]}

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        c, high, low, vol = df["close"], df["high"], df["low"], df["volume"]
        for n in (20, 50, 144, 169):
            df[f"ema{n}"] = ind.ema(c, n)
        df["rsi"] = ind.rsi(c)
        df["atr"] = ind.atr(df)
        df["prior_high"] = high.shift(1).rolling(self.lookback).max()
        df["vol_ratio"] = vol / vol.shift(1).rolling(20).mean()
        k = self.pivot_k
        # 波段低點要等右邊 k 根收完才確認，所以往後移 k 根
        df["swing_low"] = low.where(low == low.rolling(2 * k + 1, center=True).min()).ffill().shift(k)
        df["low24"] = low.rolling(24).min()
        # 4H 維加斯通道：只用已收盤的 4H（4H K 線收盤時間 <= 這根 1H 的收盤時間）
        h4 = resample(df[["time", "open", "high", "low", "close", "volume"]], "4h", origin="epoch")
        top = np.maximum(ind.ema(h4["close"], 144), ind.ema(h4["close"], 169))
        h = pd.DataFrame({"avail": h4["time"] + pd.Timedelta(hours=4), "h4_close": h4["close"], "h4_top": top})
        bar_close = df["time"] + pd.Timedelta(hours=1)
        m = pd.merge_asof(pd.DataFrame({"bar_close": bar_close}), h.sort_values("avail"),
                          left_on="bar_close", right_on="avail", direction="backward")
        df["h4_close"] = m["h4_close"].to_numpy()
        df["h4_top"] = m["h4_top"].to_numpy()
        return df

    def candidates(self, df: pd.DataFrame):
        c = df["close"]
        return ((c > df["prior_high"]) & (df["rsi"] >= self.rsi_min) & (df["vol_ratio"] >= self.vol_mult)
                & (df["h4_close"] > df["h4_top"])).to_numpy()

    def check(self, df: pd.DataFrame, i: int):
        r = df.iloc[i]
        c = float(r.close)
        if not (c > r.prior_high and r.rsi >= self.rsi_min and r.vol_ratio >= self.vol_mult):
            return None
        if not all(c > r[f"ema{n}"] for n in (20, 50, 144, 169)):
            return None
        if not (pd.notna(r.h4_top) and r.h4_close > r.h4_top):
            return None
        swing = r.swing_low if pd.notna(r.swing_low) and r.swing_low < c else r.low24
        if not pd.notna(swing):
            return None
        stop = float(swing) * (1 - self.stop_buffer)
        risk = c - stop
        if not self.min_risk <= risk / c <= self.max_risk:
            return None
        base = "1H 波段低點" if swing == r.swing_low else "近 24 根 1H 最低點"
        info = {
            "rsi": round(float(r.rsi), 2), "atr": float(r.atr), "ema144": float(r.ema144), "ema169": float(r.ema169),
            "reason": "1H 帶量突破前 20 根高點、RSI ≥ 75，4H 在維加斯通道上方",
            "details": [
                ["1H RSI", f"{r.rsi:.1f}"],
                ["成交量", f"前 20 根平均的 {r.vol_ratio:.1f} 倍"],
                ["突破的前高", f"{r.prior_high:.6g}"],
                ["4H 收盤 / 通道上緣", f"{r.h4_close:.6g} / {r.h4_top:.6g}"],
                ["止損依據", f"{base} {float(swing):.6g} 下方 0.3%（距離 {risk / c * 100:.1f}%）"],
            ],
            "candle": {"open": float(r.open), "high": float(r.high), "low": float(r.low), "close": c,
                       "time": int(r.time.timestamp() * 1000)},
        }
        return LONG, stop, [c + k * risk for k in self.tp_r], info
