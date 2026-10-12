"""第一個策略：維加斯通道定方向，MACD 翻轉 + RSI 確認進場。

- 趨勢：收盤和 EMA12 都在通道（EMA144/169）上方，且 EMA144 > EMA169 → 只找多單；反之只找空單。
- 進場：MACD 柱狀體由負翻正（多）或由正翻負（空），RSI 在 50~70（多）或 30~50（空），
  而且價格離通道不超過 max_ext_atr 倍 ATR（追太高不進）。
- 停損：進場價 ∓ sl_atr 倍 ATR，風險記為 1R。
- 停利：TP1/TP2/TP3 = 1R / 2R / 3R（倍數可調）。
"""

from __future__ import annotations

import pandas as pd

from .. import indicators as ind
from .strategy import LONG, SHORT, Strategy, register


@register
class VegasTrend(Strategy):
    name = "vegas_trend"
    title = "維加斯通道順勢 + MACD/RSI"
    description = ("價格在 EMA144/169 通道上方且通道向上時只做多、下方向下時只做空；"
                   "MACD 柱翻轉且 RSI 同向時進場。停損 ATR 倍數，TP 為 1R/2R/3R。")
    chart_emas = {"EMA12": 12, "EMA144": 144, "EMA169": 169, "EMA576": 576, "EMA676": 676}
    min_bars = 200

    def __init__(self, sl_atr: float = 1.5, tp_r: tuple[float, float, float] = (1.0, 2.0, 3.0),
                 max_ext_atr: float = 3.0, rsi_long: tuple[float, float] = (50, 70),
                 rsi_short: tuple[float, float] = (30, 50)):
        self.sl_atr = sl_atr
        self.tp_r = tuple(tp_r)
        self.max_ext_atr = max_ext_atr
        self.rsi_long = rsi_long
        self.rsi_short = rsi_short

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        c = df["close"]
        df["ema12"] = ind.ema(c, 12)
        df["ema144"] = ind.ema(c, 144)
        df["ema169"] = ind.ema(c, 169)
        df["rsi"] = ind.rsi(c)
        m = ind.macd(c)
        df["macd"], df["macd_signal"], df["hist"] = m["macd"], m["signal"], m["hist"]
        df["atr"] = ind.atr(df)
        return df

    def candidates(self, df: pd.DataFrame):
        # 和 check() 開頭的條件相同：MACD 柱由負翻正或由正翻負
        h = df["hist"]
        prev = h.shift(1)
        return (((prev <= 0) & (h > 0)) | ((prev >= 0) & (h < 0))).to_numpy()

    def check(self, df: pd.DataFrame, i: int):
        h = df["hist"]
        prev_hist, hist = h.iat[i - 1], h.iat[i]
        if not ((prev_hist <= 0 < hist) or (prev_hist >= 0 > hist)):
            return None  # 先用最便宜的條件過濾，回測才不會慢
        r = df.iloc[i]
        top = max(r.ema144, r.ema169)
        bottom = min(r.ema144, r.ema169)
        info = {
            "rsi": round(float(r.rsi), 2),
            "macd": float(r.macd), "macd_signal": float(r.macd_signal), "hist": float(hist),
            "atr": float(r.atr), "ema144": float(r.ema144), "ema169": float(r.ema169),
            "candle": {"open": float(r.open), "high": float(r.high), "low": float(r.low),
                       "close": float(r.close), "time": int(r.time.timestamp() * 1000)},
        }
        risk = self.sl_atr * r.atr
        if risk <= 0:
            return None

        up = r.close > top and r.ema12 > top and r.ema144 > r.ema169
        if (up and prev_hist <= 0 < hist and self.rsi_long[0] <= r.rsi <= self.rsi_long[1]
                and r.close - top <= self.max_ext_atr * r.atr):
            info["reason"] = "價格在維加斯通道上方、通道向上；MACD 柱翻正；RSI 偏多未過熱"
            return LONG, r.close - risk, [r.close + k * risk for k in self.tp_r], info

        down = r.close < bottom and r.ema12 < bottom and r.ema144 < r.ema169
        if (down and prev_hist >= 0 > hist and self.rsi_short[0] <= r.rsi <= self.rsi_short[1]
                and bottom - r.close <= self.max_ext_atr * r.atr):
            info["reason"] = "價格在維加斯通道下方、通道向下；MACD 柱翻負；RSI 偏空未超賣"
            return SHORT, r.close + risk, [r.close - k * risk for k in self.tp_r], info
        return None
