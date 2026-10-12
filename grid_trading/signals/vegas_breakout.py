"""維加斯趨勢 + 突破：4H 用維加斯通道定方向，1H 突破前 20 根高（低）點時進場。

結合兩個策略：
- 維加斯趨勢（vegas_trend）的方向判斷：4H 收盤和 EMA12 都在 EMA144/169 通道上方且通道向上 → 只做多，
  反之只做空；4H MACD 柱同向而且在放大。
- 群組訊號回推的突破進場（grid_trading/breakout.py）：1H 收盤突破前 20 根高點、站上 EMA20/50/144/169、
  RSI 60~80（空單鏡像：跌破前 20 根低點、RSI 20~40）。
- 停損：最近一個已確認的 1H 波段低點（高點）外 0.3%，距離需在 3%~12%。TP 為 1R/2R/3R。

可選的額外條件（預設關閉，用來回測比較）：
- divergence：突破時如果出現頂背離（價格創波段新高、RSI 或 MACD 沒有創新高）就不進場，空單看底背離。
  "rsi"、"macd" 只看其中一個，"any" 兩個任一出現就略過。
- oil：參考油價（派網 WTI 永續合約，追蹤 NYMEX 輕原油）。"with" 只在油價 4H 趨勢和交易方向相同時進場，
  "against" 相反。油價趨勢 = 4H 收盤在 EMA50 之上（多）或之下（空）。WTI 合約 2026-03 才上市，之前沒有資料就不進場。
- ribbon：4H 方向改用完整的維加斯均線組（EMA12/144/169/288/338/576/676），收盤和 EMA12 要在全部均線之外。
- slow_ma：1H 收盤要在 N 期簡單均線（慢趨勢線，例如 200MA）同一側。
- hma：1H 收盤要在 N 期 Hull 均線（快趨勢線）同一側，而且 HMA 往交易方向走。
- psar：1H 拋物線轉向指標（0.02/0.02/0.2）要在價格下方（多）或上方（空）。

用 1H K 線跑，4H 指標由 1H 合成，只用已收盤的 4H K 線。
"""

from __future__ import annotations

import pandas as pd

from .. import indicators as ind
from ..history import fetch_history, resample
from .strategy import LONG, SHORT, Strategy, register


@register
class VegasBreakout(Strategy):
    name = "vegas_breakout"
    title = "維加斯趨勢 + 1H 突破"
    description = ("4H 維加斯通道定方向（通道上方且向上只做多，下方且向下只做空），4H MACD 柱同向放大；"
                   "1H 突破前 20 根高/低點時進場，停損放在最近的 1H 波段低/高點外。TP 為 1R/2R/3R。")
    chart_emas = {"EMA20": 20, "EMA144": 144, "EMA169": 169}
    min_bars = 200

    def __init__(self, trend: str = "vegas", allow_short: bool = True, lookback: int = 20,
                 rsi_long: tuple[float, float] = (60, 80), rsi_short: tuple[float, float] = (20, 40),
                 pivot_k: int = 3, stop_buffer: float = 0.003, min_risk: float = 0.03, max_risk: float = 0.12,
                 tp_r: tuple[float, float, float] = (1.0, 2.0, 3.0), divergence: str = "none",
                 oil: str = "none", oil_symbol: str = "WTI_USDT_PERP", ribbon: bool = False,
                 slow_ma: int = 0, hma: int = 0, psar: bool = False):
        # trend="vegas"：照 vegas_trend 的定義（收盤、EMA12 都在通道外，且通道方向一致）
        # trend="basic"：只看 4H 收盤在通道上方/下方（群組訊號回推時用的寬鬆版）
        self.trend = trend
        self.allow_short = allow_short
        self.lookback = lookback
        self.rsi_long = rsi_long
        self.rsi_short = rsi_short
        self.pivot_k = pivot_k
        self.stop_buffer = stop_buffer
        self.min_risk = min_risk
        self.max_risk = max_risk
        self.tp_r = tuple(tp_r)
        self.divergence = divergence
        self.oil = oil
        self.oil_symbol = oil_symbol
        self.ribbon = ribbon
        self.slow_ma = slow_ma
        self.hma = hma
        self.psar = psar

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        c, low, high = df["close"], df["low"], df["high"]
        for n in (20, 50, 144, 169):
            df[f"ema{n}"] = ind.ema(c, n)
        df["rsi"] = ind.rsi(c)
        df["prior_high"] = high.shift(1).rolling(self.lookback).max()
        df["prior_low"] = low.shift(1).rolling(self.lookback).min()
        k = self.pivot_k
        # 波段高低點要等右邊 k 根收完才確認，所以往後移 k 根
        df["swing_low"] = low.where(low == low.rolling(2 * k + 1, center=True).min()).ffill().shift(k)
        df["swing_high"] = high.where(high == high.rolling(2 * k + 1, center=True).max()).ffill().shift(k)
        # 背離用：上一個已確認波段高（低）點的價格、RSI、MACD 線
        macd_line = ind.macd(c)["macd"]
        df["macd_line"] = macd_line
        rsi_ = df["rsi"]
        ph = high == high.rolling(2 * k + 1, center=True).max()
        pl = low == low.rolling(2 * k + 1, center=True).min()
        for name, mask, src in (("ph", ph, high), ("pl", pl, low)):
            df[f"{name}_price"] = src.where(mask).ffill().shift(k)
            df[f"{name}_rsi"] = rsi_.where(mask).ffill().shift(k)
            df[f"{name}_macd"] = macd_line.where(mask).ffill().shift(k)
        if self.slow_ma:
            df["slow_ma"] = c.rolling(self.slow_ma).mean()
        if self.hma:
            df["hma"] = ind.hma(c, self.hma)
            df["hma_prev"] = df["hma"].shift(1)
        if self.psar:
            df["psar"] = ind.psar(df)
        df["low24"] = low.rolling(24).min()
        df["high24"] = high.rolling(24).max()

        h4 = resample(df[["time", "open", "high", "low", "close", "volume"]], "4h")
        hc = h4["close"]
        e12, e144, e169 = ind.ema(hc, 12), ind.ema(hc, 144), ind.ema(hc, 169)
        top, bottom = pd.concat([e144, e169], axis=1).max(axis=1), pd.concat([e144, e169], axis=1).min(axis=1)
        hist = ind.macd(hc)["hist"]
        if self.ribbon:
            band = pd.concat([ind.ema(hc, n) for n in (144, 169, 288, 338, 576, 676)], axis=1)
            hi, lo = band.max(axis=1), band.min(axis=1)
            up = (hc > hi) & (e12 > hi)
            down = (hc < lo) & (e12 < lo)
        elif self.trend == "vegas":
            up = (hc > top) & (e12 > top) & (e144 > e169)
            down = (hc < bottom) & (e12 < bottom) & (e144 < e169)
        else:
            up, down = hc > top, hc < bottom
        h4f = pd.DataFrame({
            "avail": h4["time"] + pd.Timedelta(hours=4),  # 這根 4H 收盤後才能用
            "h4_up": up & (hist > 0) & (hist > hist.shift(1)),
            "h4_down": down & (hist < 0) & (hist < hist.shift(1)),
        })
        df["bar_close"] = df["time"] + pd.Timedelta(hours=1)
        df = pd.merge_asof(df, h4f, left_on="bar_close", right_on="avail", direction="backward")
        df[["h4_up", "h4_down"]] = df[["h4_up", "h4_down"]].fillna(False).astype(bool)
        df = df.drop(columns=["avail"])
        if self.oil != "none":
            df = self._add_oil(df)
        return df.drop(columns=["bar_close"])

    def _add_oil(self, df: pd.DataFrame) -> pd.DataFrame:
        days = int((pd.Timestamp.now(tz="UTC") - df["time"].iloc[0]).days) + 2
        oil = fetch_history(self.oil_symbol, "60M", min(days, 420))
        df["oil_up"] = False
        df["oil_down"] = False
        if oil.empty:
            return df
        o4 = resample(oil, "4h")
        e50 = ind.ema(o4["close"], 50)
        of = pd.DataFrame({"avail": o4["time"] + pd.Timedelta(hours=4),
                           "o_up": (o4["close"] > e50).values, "o_down": (o4["close"] < e50).values})
        of = of.iloc[50:]  # EMA 需要暖機
        df = pd.merge_asof(df.drop(columns=["oil_up", "oil_down"]), of, left_on="bar_close", right_on="avail",
                           direction="backward", tolerance=pd.Timedelta(days=4))  # 油價休市時沿用最後狀態
        df["oil_up"] = df["o_up"].fillna(False).astype(bool)
        df["oil_down"] = df["o_down"].fillna(False).astype(bool)
        return df.drop(columns=["avail", "o_up", "o_down"])

    def _diverges(self, r: pd.Series, side: str) -> bool:
        if self.divergence == "none":
            return False
        if side == LONG:  # 頂背離：價格高過上一個波段高點，指標卻比較低
            new_extreme = r.close > r.ph_price
            rsi_div = new_extreme and r.rsi < r.ph_rsi
            macd_div = new_extreme and r.macd_line < r.ph_macd
        else:             # 底背離：價格低過上一個波段低點，指標卻比較高
            new_extreme = r.close < r.pl_price
            rsi_div = new_extreme and r.rsi > r.pl_rsi
            macd_div = new_extreme and r.macd_line > r.pl_macd
        return {"rsi": rsi_div, "macd": macd_div, "any": rsi_div or macd_div}[self.divergence]

    def _lines_ok(self, r: pd.Series, side: str) -> bool:
        d = 1 if side == LONG else -1
        if self.slow_ma and not d * (r.close - r.slow_ma) > 0:
            return False
        if self.hma and not (d * (r.close - r.hma) > 0 and d * (r.hma - r.hma_prev) > 0):
            return False
        if self.psar and not d * (r.close - r.psar) > 0:
            return False
        return True

    def _oil_ok(self, r: pd.Series, side: str) -> bool:
        if self.oil == "none":
            return True
        same = r.oil_up if side == LONG else r.oil_down
        opposite = r.oil_down if side == LONG else r.oil_up
        return bool(same if self.oil == "with" else opposite)

    def check(self, df: pd.DataFrame, i: int):
        close = df["close"].iat[i]
        long_ok = df["h4_up"].iat[i] and close > df["prior_high"].iat[i]
        short_ok = self.allow_short and df["h4_down"].iat[i] and close < df["prior_low"].iat[i]
        if not (long_ok or short_ok):
            return None  # 先用最便宜的條件過濾，回測才不會慢
        r = df.iloc[i]
        info = {"rsi": round(float(r.rsi), 2), "ema144": float(r.ema144), "ema169": float(r.ema169),
                "candle": {"open": float(r.open), "high": float(r.high), "low": float(r.low),
                           "close": float(r.close), "time": int(r.time.timestamp() * 1000)}}
        emas = [r[f"ema{n}"] for n in (20, 50, 144, 169)]
        if (long_ok and all(close > e for e in emas) and self.rsi_long[0] <= r.rsi <= self.rsi_long[1]
                and not self._diverges(r, LONG) and self._oil_ok(r, LONG) and self._lines_ok(r, LONG)):
            base = r.swing_low if r.swing_low < close else r.low24
            stop = base * (1 - self.stop_buffer)
            risk = close - stop
            if self.min_risk <= risk / close <= self.max_risk:
                info["reason"] = "4H 維加斯通道向上、MACD 柱放大；1H 突破前 20 根高點；停損在 1H 波段低點下方"
                return LONG, stop, [close + k * risk for k in self.tp_r], info
        if (short_ok and all(close < e for e in emas) and self.rsi_short[0] <= r.rsi <= self.rsi_short[1]
                and not self._diverges(r, SHORT) and self._oil_ok(r, SHORT) and self._lines_ok(r, SHORT)):
            base = r.swing_high if r.swing_high > close else r.high24
            stop = base * (1 + self.stop_buffer)
            risk = stop - close
            if self.min_risk <= risk / close <= self.max_risk:
                info["reason"] = "4H 維加斯通道向下、MACD 柱放大；1H 跌破前 20 根低點；停損在 1H 波段高點上方"
                return SHORT, stop, [close - k * risk for k in self.tp_r], info
        return None
