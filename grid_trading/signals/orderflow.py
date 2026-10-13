"""訂單流策略（研究用，還沒接到網頁）。K 線要用 grid_trading.orderflow.flow_bars() 產生，帶主動買賣量和持倉量。

出場和 vegas_trend 一樣：停損 sl_atr 倍 ATR = 1R，TP 1R/2R/3R，交給 tracker。
每個策略在 prepare() 一次算出整段的訊號欄 sig（1 做多、-1 做空、0 沒訊號），check() 只讀這一欄。

- of_cvd_div：CVD 背離。價格創 lookback 根新高、CVD（累積主動買賣差）卻比前一個高點低 → 空；新低反之 → 多。
  mode="reversal" 直接反向做；mode="trend" 只做順著 EMA144/169 趨勢的那一邊（上升趨勢回檔時的底背離）。
- of_taker：主動買賣失衡。n 根的 (主動買額−主動賣額)/成交額，標準化後超過 k 個標準差時觸發。
  mode="follow" 跟著買方；"fade" 反向；"absorb" 買方很強但價格沒漲（被被動賣單吸收）→ 空，反之 → 多。
- of_oi：持倉量和價格。n 根的價格變化和持倉量變化都標準化：
  "build"：價漲倉增做多、價跌倉增做空（新資金進場，順勢）；"build_fade" 反過來。
  "unwind_fade"：價漲倉減（空單回補）做空、價跌倉減（多單爆倉/停損）做多；"unwind_follow" 反過來。
  "price_only"：對照組，只看價格的大漲大跌（順勢），用來確認持倉量有沒有多貢獻。
- of_crowd：帳戶多空比。field 是 account_ratio（全體帳戶）或 top_position_ratio（大戶持倉）；
  "fade"：多空比異常偏多 → 空、偏空 → 多；"follow" 反過來。

trend_filter=True 時，只做和 EMA144/169 趨勢同方向的單（df["trend"]）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..orderflow import add_features, prior_peak
from .strategy import LONG, SHORT, Strategy, register


class _OrderFlow(Strategy):
    chart_emas = {"EMA144": 144, "EMA169": 169}
    min_bars = 200

    def __init__(self, sl_atr: float = 1.5, tp_r=(1.0, 2.0, 3.0), trend_filter: bool = False):
        self.sl_atr = sl_atr
        self.tp_r = tuple(tp_r)
        self.trend_filter = trend_filter

    def raw_signal(self, df: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = add_features(df)
        sig = np.nan_to_num(self.raw_signal(df)).astype(int)
        if self.trend_filter:
            sig = np.where(sig == df["trend"].to_numpy(), sig, 0)
        df = df.copy()
        df["sig"] = sig
        return df

    def candidates(self, df: pd.DataFrame):
        return (df["sig"] != 0).to_numpy()

    def check(self, df: pd.DataFrame, i: int):
        s = df["sig"].iat[i]
        if s == 0:
            return None
        r = df.iloc[i]
        risk = self.sl_atr * r.atr
        if not risk > 0:
            return None
        info = {"atr": float(r.atr), "trend": int(r.trend)}
        if s > 0:
            return LONG, r.close - risk, [r.close + k * risk for k in self.tp_r], info
        return SHORT, r.close + risk, [r.close - k * risk for k in self.tp_r], info


def _first(cond: pd.Series) -> np.ndarray:
    """條件剛成立的那一根（前一根不成立），避免同一段行情連續觸發。"""
    cond = cond.fillna(False).astype(bool)
    return (cond & ~cond.shift(1, fill_value=False)).to_numpy()


@register
class CvdDivergence(_OrderFlow):
    name = "of_cvd_div"
    title = "CVD 背離"

    def __init__(self, lookback: int = 20, gap: int = 3, mode: str = "reversal", **kw):
        super().__init__(**kw)
        self.lookback, self.gap, self.mode = lookback, gap, mode

    def raw_signal(self, df):
        c = df["close"].to_numpy()
        cvd = df["cvd"].to_numpy()
        n = len(df)
        prev_max = df["close"].shift(1).rolling(self.lookback).max().to_numpy()
        prev_min = df["close"].shift(1).rolling(self.lookback).min().to_numpy()
        hi = prior_peak(c, self.lookback, self.gap, high=True)
        lo = prior_peak(c, self.lookback, self.gap, high=False)
        ok_hi, ok_lo = hi >= 0, lo >= 0
        hi_i, lo_i = np.where(ok_hi, hi, 0), np.where(ok_lo, lo, 0)
        bear = ok_hi & (c >= prev_max) & (c > c[hi_i]) & (cvd < cvd[hi_i])
        bull = ok_lo & (c <= prev_min) & (c < c[lo_i]) & (cvd > cvd[lo_i])
        sig = np.zeros(n, dtype=int)
        if self.mode == "reversal":
            sig[bull] = 1
            sig[bear] = -1
        else:  # trend：只做順勢那邊（上升趨勢裡的底背離、下降趨勢裡的頂背離）
            t = df["trend"].to_numpy()
            sig[bull & (t > 0)] = 1
            sig[bear & (t < 0)] = -1
        return sig


@register
class TakerImbalance(_OrderFlow):
    name = "of_taker"
    title = "主動買賣失衡"

    def __init__(self, n: int = 3, k: float = 2.0, mode: str = "follow", **kw):
        super().__init__(**kw)
        self.n, self.k, self.mode = n, k, mode

    def raw_signal(self, df):
        z = df[f"imb{self.n}_z"]
        ret = df[f"ret{self.n}"]
        if self.mode == "absorb":
            buy = _first((z > self.k) & (ret <= 0))
            sell = _first((z < -self.k) & (ret >= 0))
            return np.where(sell, 1, np.where(buy, -1, 0))
        up, down = _first(z > self.k), _first(z < -self.k)
        sig = np.where(up, 1, np.where(down, -1, 0))
        return sig if self.mode == "follow" else -sig


@register
class OpenInterestPrice(_OrderFlow):
    name = "of_oi"
    title = "持倉量 × 價格"

    def __init__(self, n: int = 3, k: float = 1.5, mode: str = "build", **kw):
        super().__init__(**kw)
        self.n, self.k, self.mode = n, k, mode

    def raw_signal(self, df):
        p, oi = df[f"ret{self.n}_z"], df[f"doi{self.n}_z"]
        if self.mode == "price_only":  # 對照組：只看價格（限有持倉量資料的期間），持倉量不管
            has = oi.notna()
            return np.where(_first(has & (p > self.k)), 1, np.where(_first(has & (p < -self.k)), -1, 0))
        if self.mode.startswith("build"):
            up = _first((p > self.k) & (oi > self.k))
            down = _first((p < -self.k) & (oi > self.k))
            sig = np.where(up, 1, np.where(down, -1, 0))
            return sig if self.mode == "build" else -sig
        up = _first((p > self.k) & (oi < -self.k))      # 空單回補
        down = _first((p < -self.k) & (oi < -self.k))   # 多單出場/爆倉
        sig = np.where(up, -1, np.where(down, 1, 0))
        return sig if self.mode == "unwind_fade" else -sig


@register
class Crowding(_OrderFlow):
    name = "of_crowd"
    title = "多空比擁擠"

    def __init__(self, field: str = "account_ratio", k: float = 2.0, mode: str = "fade", **kw):
        super().__init__(**kw)
        self.field, self.k, self.mode = field, k, mode

    def raw_signal(self, df):
        z = df[f"{self.field}_z"]
        hi, lo = _first(z > self.k), _first(z < -self.k)
        sig = np.where(hi, -1, np.where(lo, 1, 0))
        return sig if self.mode == "fade" else -sig
