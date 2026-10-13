"""基礎技術指標（研究用）：均線、布林帶、KD、威廉、CCI、MFI、OBV、DMI、超級趨勢、唐奇安、肯特納、一目均衡表、SMI、轉折點。

輸入都是依時間由舊到新排序的 pandas Series / DataFrame，第 i 根的值只用到第 i 根（含）以前的資料，不會偷看未來。
參數預設值用各指標最常見的寫法（TradingView、幣安/派網 App 的預設），研究時不另外調參數：

- 布林帶 20, 2（標準差用母體標準差，和 TradingView 相同）
- KD 9, 3, 3（台灣/幣安 App 的 KDJ 寫法：K = 2/3 前K + 1/3 RSV，D = 2/3 前D + 1/3 K）
- 威廉指標 14、CCI 20、MFI 14、DMI 14、超級趨勢 10, 3、肯特納 EMA20 ± 2 × ATR10、一目均衡表 9/26/52、SMI 10/3/3

已經在 indicators.py 的（EMA、ATR、ADX、RSI、MACD、HMA、PSAR）直接用那邊的。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import indicators as ind


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n).mean()


def bollinger(close: pd.Series, n: int = 20, k: float = 2.0) -> pd.DataFrame:
    """回傳 mid（SMA）、upper、lower、width（帶寬 = (上軌 - 下軌) / 中線）。"""
    mid = close.rolling(n).mean()
    sd = close.rolling(n).std(ddof=0)
    up, lo = mid + k * sd, mid - k * sd
    return pd.DataFrame({"mid": mid, "upper": up, "lower": lo, "width": (up - lo) / mid})


def kd(df: pd.DataFrame, n: int = 9, k_smooth: int = 3, d_smooth: int = 3) -> pd.DataFrame:
    """KD 隨機指標（台灣/幣安 KDJ 寫法）。RSV = (收盤 - n 根最低) / (n 根最高 - n 根最低) × 100，
    K = (1 - 1/k_smooth) × 前 K + RSV / k_smooth，D 同理對 K 平滑；起始值 50。回傳 rsv、k、d、j。"""
    hh = df["high"].rolling(n).max()
    ll = df["low"].rolling(n).min()
    rng = (hh - ll).to_numpy(float)
    rsv = np.where(rng > 0, (df["close"].to_numpy(float) - ll.to_numpy(float)) / np.where(rng > 0, rng, 1) * 100, 50.0)
    rsv = np.where(np.isfinite(rng), rsv, np.nan)
    k = np.full(len(df), np.nan)
    d = np.full(len(df), np.nan)
    pk, pd_ = 50.0, 50.0
    for i, x in enumerate(rsv):
        if np.isnan(x):
            continue
        pk = pk + (x - pk) / k_smooth
        pd_ = pd_ + (pk - pd_) / d_smooth
        k[i], d[i] = pk, pd_
    return pd.DataFrame({"rsv": rsv, "k": k, "d": d, "j": 3 * k - 2 * d}, index=df.index)


def williams_r(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """威廉指標 %R：-100 ~ 0，-80 以下超賣、-20 以上超買。"""
    hh = df["high"].rolling(n).max()
    ll = df["low"].rolling(n).min()
    return -100 * (hh - df["close"]) / (hh - ll).replace(0, np.nan)


def cci(df: pd.DataFrame, n: int = 20) -> pd.Series:
    """CCI = (典型價 - n 期均值) / (0.015 × 平均絕對偏差)。"""
    tp = (df["high"] + df["low"] + df["close"]) / 3
    ma = tp.rolling(n).mean()
    x = tp.to_numpy(float)
    mad = np.full(len(x), np.nan)
    if len(x) >= n:
        w = np.lib.stride_tricks.sliding_window_view(x, n)
        mad[n - 1:] = np.abs(w - w.mean(axis=1, keepdims=True)).mean(axis=1)
    return (tp - ma) / (0.015 * pd.Series(mad, index=tp.index).replace(0, np.nan))


def mfi(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """資金流量指標 MFI（加了成交量的 RSI）：典型價上漲那根的「價 × 量」算流入，下跌算流出。"""
    tp = (df["high"] + df["low"] + df["close"]) / 3
    flow = tp * df["volume"]
    up = tp.diff()
    pos = flow.where(up > 0, 0.0).rolling(n).sum()
    neg = flow.where(up < 0, 0.0).rolling(n).sum()
    out = 100 - 100 / (1 + pos / neg.replace(0, np.nan))
    return out.where(neg > 0, 100.0).where(pos.notna())


def obv(df: pd.DataFrame) -> pd.Series:
    """能量潮 OBV：收盤上漲加上當根成交量，下跌減掉。"""
    sign = np.sign(df["close"].diff()).fillna(0)
    return (sign * df["volume"]).cumsum()


def dmi(df: pd.DataFrame, n: int = 14) -> pd.DataFrame:
    """Wilder DMI：回傳 plus_di、minus_di、adx（adx 和 indicators.adx 相同）。"""
    up = df["high"].diff()
    down = -df["low"].diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)
    tr = ind.atr(df, n)
    plus_di = 100 * plus_dm.ewm(alpha=1 / n, adjust=False).mean() / tr
    minus_di = 100 * minus_dm.ewm(alpha=1 / n, adjust=False).mean() / tr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return pd.DataFrame({"plus_di": plus_di, "minus_di": minus_di, "adx": dx.ewm(alpha=1 / n, adjust=False).mean()})


def supertrend(df: pd.DataFrame, n: int = 10, mult: float = 3.0) -> pd.Series:
    """超級趨勢（TradingView 寫法，ATR 用 Wilder）：回傳方向 +1（多）/ -1（空）。"""
    atr = ind.atr(df, n).to_numpy(float)
    hl2 = ((df["high"] + df["low"]) / 2).to_numpy(float)
    close = df["close"].to_numpy(float)
    up_basic = hl2 - mult * atr
    dn_basic = hl2 + mult * atr
    out = np.zeros(len(df))
    up = dn = np.nan
    trend = 1
    for i in range(len(df)):
        if not np.isfinite(atr[i]):
            continue
        if np.isnan(up):
            up, dn = up_basic[i], dn_basic[i]
        else:
            up = max(up_basic[i], up) if close[i - 1] > up else up_basic[i]
            dn = min(dn_basic[i], dn) if close[i - 1] < dn else dn_basic[i]
        if trend == -1 and close[i] > dn:
            trend = 1
        elif trend == 1 and close[i] < up:
            trend = -1
        out[i] = trend
    return pd.Series(out, index=df.index)


def donchian(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """前 n 根（不含這一根）的最高價 / 最低價，收盤超過就是突破。"""
    return pd.DataFrame({"high": df["high"].rolling(n).max().shift(1), "low": df["low"].rolling(n).min().shift(1)})


def keltner(df: pd.DataFrame, n: int = 20, mult: float = 2.0, atr_n: int = 10) -> pd.DataFrame:
    mid = ind.ema(df["close"], n)
    a = ind.atr(df, atr_n)
    return pd.DataFrame({"mid": mid, "upper": mid + mult * a, "lower": mid - mult * a})


def ichimoku(df: pd.DataFrame, tenkan: int = 9, kijun: int = 26, senkou: int = 52) -> pd.DataFrame:
    """一目均衡表。span_a / span_b 是畫在第 i 根位置的雲帶（也就是 kijun 根以前算出來的值），沒有偷看。"""
    def mid(n):
        return (df["high"].rolling(n).max() + df["low"].rolling(n).min()) / 2
    t, k = mid(tenkan), mid(kijun)
    return pd.DataFrame({"tenkan": t, "kijun": k, "span_a": ((t + k) / 2).shift(kijun),
                         "span_b": mid(senkou).shift(kijun)})


def smi(df: pd.DataFrame, n: int = 10, smooth1: int = 3, smooth2: int = 3, signal: int = 3) -> pd.DataFrame:
    """Blau 隨機動量指標 SMI：收盤離 n 根高低點中線多遠（雙重 EMA 平滑），-100 ~ 100；回傳 smi、signal。"""
    hh = df["high"].rolling(n).max()
    ll = df["low"].rolling(n).min()
    rel = df["close"] - (hh + ll) / 2
    rng = hh - ll
    num = ind.ema(ind.ema(rel, smooth1), smooth2)
    den = ind.ema(ind.ema(rng, smooth1), smooth2) / 2
    s = 100 * num / den.replace(0, np.nan)
    return pd.DataFrame({"smi": s, "signal": ind.ema(s, signal)})


def pivots(values: np.ndarray, k: int, kind: str) -> np.ndarray:
    """轉折點：左右各 k 根都比它低（高點）或都比它高（低點）。回傳布林陣列，標在轉折點那一根；
    要到第 p + k 根收盤才知道 p 是轉折點，使用時要往後移 k 根。"""
    values = np.asarray(values, float)
    n = len(values)
    out = np.zeros(n, dtype=bool)
    if n < 2 * k + 1:
        return out
    w = np.lib.stride_tricks.sliding_window_view(values, 2 * k + 1)  # 第 p-k 列的中心是第 p 根
    c = w[:, k:k + 1]
    if kind == "high":
        ok = (w[:, :k] < c).all(axis=1) & (w[:, k + 1:] <= c).all(axis=1)
    else:
        ok = (w[:, :k] > c).all(axis=1) & (w[:, k + 1:] >= c).all(axis=1)
    out[k:n - k] = ok & np.isfinite(w).all(axis=1)
    return out


def divergence(price_hi: np.ndarray, price_lo: np.ndarray, osc: np.ndarray, k: int = 3,
               min_gap: int = 5, max_gap: int = 60, osc_lo: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """背離訊號（在轉折點確認的那根 p + k 收盤時才成立）。

    底背離（多）：最新的低點比前一個低點更低（相隔 min_gap~max_gap 根），指標在這兩點卻是墊高。
    頂背離（空）：最新的高點比前一個高點更高，指標卻是降低。osc_lo 給的話，底背離用 osc_lo 比較
    （例如量價背離：頂背離比成交量、底背離也比成交量）。回傳 (bull, bear) 布林陣列。
    """
    n = len(price_hi)
    lo_osc = osc if osc_lo is None else osc_lo
    bull = np.zeros(n, dtype=bool)
    bear = np.zeros(n, dtype=bool)
    ph = np.flatnonzero(pivots(price_hi, k, "high"))
    pl = np.flatnonzero(pivots(price_lo, k, "low"))
    for arr, piv, out, sign in ((price_hi, ph, bear, 1), (price_lo, pl, bull, -1)):
        o = osc if sign > 0 else lo_osc
        for a, b in zip(piv[:-1], piv[1:]):
            gap = b - a
            if gap < min_gap or gap > max_gap or b + k >= n:
                continue
            if not (np.isfinite(o[a]) and np.isfinite(o[b])):
                continue
            if sign > 0 and arr[b] > arr[a] and o[b] < o[a]:
                out[b + k] = True
            if sign < 0 and arr[b] < arr[a] and o[b] > o[a]:
                out[b + k] = True
    return bull, bear
