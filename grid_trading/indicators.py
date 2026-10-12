"""技術指標。輸入都是依時間由舊到新排序的 pandas Series / DataFrame。"""

from __future__ import annotations

import numpy as np
import pandas as pd


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def true_range(df: pd.DataFrame) -> pd.Series:
    prev = df["close"].shift(1)
    return pd.concat(
        [df["high"] - df["low"], (df["high"] - prev).abs(), (df["low"] - prev).abs()], axis=1
    ).max(axis=1)


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """Wilder ATR。"""
    return true_range(df).ewm(alpha=1 / n, adjust=False).mean()


def adx(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """Wilder ADX：低於 20 左右代表沒有明顯趨勢（震盪），高於 25 代表趨勢明顯。"""
    up = df["high"].diff()
    down = -df["low"].diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)
    tr = atr(df, n)
    plus_di = 100 * plus_dm.ewm(alpha=1 / n, adjust=False).mean() / tr
    minus_di = 100 * minus_dm.ewm(alpha=1 / n, adjust=False).mean() / tr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / n, adjust=False).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    """Wilder RSI。"""
    diff = close.diff()
    gain = diff.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    loss = (-diff.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(100.0)


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """回傳 macd（DIF）、signal（DEA）、hist（柱狀體）三欄。"""
    dif = ema(close, fast) - ema(close, slow)
    dea = ema(dif, signal)
    return pd.DataFrame({"macd": dif, "signal": dea, "hist": dif - dea})


def wma(s: pd.Series, n: int) -> pd.Series:
    w = np.arange(1, n + 1, dtype=float)
    return s.rolling(n).apply(lambda x: np.dot(x, w) / w.sum(), raw=True)


def hma(close: pd.Series, n: int = 55) -> pd.Series:
    """Hull 移動平均：WMA(2*WMA(n/2) - WMA(n), sqrt(n))，反應比 EMA 快。"""
    half, root = max(int(n / 2), 1), max(int(np.sqrt(n)), 1)
    return wma(2 * wma(close, half) - wma(close, n), root)


def psar(df: pd.DataFrame, start: float = 0.02, step: float = 0.02, maximum: float = 0.2) -> pd.Series:
    """拋物線轉向指標（Parabolic SAR），和 TradingView 預設參數相同。點在價格下方代表多頭。"""
    high, low = df["high"].to_numpy(), df["low"].to_numpy()
    n = len(df)
    out = np.full(n, np.nan)
    if n < 2:
        return pd.Series(out, index=df.index)
    up = high[1] >= high[0]
    sar = low[0] if up else high[0]
    ep = high[0] if up else low[0]
    af = start
    for i in range(1, n):
        sar = sar + af * (ep - sar)
        if up:
            sar = min(sar, low[i - 1], low[i - 2] if i >= 2 else low[i - 1])
            if low[i] < sar:
                up, sar, ep, af = False, ep, low[i], start
            elif high[i] > ep:
                ep, af = high[i], min(af + step, maximum)
        else:
            sar = max(sar, high[i - 1], high[i - 2] if i >= 2 else high[i - 1])
            if high[i] > sar:
                up, sar, ep, af = True, ep, high[i], start
            elif low[i] < ep:
                ep, af = low[i], min(af + step, maximum)
        out[i] = sar
    return pd.Series(out, index=df.index)
