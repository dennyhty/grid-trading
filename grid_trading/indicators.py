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


def bollinger_width(close: pd.Series, n: int = 20, k: float = 2.0) -> pd.Series:
    """布林帶寬 = (上軌 - 下軌) / 中軌。"""
    mid = close.rolling(n).mean()
    std = close.rolling(n).std(ddof=0)
    return (2 * k * std) / mid


def efficiency_ratio(close: pd.Series, n: int) -> float:
    """Kaufman 效率比：最近 n 根的淨漲跌 / 路徑總長。越接近 0 越震盪，越接近 1 越單邊。"""
    window = close.iloc[-(n + 1):]
    path = window.diff().abs().sum()
    if path == 0:
        return 0.0
    return float(abs(window.iloc[-1] - window.iloc[0]) / path)


def mean_crossings(close: pd.Series, n: int, ma_len: int = 20) -> int:
    """最近 n 根 K 線中，收盤價穿越 ma_len 均線的次數。次數多代表來回震盪、網格容易成交。"""
    ma = close.rolling(ma_len).mean()
    side = np.sign(close - ma).iloc[-n:]
    side = side[side != 0].dropna()
    return int((side.diff().abs() > 0).sum())
