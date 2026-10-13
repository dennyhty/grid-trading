"""訂單流研究用的 K 線：幣安 1H 月檔（含主動買入量）＋每日 metrics（持倉量、多空比），合成 2H/4H。

欄位（每根 K 線）：
- 價格：open/high/low/close/volume（和 binance_history 相同）
- 成交：quote_volume（USDT）、trades、taker_buy_quote（主動買入額）
- delta = 主動買入額 − 主動賣出額（USDT）；cvd = delta 的累積
- metrics（K 線收盤前最後一筆，約收盤前 5 分鐘）：oi（持倉量，幣）、oi_value、account_ratio（全體帳戶多空比）、
  top_account_ratio、top_position_ratio（大戶帳戶 / 持倉多空比）、taker_ratio
  metrics 沒有資料的期間是 NaN（BTC 從 2020-09、大部分幣從 2021-12 才有）。

沒有歷史掛單簿，所以掛單牆、冰山單、掛單撤單這類訊號無法回測。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import indicators as ind
from .binance_history import load as binance_load
from .binance_metrics import load_metrics

FLOW_AGG = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum",
            "quote_volume": "sum", "trades": "sum", "taker_buy_quote": "sum"}
METRIC_COLS = ["oi", "oi_value", "account_ratio", "top_account_ratio", "top_position_ratio", "taker_ratio"]
HOURS = {"1h": 1, "2h": 2, "4h": 4}


def flow_bars(coin: str, rule: str = "4h") -> pd.DataFrame:
    """已收盤的 rule 週期 K 線（從 1H 合成，第一天 00:00 UTC 起算），附上 delta 和 metrics。"""
    k = binance_load(coin, "1h_flow")
    if k.empty:
        return k
    m = load_metrics(coin)
    if not m.empty:
        k = k.merge(m[["time", *METRIC_COLS]], on="time", how="left")
    else:
        for c in METRIC_COLS:
            k[c] = np.nan
    agg = dict(FLOW_AGG, **{c: "last" for c in METRIC_COLS})
    out = k.set_index("time").resample(rule, label="left", closed="left", origin="start_day").agg(agg)
    out = out.dropna(subset=["close"]).reset_index()
    last_close = k["time"].iloc[-1] + pd.Timedelta(hours=1)
    out = out[out["time"] + pd.Timedelta(hours=HOURS[rule]) <= last_close].reset_index(drop=True)
    out["delta"] = 2 * out["taker_buy_quote"] - out["quote_volume"]
    out["cvd"] = out["delta"].cumsum()
    return out


def zscore(s: pd.Series, n: int = 200) -> pd.Series:
    mean = s.rolling(n, min_periods=n // 2).mean()
    std = s.rolling(n, min_periods=n // 2).std()
    return (s - mean) / std.replace(0, np.nan)


def add_features(df: pd.DataFrame, windows=(1, 3, 6), z_len: int = 200) -> pd.DataFrame:
    """所有訂單流策略共用的欄位。只用到當根（含）以前的資料。"""
    if "f_ready" in df:
        return df
    df = df.copy()
    c = df["close"]
    df["ema144"] = ind.ema(c, 144)
    df["ema169"] = ind.ema(c, 169)
    df["atr"] = ind.atr(df)
    df["trend"] = np.where((df["ema144"] > df["ema169"]) & (c > df[["ema144", "ema169"]].min(axis=1)), 1,
                           np.where((df["ema144"] < df["ema169"]) & (c < df[["ema144", "ema169"]].max(axis=1)), -1, 0))
    oi = df["oi"].where(df["oi"] > 0)
    for n in windows:
        imb = df["delta"].rolling(n).sum() / df["quote_volume"].rolling(n).sum()
        df[f"imb{n}"] = imb
        df[f"imb{n}_z"] = zscore(imb, z_len)
        df[f"ret{n}_z"] = zscore(np.log(c / c.shift(n)), z_len)
        df[f"doi{n}_z"] = zscore(np.log(oi / oi.shift(n)), z_len)
        df[f"ret{n}"] = c / c.shift(n) - 1
    df["vol_z"] = zscore(np.log(df["quote_volume"].replace(0, np.nan)), z_len)
    for col in ("account_ratio", "top_position_ratio"):
        df[f"{col}_z"] = zscore(np.log(df[col].where(df[col] > 0)), z_len)
    df["f_ready"] = True
    return df


def prior_peak(values: np.ndarray, lookback: int, gap: int, high: bool) -> np.ndarray:
    """每根 i 在 [i-lookback, i-gap] 區間裡最高（high=True）或最低那一根的位置，沒有足夠資料時是 -1。"""
    n = len(values)
    out = np.full(n, -1, dtype=np.int64)
    width = lookback - gap + 1
    if n <= lookback or width <= 0:
        return out
    v = np.where(np.isnan(values), -np.inf if high else np.inf, values)
    win = np.lib.stride_tricks.sliding_window_view(v, width)  # win[s] = v[s : s+width]
    arg = (win.argmax(axis=1) if high else win.argmin(axis=1)) + np.arange(len(win))
    # 第 i 根用的是從 i-lookback 開始的視窗
    idx = np.arange(lookback, n)
    out[idx] = arg[idx - lookback]
    return out
