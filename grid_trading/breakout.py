"""從群組交易訊號回推的「順勢突破做多」規則。

三則訊號（AERO 10/10、S 10/11、ZRO 10/12）在發出當下的共同點：
- 1H 收盤剛突破前 20 根 K 線的最高價
- 1H 收盤在 EMA20/50/144/169 之上（站上維加斯通道）
- 1H RSI 約 66~69，偏強但不追到極端
- 4H 收盤在維加斯通道之上，MACD 柱狀體為正且放大
- 停損放在最近的 1H 波段低點下方一點點（AERO、S 幾乎完全吻合，ZRO 差約 3%）
- TP1 約 1R（停損距離的 1 倍），TP2 約 2~3R，其餘目標是幾個月前的 4H 壓力位

只有三個樣本，這是「最符合的猜測」，不是對方的原始程式。
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .history import resample
from .indicators import ema


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    down = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / down)


@dataclass(frozen=True)
class BreakoutConfig:
    lookback: int = 20          # 突破前幾根 1H 的最高價
    rsi_min: float = 60.0
    rsi_max: float = 80.0
    use_4h_filter: bool = True  # 4H 站上維加斯 + MACD 柱放大
    pivot_k: int = 3            # 停損放在最近一個 1H 波段低點（左右各 k 根都比它高）下方
    stop_buffer: float = 0.003  # 低點再往下 0.3%
    min_risk: float = 0.03      # 停損距離太近容易被洗掉，太遠不划算
    max_risk: float = 0.12
    tp1_r: float = 1.0
    tp2_r: float = 2.0

    def label(self) -> str:
        f = "4H濾網" if self.use_4h_filter else "無4H濾網"
        return f"RSI{self.rsi_min:.0f}-{self.rsi_max:.0f}_{f}_TP{self.tp1_r:g}R/{self.tp2_r:g}R"


def add_indicators(df_1h: pd.DataFrame, cfg: BreakoutConfig = BreakoutConfig()) -> pd.DataFrame:
    """在 1H K 線上算好判斷需要的欄位，4H 指標只用已收盤的 4H K 線，不偷看未來。"""
    df = df_1h.copy()
    c = df["close"]
    for n in (20, 50, 144, 169):
        df[f"ema{n}"] = ema(c, n)
    df["rsi"] = rsi(c)
    df["prior_high"] = df["high"].shift(1).rolling(cfg.lookback).max()
    k = cfg.pivot_k
    low = df["low"]
    is_pivot = low == low.rolling(2 * k + 1, center=True).min()
    # 波段低點要等右邊 k 根收完才確認，所以往後移 k 根才能用
    df["swing_low"] = low.where(is_pivot).ffill().shift(k)

    h4 = resample(df_1h[["time", "open", "high", "low", "close", "volume"]], "4h")
    macd = ema(h4["close"], 12) - ema(h4["close"], 26)
    hist = macd - ema(macd, 9)
    h4 = pd.DataFrame({
        # 這根 4H 收盤後才能用：時間標成收盤時刻
        "avail": h4["time"] + pd.Timedelta(hours=4),
        "h4_trend": (h4["close"] > ema(h4["close"], 144)) & (h4["close"] > ema(h4["close"], 169)),
        "h4_macd": (hist > 0) & (hist > hist.shift(1)),
    })
    df["bar_close"] = df["time"] + pd.Timedelta(hours=1)
    df = pd.merge_asof(df.sort_values("bar_close"), h4.sort_values("avail"),
                       left_on="bar_close", right_on="avail", direction="backward")
    df[["h4_trend", "h4_macd"]] = df[["h4_trend", "h4_macd"]].fillna(False).astype(bool)
    return df.drop(columns=["avail"])


def signal_at(row: pd.Series, cfg: BreakoutConfig = BreakoutConfig()) -> dict | None:
    """傳入 add_indicators 之後某一根「已收盤」的 1H，符合就回傳開單卡內容。"""
    if pd.isna(row["prior_high"]) or pd.isna(row["ema169"]):
        return None
    close = row["close"]
    if close <= row["prior_high"]:
        return None
    if not all(close > row[f"ema{n}"] for n in (20, 50, 144, 169)):
        return None
    if not cfg.rsi_min <= row["rsi"] <= cfg.rsi_max:
        return None
    if cfg.use_4h_filter and not (row["h4_trend"] and row["h4_macd"]):
        return None
    stop = row["swing_low"] * (1 - cfg.stop_buffer)
    risk = close - stop
    if not cfg.min_risk <= risk / close <= cfg.max_risk:
        return None
    return {
        "direction": "long",
        "entry": close,
        "stop": stop,
        "tp1": close + cfg.tp1_r * risk,
        "tp2": close + cfg.tp2_r * risk,
        "risk_pct": risk / close,
    }


def scan(df_1h: pd.DataFrame, cfg: BreakoutConfig = BreakoutConfig()) -> dict | None:
    """看最新一根已收盤的 1H 有沒有訊號，給即時訊號系統用。"""
    df = add_indicators(df_1h, cfg)
    return signal_at(df.iloc[-1], cfg)
