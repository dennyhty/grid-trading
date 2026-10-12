"""選幣篩選：從 K 線判斷哪些幣種目前適合開網格，並給出初步的方向、區間與停損。

邏輯：網格靠來回震盪賺錢，單邊趨勢是主要虧損來源。所以分數偏好
  1. 效率比低（走勢來回、不單邊）
  2. ADX 低（沒有強趨勢）
  3. 穿越均線次數多（網格容易成交）
  4. 波動度適中（夠付手續費，又不至於太容易被打穿）
  5. 成交額高（流動性好）
方向用維加斯通道（EMA144 / EMA169）判斷：價格在通道上方且通道明顯向上 → 做多網格，
下方且明顯向下 → 做空網格，其他 → 中性網格。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import indicators as ind

# 派網合約網格每次成交的手續費約 0.05%，一格買賣要付兩次；格距低於這個值等於白做。
MIN_GRID_SPACING_PCT = 0.3
# 區間寬度超過這個值的幣不適合網格（例如 20 天內翻倍過的幣）
MAX_RANGE_WIDTH_PCT = 50.0


@dataclass
class ScreenConfig:
    lookback: int = 120          # 用最近幾根 K 線評估（4H × 120 = 20 天）
    min_bars: int = 200          # K 線不足（新上市幣）就跳過，維加斯通道需要足夠歷史
    range_quantile: float = 0.05  # 區間上下限取最近 lookback 根高低點的 5% / 95% 分位，排除插針
    stop_atr_mult: float = 1.0   # 停損設在區間外 1 倍 ATR
    trend_slope_bars: int = 12   # 判斷通道方向時看幾根前的 EMA
    min_trend_slope_pct: float = 0.5  # 通道斜率小於這個值視為走平，不算趨勢


def analyze(df: pd.DataFrame, cfg: ScreenConfig = ScreenConfig()) -> dict | None:
    """計算單一幣種的指標與網格建議。K 線不足時回傳 None。"""
    if len(df) < cfg.min_bars:
        return None
    close = df["close"]
    last = float(close.iloc[-1])
    recent = df.iloc[-cfg.lookback:]

    atr_now = float(ind.atr(df).iloc[-1])
    atr_pct = atr_now / last * 100
    adx_now = float(ind.adx(df).iloc[-1])
    er = ind.efficiency_ratio(close, cfg.lookback)
    crossings = ind.mean_crossings(close, cfg.lookback)
    bbw = float(ind.bollinger_width(close).iloc[-1]) * 100

    ema144 = ind.ema(close, 144)
    ema169 = ind.ema(close, 169)
    top, bottom = max(ema144.iloc[-1], ema169.iloc[-1]), min(ema144.iloc[-1], ema169.iloc[-1])
    slope = (ema144.iloc[-1] - ema144.iloc[-1 - cfg.trend_slope_bars]) / ema144.iloc[-1] * 100
    if last > top and slope > cfg.min_trend_slope_pct:
        direction = "做多"
    elif last < bottom and slope < -cfg.min_trend_slope_pct:
        direction = "做空"
    else:
        direction = "中性"

    lower = float(recent["low"].quantile(cfg.range_quantile))
    upper = float(recent["high"].quantile(1 - cfg.range_quantile))
    width_pct = (upper - lower) / lower * 100
    in_range = lower <= last <= upper
    position = (last - lower) / (upper - lower) if upper > lower else np.nan

    # 格距抓半根 K 線的波動，讓網格在一根 4H K 線內有機會來回成交
    spacing_pct = max(atr_pct * 0.5, MIN_GRID_SPACING_PCT)
    grids = int(np.clip(round(width_pct / spacing_pct), 5, 100))

    stop_low = lower - cfg.stop_atr_mult * atr_now
    stop_high = upper + cfg.stop_atr_mult * atr_now

    return {
        "price": last,
        "atr_pct": atr_pct,
        "adx": adx_now,
        "efficiency": er,
        "crossings": crossings,
        "bb_width_pct": bbw,
        "vegas_slope_pct": float(slope),
        "direction": direction,
        "lower": lower,
        "upper": upper,
        "range_width_pct": width_pct,
        "position_in_range": position,
        "in_range": in_range,
        "grids": grids,
        "grid_spacing_pct": width_pct / grids,
        "stop_low": stop_low if direction != "做空" else np.nan,
        "stop_high": stop_high if direction != "做多" else np.nan,
    }


def _volatility_fit(atr_pct: float) -> float:
    """4H ATR 在 0.8%~3% 給滿分，往外線性遞減。太小賺不到手續費，太大容易被打穿區間。"""
    if 0.8 <= atr_pct <= 3.0:
        return 1.0
    if atr_pct < 0.8:
        return max(0.0, atr_pct / 0.8)
    return max(0.0, 1 - (atr_pct - 3.0) / 3.0)


def score(rows: pd.DataFrame) -> pd.Series:
    """0~100 分，越高越適合開網格。"""
    chop = 1 - rows["efficiency"].clip(0, 1)
    no_trend = ((40 - rows["adx"]) / 30).clip(0, 1)
    cross = (rows["crossings"] / 20).clip(0, 1)
    vol = rows["atr_pct"].apply(_volatility_fit)
    if "amount" in rows and rows["amount"].notna().any():
        liq = np.log10(rows["amount"].clip(lower=1)).rank(pct=True)
    else:
        liq = pd.Series(0.5, index=rows.index)
    s = 30 * chop + 25 * no_trend + 20 * cross + 15 * vol + 10 * liq
    # 價格已經跑出區間 → 可能正在突破，先降分
    s = s.where(rows["in_range"], s * 0.6)
    # 區間太寬（通常是期間內暴漲暴跌過）→ 格距被迫拉大、資金效率差，也降分
    s = s.where(rows["range_width_pct"] <= MAX_RANGE_WIDTH_PCT, s * 0.6)
    return s.round(1)


def rank(results: dict[str, dict], tickers: pd.DataFrame | None = None) -> pd.DataFrame:
    """把每個幣的 analyze() 結果合成表格並排序。"""
    rows = pd.DataFrame.from_dict(results, orient="index")
    rows.index.name = "symbol"
    if tickers is not None and not tickers.empty and "amount" in tickers:
        rows = rows.join(tickers.set_index("symbol")[["amount"]], how="left")
    rows["score"] = score(rows)
    return rows.sort_values("score", ascending=False)
