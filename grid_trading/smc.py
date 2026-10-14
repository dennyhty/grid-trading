"""SMC / ICT（聰明錢概念）的客觀定義：擺動高低點、BOS/CHoCH、掃流動性、FVG、訂單塊、折價/溢價區。

SMC 的說法很多，這裡每個概念都選一種能寫成程式的定義（選擇理由見 research/smc_backtest.py 的報告）：

- 擺動高點 / 低點：左右各 k 根都不比它高（低）的 K 線（分形）。要等右邊 k 根收完才確認，所以會晚 k 根知道。
- 結構突破：收盤價（不是影線）突破最近一個已確認、還沒被突破過的擺動高點（低點）。
  - 原本趨勢就是同方向 → BOS（順勢延續）；原本是反方向或還沒有趨勢 → CHoCH（轉向）。
  - 突破後趨勢狀態改成突破的方向。每個擺動點只能被突破一次。
- 掃流動性（以做多為例）：影線跌破最近 liq_lookback 根內、還沒被掃過的擺動低點，但收盤收回它上方。
  收盤直接跌破的不算掃，那個低點就作廢。掃完之後只要有收盤跌破掃出來的最低點，這次掃就失效。
- 公允價值缺口 FVG（做多）：三根 K 線中第 3 根的低點高於第 1 根的高點，缺口 = [第 1 根高點, 第 3 根低點]。
- 訂單塊 OB（做多）：這段上漲的起點（波段最低那根）往回最多 5 根內，最後一根收黑的 K 線，區間取整根影線。
- 折價 / 溢價：這段波段（最低點到突破時的最高點）的 50% 以下是折價區，多單只在折價區掛單。

check 的資料都只用到當根以前（含）的 K 線，可以拿來回測也可以即時掃描。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import indicators as ind

BOS = "BOS"
CHOCH = "CHOCH"


@dataclass
class Break:
    """第 bar 根收盤時發生的結構突破。"""
    bar: int
    direction: int        # +1 向上突破擺動高點，-1 向下突破擺動低點
    kind: str             # BOS / CHOCH
    pivot: int            # 被突破的擺動點在第幾根
    sweep: int            # 突破前最近一次「同方向有效」的掃流動性在第幾根（做多看掃低點），沒有就 -1


def pivots(df: pd.DataFrame, k: int) -> tuple[np.ndarray, np.ndarray]:
    """回傳兩個布林陣列：第 i 根收盤時，第 i-k 根是否被確認為擺動高點 / 擺動低點。"""
    w = 2 * k + 1
    hh = df["high"].rolling(w).max().to_numpy()
    ll = df["low"].rolling(w).min().to_numpy()
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    ph = np.zeros(len(df), bool)
    pl = np.zeros(len(df), bool)
    ph[2 * k:] = high[k:len(df) - k] == hh[2 * k:]
    pl[2 * k:] = low[k:len(df) - k] == ll[2 * k:]
    return ph, pl


def structure(df: pd.DataFrame, k: int = 3, liq_lookback: int = 100) -> tuple[np.ndarray, list[Break]]:
    """逐根跑市場結構。回傳 (每根收盤後的趨勢狀態 +1/-1/0, 結構突破清單)。"""
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    close = df["close"].to_numpy()
    n = len(df)
    ph, pl = pivots(df, k)
    trend = np.zeros(n, np.int8)
    breaks: list[Break] = []
    state = 0
    sh = sl = None          # (價格, 第幾根)，還沒被突破的最近擺動高 / 低點
    sell_liq: list[tuple[float, int]] = []   # 擺動低點下方的停損單（多單掃這裡）
    buy_liq: list[tuple[float, int]] = []    # 擺動高點上方的停損單（空單掃這裡）
    sweep_low = sweep_high = None            # (掃出的極值, 第幾根)

    for i in range(n):
        if ph[i]:
            j = i - k
            sh = (high[j], j)
            buy_liq.append((high[j], j))
        if pl[i]:
            j = i - k
            sl = (low[j], j)
            sell_liq.append((low[j], j))

        # 掃流動性
        keep = []
        for lvl, j in sell_liq:
            if i - j > liq_lookback:
                continue
            if low[i] < lvl:
                if close[i] > lvl:
                    sweep_low = (low[i], i)
                continue          # 被掃或被跌破，都不再是流動性
            keep.append((lvl, j))
        sell_liq = keep
        keep = []
        for lvl, j in buy_liq:
            if i - j > liq_lookback:
                continue
            if high[i] > lvl:
                if close[i] < lvl:
                    sweep_high = (high[i], i)
                continue
            keep.append((lvl, j))
        buy_liq = keep
        if sweep_low is not None and close[i] < sweep_low[0]:
            sweep_low = None
        if sweep_high is not None and close[i] > sweep_high[0]:
            sweep_high = None

        # 結構突破（收盤價）
        if sh is not None and close[i] > sh[0]:
            breaks.append(Break(i, 1, BOS if state > 0 else CHOCH, sh[1], sweep_low[1] if sweep_low else -1))
            state = 1
            sh = None
        if sl is not None and close[i] < sl[0]:
            breaks.append(Break(i, -1, BOS if state < 0 else CHOCH, sl[1], sweep_high[1] if sweep_high else -1))
            state = -1
            sl = None
        trend[i] = state
    return trend, breaks


def htf_trend(df: pd.DataFrame, htf: pd.DataFrame, tf_hours: float, htf_hours: float, k: int) -> np.ndarray:
    """每根小週期 K 線收盤時，最後一根已收盤的大週期 K 線的結構趨勢（+1/-1/0）。"""
    trend, _ = structure(htf, k)
    htf_close = (htf["time"] + pd.Timedelta(hours=htf_hours)).to_numpy()
    ltf_close = (df["time"] + pd.Timedelta(hours=tf_hours)).to_numpy()
    idx = np.searchsorted(htf_close, ltf_close, side="right") - 1
    return np.where(idx >= 0, trend[np.clip(idx, 0, None)], 0)


@dataclass
class Setup:
    """在第 bar 根收盤時成立的進場計畫。entry 是限價（market 模式就是收盤價）。"""
    bar: int
    direction: int
    entry: float
    sl: float
    market: bool
    kind: str
    leg_low: float
    leg_high: float


def find_setups(df: pd.DataFrame, breaks: list[Break], htf_trend: np.ndarray | None, *,
                setup: str = "sweep", entry: str = "fvg", htf: bool = True, discount: bool = True,
                sweep_window: int = 20, sl_buf_atr: float = 0.1, min_risk_atr: float = 0.5,
                min_fvg_atr: float = 0.1, ob_search: int = 5) -> list[Setup]:
    """把結構突破轉成進場計畫。

    setup：sweep = 先掃流動性再突破（任何突破都算，ICT 的 MSS）；choch = 只要轉向突破；bos = 只要順勢突破。
    entry：market = 突破那根收盤進場；fvg = 掛在缺口靠近價格的那一邊；fvg_ce = 掛在缺口一半；ob = 掛在訂單塊靠近價格的那一邊。
    htf：只做和大週期結構同方向的單。discount：限價必須在波段的折價區（多）/ 溢價區（空）。
    """
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    close = df["close"].to_numpy()
    opn = df["open"].to_numpy()
    atr = df["atr"].to_numpy() if "atr" in df else ind.atr(df).to_numpy()
    out = []
    for b in breaks:
        d, i = b.direction, b.bar
        if setup == "sweep":
            if b.sweep < 0 or b.sweep < b.pivot or i - b.sweep > sweep_window:
                continue
        elif setup == "choch" and b.kind != CHOCH:
            continue
        elif setup == "bos" and b.kind != BOS:
            continue
        if htf and (htf_trend is None or htf_trend[i] != d):
            continue
        a = atr[i]
        if not a > 0:
            continue
        seg = slice(b.pivot, i + 1)
        # 波段：被突破的擺動點到突破那根之間，反方向的極值是波段起點 m
        if d > 0:
            m = b.pivot + int(np.argmin(low[seg]))
            leg_low, leg_high = low[m], high[m:i + 1].max()
            ext = leg_low
        else:
            m = b.pivot + int(np.argmax(high[seg]))
            leg_low, leg_high = low[m:i + 1].min(), high[m]
            ext = leg_high
        px = None
        if entry == "market":
            px = close[i]
        elif entry in ("fvg", "fvg_ce"):
            for t in range(i, m + 1, -1):
                if d > 0:
                    top, bot = low[t], high[t - 2]
                    ok = top > bot and top - bot >= min_fvg_atr * a and (t == i or low[t + 1:i + 1].min() > top)
                    near = top
                else:
                    top, bot = low[t - 2], high[t]
                    ok = top > bot and top - bot >= min_fvg_atr * a and (t == i or high[t + 1:i + 1].max() < bot)
                    near = bot
                if ok:
                    px = near if entry == "fvg" else (top + bot) / 2
                    break
        elif entry == "ob":
            for j in range(m, max(m - ob_search, b.pivot) - 1, -1):
                if (d > 0 and close[j] < opn[j]) or (d < 0 and close[j] > opn[j]):
                    near = high[j] if d > 0 else low[j]
                    after = slice(j + 1, i + 1)
                    # 價格離開訂單塊之後到突破為止不能再回到區間內（沒被測試過）
                    if d > 0:
                        left = np.flatnonzero(low[after] > near)
                        if len(left) and low[j + 1 + left[0]:i + 1].min() > near:
                            px = near
                    else:
                        left = np.flatnonzero(high[after] < near)
                        if len(left) and high[j + 1 + left[0]:i + 1].max() < near:
                            px = near
                    break
        if px is None:
            continue
        stop = ext - d * sl_buf_atr * a
        if d * (px - stop) < min_risk_atr * a:
            stop = px - d * min_risk_atr * a
        if entry != "market":
            if d * (close[i] - px) <= 0:
                continue  # 收盤已經在限價另一邊，不掛
            if discount and d * (px - (leg_high + leg_low) / 2) > 0:
                continue
        out.append(Setup(i, d, float(px), float(stop), entry == "market", b.kind, float(leg_low), float(leg_high)))
    return out
