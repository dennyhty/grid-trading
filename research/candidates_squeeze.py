"""候選 C：日線波動壓縮（squeeze）後的突破，多空都做，時點幣池。

經濟理由（誰把錢輸給你）：波動有群聚性，長時間的窄幅盤整之後通常接著放大。盤整期間賣波動的人
（網格機器人、賣選擇權、區間來回做的人）在區間外累積反向部位，突破後被迫停損追價，推動價格往突破方向走。
和 A（唐奇安）比，它只在「先壓縮」的時候才做，訊號更少、更早，想要的是較高的勝率（出場用 TP 分批）。

訊號（日線收盤）：
- 壓縮：布林通道（20 日、2 倍標準差）寬度 / 中線，在自己過去 120 天（不含當天）的分位數 <= q。
- 進場：前一天處於壓縮狀態（或前 5 天內任何一天），今天收盤突破上軌做多、跌破下軌做空（只取第一次）。
- 1R = stop 倍 ATR(20)，收盤進場；同一個幣同時只一張。
出場（兩種，預先登記）：
- ladder：實盤式 1/3/5R 各平 1/4、TP1 後保本、TP2 後移到 TP1、最後 1/4 追蹤 3 ATR。
- half1r：1R 平一半、保本、剩下追蹤 3 ATR（舊 4H 實盤出場；勝率較高）。

網格（預先登記，共 8 組）：q ∈ {0.10, 0.20} × stop ∈ {1.5, 2.5} ATR × 出場 ∈ {ladder, half1r}。
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from research.candidates_common import HALF_1R, LADDER, ExitPlan, adv30, simulate, trade_frame, wilder_atr

FAMILY = "cand_daily_squeeze"
BB_N, BB_K, PCT_WIN, RECENT = 20, 2.0, 120, 5
WARMUP = BB_N + PCT_WIN + 5


def configs() -> dict[str, dict]:
    out = {}
    for q, k, ex in itertools.product((0.10, 0.20), (1.5, 2.5), ("ladder", "half1r")):
        out[f"q{q:g}|sl{k:g}|{ex}"] = {"q": q, "stop": k, "exit": ex}
    return out


def neighbors(name: str) -> list[str]:
    """鄰近設定：q 和 stop 各換到另一格（出場方式不算鄰近）。"""
    p = configs()[name]
    q2 = 0.2 if p["q"] == 0.1 else 0.1
    k2 = 2.5 if p["stop"] == 1.5 else 1.5
    return [f"q{q2:g}|sl{p['stop']:g}|{p['exit']}", f"q{p['q']:g}|sl{k2:g}|{p['exit']}"]


def plan_for(p: dict) -> ExitPlan:
    return ExitPlan(stop_atr=p["stop"], **(LADDER if p["exit"] == "ladder" else HALF_1R))


def trailing_pct_rank(x, win: int) -> np.ndarray:
    """x[i] 在 x[i-win .. i-1]（不含當天）裡的分位數（比它小的比例），資料不足為 NaN。"""
    x = np.asarray(x, float)
    out = np.full(len(x), np.nan)
    if len(x) <= win:
        return out
    past = np.lib.stride_tricks.sliding_window_view(x, win)[:-1]   # 第 i 列 = x[i .. i+win-1]，對應今天 i+win
    cur = x[win:]
    valid = np.isfinite(past).sum(axis=1)
    less = (past < cur[:, None]).sum(axis=1)   # NaN 比較為 False，不算進去
    with np.errstate(invalid="ignore", divide="ignore"):
        r = less / valid
    out[win:] = np.where((valid >= win // 2) & np.isfinite(cur), r, np.nan)
    return out


def bands(close, n: int = BB_N, k: float = BB_K):
    """布林上下軌和寬度分位數（寬度 / 中線，在過去 win 天裡的分位數）。"""
    c = pd.Series(np.asarray(close, float))
    mid = c.rolling(n).mean()
    sd = c.rolling(n).std(ddof=0)
    up, lo = mid + k * sd, mid - k * sd
    return up.to_numpy(), lo.to_numpy(), ((up - lo) / mid).to_numpy()


def squeeze_signals(close, q: float, n: int = BB_N, k: float = BB_K, win: int = PCT_WIN, recent: int = RECENT,
                    rank: np.ndarray | None = None) -> np.ndarray:
    """+1 / -1 / 0：前 recent 天內（不含今天）曾經壓縮（寬度分位數 <= q），今天第一次收盤突破上軌/跌破下軌。"""
    c = pd.Series(np.asarray(close, float))
    up, lo, width = bands(close, n, k)
    if rank is None:
        rank = trailing_pct_rank(width, win)
    sq = np.nan_to_num(rank, nan=1.0) <= q
    was_sq = pd.Series(sq).shift(1).rolling(recent, min_periods=1).max().fillna(0).astype(bool).to_numpy()
    above = (c > up).to_numpy()
    below = (c < lo).to_numpy()
    first_up = above & ~np.r_[False, above[:-1]]
    first_dn = below & ~np.r_[False, below[:-1]]
    return np.where(was_sq & first_up, 1, np.where(was_sq & first_dn, -1, 0)).astype(np.int64)


def arrays(d: pd.DataFrame) -> dict:
    o, h, l, c = (d[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    return {"open": o, "high": h, "low": l, "close": c, "atr": wilder_atr(h, l, c, 20)}


_SIG_CACHE: dict = {}


def symbol_trades(sym: str, d: pd.DataFrame, p: dict) -> pd.DataFrame | None:
    if len(d) < WARMUP + 10:
        return None
    arr = arrays(d)
    key = (sym, len(d))
    if key not in _SIG_CACHE:   # 寬度分位數和 q 無關，算一次就好
        _SIG_CACHE[key] = trailing_pct_rank(bands(arr["close"])[2], PCT_WIN)
    sig = squeeze_signals(arr["close"], p["q"], rank=_SIG_CACHE[key])
    ok = (np.arange(len(d)) >= WARMUP) & (np.arange(len(d)) + 1 < len(d)) & np.isfinite(arr["atr"])
    idx = np.flatnonzero((sig != 0) & ok)
    if len(idx) == 0:
        return None
    ds = sig[idx]
    sim = simulate(arr, idx, ds, plan_for(p))
    return trade_frame(sym, d, idx, ds, sim, adv30(d["qv"].to_numpy()))


def raw_trades(data: dict[str, pd.DataFrame], p: dict) -> pd.DataFrame:
    parts = [t for s, d in data.items() if (t := symbol_trades(s, d, p)) is not None]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def plan_of_factory(data: dict[str, pd.DataFrame], p: dict):
    cache = {}

    def f(sym):
        if sym not in cache:
            cache[sym] = (arrays(data[sym]), plan_for(p))
        return cache[sym]
    return f
