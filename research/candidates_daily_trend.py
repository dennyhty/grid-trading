"""候選 A：日線唐奇安突破（海龜式趨勢追蹤），多空都做，時點幣池。

經濟理由（誰把錢輸給你）：日線級別的趨勢延續來自反應不足與追價（新資訊慢慢被價格吸收、FOMO 資金後到），
以及突破時被迫平倉的人（被軋的空單、被清算的槓桿、網格/區間機器人在區間外的反向庫存）。
賺的是「少數大波段」，代價是勝率低、要忍受很多小止損；週期比實盤 4H/2H 維加斯長很多，持有數週，
希望和維加斯（短週期、靠崩盤日的空單）的日報酬相關性低、在多頭年做多也能賺。

訊號（日線收盤，UTC 00:00）：
- 做多：收盤 > 前 N 天（不含當天）的最高價，且前一天收盤還沒突破（只取第一次突破）；做空對稱。
- 1R = stop 倍 ATR(20)，收盤價進場、隔天開始算出場；同一個幣同時只一張。
出場（兩種，預先登記）：
- chan：海龜式，收盤跌破前 M 天最低（空單：漲破前 M 天最高）就出場，M = 20→10、55→20、100→50；加初始硬止損。
- ladder：實盤式 1/3/5R 各平 1/4、TP1 後保本、TP2 後移到 TP1、最後 1/4 追蹤 3 ATR。

網格（預先登記，共 12 組）：N ∈ {20, 55, 100} × stop ∈ {2, 3} ATR × 出場 ∈ {chan, ladder}。
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from research.candidates_common import LADDER, ExitPlan, adv30, simulate, trade_frame, wilder_atr

FAMILY = "cand_daily_donchian"
EXIT_LEN = {20: 10, 55: 20, 100: 50}
WARMUP = 60


def configs() -> dict[str, dict]:
    out = {}
    for n, k, ex in itertools.product((20, 55, 100), (2.0, 3.0), ("chan", "ladder")):
        out[f"N{n}|sl{k:g}|{ex}"] = {"n": n, "stop": k, "exit": ex}
    return out


def neighbors(name: str) -> list[str]:
    """參數高原用的鄰近設定：只改一個參數、往相鄰的一格（N 依 20/55/100 順序；stop 2↔3；出場方式不算鄰近）。"""
    p = configs()[name]
    ns = [20, 55, 100]
    out = []
    i = ns.index(p["n"])
    for j in (i - 1, i + 1):
        if 0 <= j < len(ns):
            out.append(f"N{ns[j]}|sl{p['stop']:g}|{p['exit']}")
    other = 3.0 if p["stop"] == 2.0 else 2.0
    out.append(f"N{p['n']}|sl{other:g}|{p['exit']}")
    return out


def plan_for(p: dict) -> ExitPlan:
    if p["exit"] == "chan":
        return ExitPlan(stop_atr=p["stop"], chan_exit=EXIT_LEN[p["n"]])
    return ExitPlan(stop_atr=p["stop"], **LADDER)


def breakout_signals(high, low, close, n: int) -> np.ndarray:
    """回傳每根 K 棒的方向：+1 第一次收盤突破前 n 天最高、-1 第一次收盤跌破前 n 天最低、0 沒有。
    「前 n 天」不含當天，所以只用當下已收盤的資料。"""
    h, l, c = (pd.Series(np.asarray(x, float)) for x in (high, low, close))
    up_lvl = h.rolling(n).max().shift(1)
    dn_lvl = l.rolling(n).min().shift(1)
    up = (c > up_lvl).to_numpy()
    dn = (c < dn_lvl).to_numpy()
    up_prev = np.r_[False, up[:-1]]
    dn_prev = np.r_[False, dn[:-1]]
    return np.where(up & ~up_prev, 1, np.where(dn & ~dn_prev, -1, 0)).astype(np.int64)


def arrays(d: pd.DataFrame) -> dict:
    o, h, l, c = (d[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    return {"open": o, "high": h, "low": l, "close": c, "atr": wilder_atr(h, l, c, 20)}


def symbol_trades(sym: str, d: pd.DataFrame, p: dict) -> pd.DataFrame | None:
    if len(d) < WARMUP + 10:
        return None
    arr = arrays(d)
    sig = breakout_signals(arr["high"], arr["low"], arr["close"], p["n"])
    ok = (np.arange(len(d)) >= max(WARMUP, p["n"] + 1)) & (np.arange(len(d)) + 1 < len(d)) & np.isfinite(arr["atr"])
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
    """給隨機進場基準用：symbol -> (arrays, ExitPlan)。"""
    cache = {}

    def f(sym):
        if sym not in cache:
            cache[sym] = (arrays(data[sym]), plan_for(p))
        return cache[sym]
    return f
