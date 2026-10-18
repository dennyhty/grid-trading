"""vegas_trend 止損深入研究（只做分析，不改實盤規則）。

python -m research.stop_research --interval 4H [--sub] [--slip 0.0005] [--source pionex]

需要另外 pip install numba（只有研究程式用到，網站不需要）。

進場和實盤完全一樣（vegas_trend，RSI 54-70/30-46），出場的 TP 結構也和實盤一樣
（4H：1R 平一半、保本、剩下 3 ATR 追蹤；2H：1/3/5R 各平 1/4、最後 1/4 追蹤 3 ATR），只換止損：

- 止損寬度：ATR 倍數 0.75~3.5，TP 跟著 R 走（實盤寫法）或 TP 價格固定在現在的位置。
- 止損算法：不同週期的 ATR、ATR 取較大值、固定百分比、維加斯通道外、前 N 根高低點再加緩衝。
- TP1 之前怎麼管：收盤確認止損（收盤跌破才出場，另設較寬的硬止損）、時間止損、
  從進場就開始的追蹤止損、浮盈到某個程度先把止損往上移、訊號失效（收回通道內、MACD 柱翻回去）就出場。

每個訊號各自獨立模擬一次（同一批訊號，方便成對比較），再照回測規則
（同一個幣同時只持有一張）挑出實際會做的交易算帳戶績效。--sub 用更細的 K 線
（4H 用 1H、2H 用 15m）判斷同一根 K 棒裡先碰到止損還是 TP。

輸出（output/）：
- stop_grid_{週期}{標記}.csv：每個止損設定 × 期間的帳戶績效、群集績效、和實盤的成對差（群集穩健標準誤）、
  趨勢一致的隨機進場下和實盤的成對差。
- stop_signals_{週期}{標記}.npz：每個訊號 × 設定的 R、出場 K 棒等，給後續分析用。
- stop_paths_{週期}{標記}.npz：每個訊號往有利/不利方向走到 k 倍 ATR 的第一根 K 棒（診斷用）。

python -m research.stop_research --interval 4H --analyze 讀上面幣安的結果（主回測，加上有跑過的 --slip、--sub 情境），輸出：
- stop_race_{週期}.csv：止損 x ATR 和目標誰先到（訊號 vs 隨機進場）。
- stop_selection_{週期}.csv：用一段時期挑前幾名設定、看另一段的表現。
- stop_bootstrap_{週期}.csv：主要候選和實盤的 Sharpe 差（30 天一塊重抽）。
- stop_reentry_{週期}.csv：被止損後收盤回到進場價就再進場。
"""

from __future__ import annotations

import argparse
import json
import sys
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit

from grid_trading import indicators as ind
from grid_trading.binance_history import load, load_extended
from grid_trading.history import resample
from grid_trading.signals.strategy import LONG
from grid_trading.signals.vegas_trend import VegasTrend
from grid_trading.universe import TOP100

MAX_BARS = 240
SPLIT = pd.Timestamp("2024-01-01", tz="UTC")
MIN_STOP_PCT = 0.001
LIVE_ATR = 1.5
N_RANDOM = 5
LEVELS = np.round(np.arange(0.25, 8.01, 0.25), 2)  # 診斷用：ATR 倍數
PLANS = {  # 實盤出場（2026-10-13）：TP 的 R 倍數、平倉比例、TP 都碰到後的追蹤 ATR 倍數
    "4H": ((1.0,), (0.5,), 3.0),
    "2H": ((1.0, 3.0, 5.0), (0.25, 0.25, 0.25), 3.0),
}
SUB = {"4H": ("1h", "4h"), "2H": ("15m", "2h")}
PARENT_MS = {"4H": 4 * 3_600_000, "2H": 2 * 3_600_000}


def V(name, group, stop=("atr", 1.5), frame="R", **m):
    """一個止損設定。stop：(種類, 參數...)；frame：R＝TP 跟著 R 走、P＝TP 價格固定在實盤位置；m：TP1 前的管理。"""
    return {"name": name, "group": group, "stop": stop, "frame": frame, **m}


def variants() -> list[dict]:
    out = [V("實盤 1.5ATR", "live")]
    for k in (0.75, 1.0, 1.25, 1.75, 2.0, 2.25, 2.5, 3.0, 3.5):
        out.append(V(f"ATR{k}", "width_R", ("atr", k)))
    for k in (0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 3.0, 3.5):
        out.append(V(f"ATR{k} TP價固定", "width_P", ("atr", k), "P"))
    for n in (7, 28, 50, 100):
        out.append(V(f"1.5×ATR{n}", "atr_basis", ("atrn", 1.5, n)))
    for n in (50, 100):
        out.append(V(f"1.5×max(ATR14,ATR{n})", "atr_basis", ("atrmax", 1.5, n)))
    out.append(V("2.0×max(ATR14,ATR50)", "atr_basis", ("atrmax", 2.0, 50)))
    for p in (2, 3, 4, 5, 6, 8):
        out.append(V(f"固定 {p}%", "pct", ("pct", p)))
    for line, b in (("near", 0.5), ("near", 1.0), ("far", 0.0), ("far", 0.5), ("far", 1.0)):
        nm = "通道近側" if line == "near" else "通道遠側"
        out.append(V(f"{nm}外 {b}ATR", "channel", ("chan", line, b, 1.0, 5.0)))
        out.append(V(f"{nm}外 {b}ATR TP價固定", "channel_P", ("chan", line, b, 1.0, 5.0), "P"))
    for n in (5, 10, 20):
        for b in (0.25, 0.5, 1.0):
            out.append(V(f"前{n}根高低點外 {b}ATR", "swing", ("swing", n, b, 1.0, 4.0)))
    for n, b in ((10, 0.5), (20, 0.5)):
        out.append(V(f"前{n}根高低點外 {b}ATR TP價固定", "swing_P", ("swing", n, b, 1.0, 4.0), "P"))
    # 收盤確認：sizing 用收盤止損距離（1R），硬止損另外設
    for soft, hard in ((1.0, 2.0), (1.0, 2.5), (1.5, 2.0), (1.5, 2.5), (1.5, 3.0), (1.5, 99.0), (2.0, 3.0)):
        hn = "無硬止損" if hard > 50 else f"硬止損 {hard}ATR"
        out.append(V(f"收盤破 {soft}ATR 出場，{hn}", "close", ("close", soft, hard)))
    for soft, hard in ((1.0, 2.0), (1.0, 2.5)):
        out.append(V(f"收盤破 {soft}ATR 出場，硬止損 {hard}ATR TP價固定", "close_P", ("close", soft, hard), "P"))
    for n in (3, 6, 12, 24, 48):
        out.append(V(f"{n} 根沒到 TP1 就出場", "time", ts_bars=n, ts_mode=0))
        out.append(V(f"{n} 根後收盤虧損就出場", "time", ts_bars=n, ts_mode=1))
    for k in (1.5, 2.0, 2.5, 3.0):
        out.append(V(f"進場就追蹤 {k}ATR", "early_trail", etrail=k))
    for trig, lvl in ((0.5, -0.5), (0.75, -0.5), (0.5, 0.0), (0.75, 0.0), (0.75, -0.25)):
        out.append(V(f"浮盈 {trig}R 後止損移到 {lvl:+}R", "lock", lock_trig=trig, lock_lvl=lvl))
    for pr, nm in ((1, "TP1 後止損移到 -0.5R（不到保本）"), (2, "TP1 後止損不動"), (5, "TP1 後止損移到 +0.25R"),
                   (3, "TP1 後 -0.5R、TP2 後保本"), (4, "TP1 後不動、TP2 後保本"), (6, "TP2 後只移到保本")):
        out.append(V(nm, "protect", protect=pr))
    for k in (2.0, 2.5, 3.5, 4.0, 5.0):
        out.append(V(f"最後一段追蹤 {k}ATR", "trail", trail_k=k))
    out.append(V("收盤回到通道內就出場", "invalid", inv_mode=1))
    out.append(V("收盤穿過通道就出場", "invalid", inv_mode=2))
    out.append(V("MACD 柱翻回就出場", "invalid", inv_mode=3))
    return out


@njit(cache=True)
def sim_many(op, hi, lo, cl, is_close, starts, ds, entries, risks, stop0s, tps, portions, protect, trails,
             fee, slip, max_bars, ts_bars, ts_mode, etrails, lock_trig, lock_lvl, softs, inv_mode, inv_lo, inv_hi,
             gapfill):
    """每個訊號各自跑一次。tps 是 (訊號數, TP 數) 的價格。回傳 R、碰到幾個 TP、出場索引、出場種類、TP1 前最大逆向（R）。

    gapfill=False 時止損一律用止損價成交（等於止損單一開始就掛著、K 棒內碰到就成交，和 tracker 相同）；
    True 時開盤已經在止損外就用開盤價。

    出場種類：1 止損（TP1 前）、2 保本/保護/追蹤止損、3 TP 全中、4 收盤規則（收盤確認、時間、訊號失效）、
    5 超過 max_bars、-1 資料結束還沒出場。
    """
    m = len(starts)
    out_r = np.zeros(m)
    out_hit = np.zeros(m, np.int64)
    out_j = np.zeros(m, np.int64)
    out_kind = np.zeros(m, np.int64)
    out_mae = np.zeros(m)
    n = len(hi)
    ntp = tps.shape[1]
    for s in range(m):
        d = ds[s]
        entry = entries[s]
        risk = risks[s]
        stop = stop0s[s]
        trail = trails[s]
        etrail = etrails[s]
        soft = softs[s]
        be = entry * (1 + d * 2 * fee)
        ecost = (fee + slip) * entry
        remaining = 1.0
        r = 0.0
        hit = 0
        extreme = entry
        ext_all = entry
        worst = entry
        bars = 0
        locked = False
        kind = -1
        j = starts[s]
        while j < n:
            a = lo[j] if d > 0 else hi[j]
            if hit == 0 and d * (worst - a) > 0:
                worst = a
            if d * (a - stop) <= 0:
                px = stop
                if gapfill and d * (op[j] - stop) < 0:  # 開盤已經跳過止損，用開盤價成交
                    px = op[j]
                r += remaining * (d * (px - entry) - ecost - (fee + slip) * px) / risk
                kind = 1 if hit == 0 else 2
                break
            f = hi[j] if d > 0 else lo[j]
            while hit < ntp and d * (f - tps[s, hit]) >= 0:
                p = portions[hit]
                r += p * (d * (tps[s, hit] - entry) - ecost - fee * tps[s, hit]) / risk
                remaining -= p
                # TP 後的保護止損。0（實盤）：TP1 後保本、TP2 後移到 TP1；1：TP1 後 -0.5R；2：TP1 後不動；
                # 3：TP1 後 -0.5R、TP2 後保本；4：TP1 後不動、TP2 後保本；5：TP1 後 +0.25R；6：TP1、TP2 後都只到保本
                ns = stop
                if hit == 0:
                    if protect == 0 or protect == 6:
                        ns = be
                    elif protect == 1 or protect == 3:
                        ns = entry - d * 0.5 * risk
                    elif protect == 5:
                        ns = entry + d * 0.25 * risk
                elif hit == 1:
                    if protect == 3 or protect == 4 or protect == 6:
                        ns = be
                    else:
                        ns = tps[s, 0]
                if d * (ns - stop) > 0:
                    stop = ns
                hit += 1
            if remaining <= 1e-9:
                kind = 3
                break
            if d * (f - ext_all) > 0:
                ext_all = f
            if etrail > 0:
                s2 = ext_all - d * etrail
                if d * (s2 - stop) > 0:
                    stop = s2
            if lock_trig > 0 and not locked and d * (ext_all - entry) >= lock_trig * risk:
                locked = True
                s2 = entry + d * lock_lvl * risk
                if d * (s2 - stop) > 0:
                    stop = s2
            if trail > 0 and hit == ntp:
                if d * (f - extreme) > 0:
                    extreme = f
                s2 = extreme - d * trail
                if d * (s2 - stop) > 0:
                    stop = s2
            if is_close[j]:
                bars += 1
                c = cl[j]
                ex = False
                if hit == 0:
                    if soft > 0 and d * (c - (entry - d * soft)) <= 0:
                        ex = True
                    if ts_bars > 0 and bars >= ts_bars and (ts_mode == 0 or d * (c - entry) <= 0):
                        ex = True
                    if inv_mode == 3:
                        if d * inv_lo[j] < 0:  # MACD 柱翻回反方向
                            ex = True
                    elif inv_mode > 0:
                        lvl = inv_lo[j] if d > 0 else inv_hi[j]
                        if d * (c - lvl) < 0:
                            ex = True
                if ex or bars >= max_bars:
                    r += remaining * (d * (c - entry) - ecost - (fee + slip) * c) / risk
                    kind = 4 if ex else 5
                    break
            j += 1
        if kind == -1:
            j = n - 1
        out_r[s] = r
        out_hit[s] = hit
        out_j[s] = j
        out_kind[s] = kind
        out_mae[s] = d * (entry - worst) / risk
    return out_r, out_hit, out_j, out_kind, out_mae


@njit(cache=True)
def first_passage(hi, lo, is_close, starts, ds, entries, atrs, levels, max_bars):
    """每個訊號：往有利/不利方向第一次走到 levels 倍 ATR 是第幾根（從 1 起算，母 K 棒；沒到 = 0）。
    子 K 棒模式下同一根母 K 棒裡也分得出先後：用子 K 棒的序號 sub_fav/sub_adv 比較。"""
    m = len(starts)
    L = len(levels)
    fav = np.zeros((m, L), np.int64)
    adv = np.zeros((m, L), np.int64)
    sfav = np.zeros((m, L), np.int64)
    sadv = np.zeros((m, L), np.int64)
    n = len(hi)
    for s in range(m):
        d = ds[s]
        e = entries[s]
        a = atrs[s]
        bars = 0
        k = 0
        j = starts[s]
        while j < n and bars < max_bars:
            k += 1
            f = (hi[j] - e) / a if d > 0 else (e - lo[j]) / a
            g = (e - lo[j]) / a if d > 0 else (hi[j] - e) / a
            for q in range(L):
                if fav[s, q] == 0 and f >= levels[q]:
                    fav[s, q] = bars + 1
                    sfav[s, q] = k
                if adv[s, q] == 0 and g >= levels[q]:
                    adv[s, q] = bars + 1
                    sadv[s, q] = k
            if is_close[j]:
                bars += 1
            j += 1
    return fav, adv, sfav, sadv


def load_frame(coin: str, interval: str, source: str) -> pd.DataFrame:
    if source == "pionex":
        from grid_trading.history import fetch_history
        sym = f"{coin}_USDT_PERP"
        if interval == "4H":
            df = fetch_history(sym, "4H", 1300)
        else:
            df = fetch_history(sym, "60M", 420)
            if len(df):
                df = resample(df, "2h", origin="epoch")
        return df.reset_index(drop=True)
    get = load_extended if source == "binance_ext" else load  # binance_ext：合約上市前接現貨（2017-08 起）
    if interval == "4H":
        return get(coin, "4h")
    df = get(coin, "1h")
    if len(df):
        df = resample(df, "2h", origin="epoch")
    return df.reset_index(drop=True)


def stop_arrays(v: dict, p: dict, idx: np.ndarray, ds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """每個訊號的 1R（下單用的止損距離）和硬止損距離（價格）。"""
    price = p["close"][idx]
    atr = p["atr"][idx]
    kind = v["stop"][0]
    hard = None
    if kind == "atr":
        dist = v["stop"][1] * atr
    elif kind == "atrn":
        dist = v["stop"][1] * p[f"atr{v['stop'][2]}"][idx]
    elif kind == "atrmax":
        dist = v["stop"][1] * np.maximum(atr, p[f"atr{v['stop'][2]}"][idx])
    elif kind == "pct":
        dist = v["stop"][1] / 100 * price
    elif kind == "chan":
        _, line, b, lo_, hi_ = v["stop"]
        top, bot = p["chan_top"][idx], p["chan_bot"][idx]
        if line == "near":
            lvl = np.where(ds > 0, top, bot)
        else:
            lvl = np.where(ds > 0, bot, top)
        dist = np.clip(ds * (price - lvl) + b * atr, lo_ * atr, hi_ * atr)
    elif kind == "swing":
        _, nb, b, lo_, hi_ = v["stop"]
        lvl = np.where(ds > 0, p[f"lo{nb}"][idx], p[f"hi{nb}"][idx])
        dist = np.clip(ds * (price - lvl) + b * atr, lo_ * atr, hi_ * atr)
    elif kind == "close":
        dist = v["stop"][1] * atr
        hard = v["stop"][2] * atr
    else:
        raise ValueError(kind)
    dist = np.maximum(dist, MIN_STOP_PCT * price)
    return dist, (dist if hard is None else hard)


def prepare(coin: str, interval: str, source: str, sub: bool) -> dict | None:
    df = load_frame(coin, interval, source)
    strat = VegasTrend()
    if len(df) < strat.min_bars + 50:
        return None
    p = strat.prepare(df)
    sigs = []
    for i in np.flatnonzero(strat.candidates(p)):
        if i < strat.min_bars:
            continue
        hit = strat.check(p, int(i))
        if hit is not None:
            sigs.append((int(i), 1 if hit[0] == LONG else -1))
    tr = ind.true_range(p)
    arr = {k: p[k].to_numpy(float) for k in ("open", "high", "low", "close", "atr", "hist", "ema12")}
    arr["time"] = p["time"].to_numpy(dtype="datetime64[ns]")
    for n in (7, 28, 50, 100):
        arr[f"atr{n}"] = tr.ewm(alpha=1 / n, adjust=False).mean().to_numpy(float)
    arr["chan_top"] = np.maximum(p["ema144"], p["ema169"]).to_numpy(float)
    arr["chan_bot"] = np.minimum(p["ema144"], p["ema169"]).to_numpy(float)
    for n in (5, 10, 20):
        arr[f"lo{n}"] = p["low"].rolling(n).min().to_numpy(float)
        arr[f"hi{n}"] = p["high"].rolling(n).max().to_numpy(float)
    # 趨勢一致的隨機進場：和訊號同方向、同樣在通道外 3 ATR 以內的 K 棒（不看 MACD/RSI）
    c, top, bot, at = p["close"], arr["chan_top"], arr["chan_bot"], p["atr"]
    up = ((c > top) & (p["ema12"] > top) & (p["ema144"] > p["ema169"]) & (c - top <= 3 * at)).to_numpy().copy()
    dn = ((c < bot) & (p["ema12"] < bot) & (p["ema144"] < p["ema169"]) & (bot - c <= 3 * at)).to_numpy().copy()
    up[: strat.min_bars] = False
    dn[: strat.min_bars] = False
    up[-2:] = False
    dn[-2:] = False
    rng = np.random.default_rng(zlib.crc32(f"{coin}{interval}".encode()))
    rnd = []
    pool = {1: np.flatnonzero(up), -1: np.flatnonzero(dn)}
    for _, d in sigs:
        if len(pool[d]):
            rnd += [(int(x), d) for x in rng.choice(pool[d], N_RANDOM)]
    data = {"coin": coin, "p": arr, "sigs": sigs, "rnd": rnd, "n": len(p)}
    if sub:
        sdf = load(coin, SUB[interval][0])
        if sdf.empty:
            return None
        t = sdf["time"].to_numpy(dtype="datetime64[ms]").astype(np.int64)
        pid_time = (t // PARENT_MS[interval]) * PARENT_MS[interval]
        ptime = p["time"].to_numpy(dtype="datetime64[ms]").astype(np.int64)
        keep = np.isin(pid_time, ptime)
        sdf, pid_time = sdf[keep].reset_index(drop=True), pid_time[keep]
        is_close = np.r_[pid_time[1:] != pid_time[:-1], True]
        pidx = np.searchsorted(ptime, pid_time)
        data["sub"] = {"open": sdf["open"].to_numpy(float), "high": sdf["high"].to_numpy(float),
                       "low": sdf["low"].to_numpy(float), "close": sdf["close"].to_numpy(float),
                       "is_close": is_close, "pidx": pidx, "first": np.searchsorted(pidx, np.arange(len(p)))}
    return data


def run_set(data: dict, entries: list, vs: list[dict], interval: str, fee: float, slip: float, use_sub: bool):
    """一組進場（訊號或隨機）跑全部設定。回傳 dict：每個欄位是 (設定數, 進場數) 的陣列。"""
    p = data["p"]
    if not entries:
        return None
    idx = np.array([i for i, _ in entries])
    ds = np.array([d for _, d in entries], dtype=np.int64)
    ok = (idx + 1 < data["n"]) & np.isfinite(p["atr"][idx])
    if use_sub:  # 細 K 線要涵蓋進場那根之後，才比較得出先後
        s = data["sub"]
        f = s["first"][np.minimum(idx + 1, data["n"] - 1)]
        ok &= (f < len(s["pidx"])) & (s["pidx"][np.minimum(f, len(s["pidx"]) - 1)] == idx + 1)
    idx, ds = idx[ok], ds[ok]
    if len(idx) == 0:
        return None
    price = p["close"][idx]
    atr = p["atr"][idx]
    tp_r, portions, trail_k = PLANS[interval]
    if use_sub:
        s = data["sub"]
        op, hi, lo, cl, isc = s["open"], s["high"], s["low"], s["close"], s["is_close"]
        starts = s["first"][idx + 1]
        valid = starts < len(hi)
        starts = np.where(valid, starts, len(hi))
        inv_src = s["pidx"]
    else:
        op, hi, lo, cl = p["open"], p["high"], p["low"], p["close"]
        isc = np.ones(len(hi), dtype=np.bool_)
        starts = idx + 1
        inv_src = np.arange(len(hi))
    res = {k: np.zeros((len(vs), len(idx))) for k in ("r", "hit", "j", "kind", "mae", "risk")}
    res["exit_t"] = np.zeros((len(vs), len(idx)), dtype="datetime64[ns]")
    for vi, v in enumerate(vs):
        risk, hard = stop_arrays(v, p, idx, ds)
        stop0 = price - ds * hard
        unit = risk if v["frame"] == "R" else LIVE_ATR * atr
        tps = np.stack([price + ds * k * unit for k in tp_r], axis=1)
        trails = v.get("trail_k", trail_k) * atr
        etr = v.get("etrail", 0.0) * atr
        softs = risk if v["stop"][0] == "close" else np.zeros(len(idx))
        inv = v.get("inv_mode", 0)
        if inv == 1:
            ilo, ihi = p["chan_top"], p["chan_bot"]
        elif inv == 2:
            ilo, ihi = p["chan_bot"], p["chan_top"]
        elif inv == 3:
            ilo = ihi = p["hist"]
        else:
            ilo = ihi = p["close"]
        ilo, ihi = ilo[inv_src], ihi[inv_src]
        r, hit, j, kind, mae = sim_many(op, hi, lo, cl, isc, starts, ds, price, risk, stop0, tps,
                                        np.array(portions), v.get("protect", 0), trails, fee, slip, MAX_BARS,
                                        v.get("ts_bars", 0), v.get("ts_mode", 0), etr,
                                        v.get("lock_trig", 0.0), v.get("lock_lvl", 0.0), softs, inv, ilo, ihi,
                                        False)
        if use_sub:
            j = data["sub"]["pidx"][np.minimum(j, len(hi) - 1)]
        res["r"][vi], res["hit"][vi], res["j"][vi], res["kind"][vi], res["mae"][vi] = r, hit, j, kind, mae
        res["risk"][vi] = risk / price
        res["exit_t"][vi] = p["time"][j.astype(np.int64)]
    meta = pd.DataFrame({"coin": data["coin"], "i": idx, "side": ds, "time": p["time"][idx + 1],
                         "atr_pct": atr / price})
    meta["ext_atr"] = np.where(ds > 0, price - p["chan_top"][idx], p["chan_bot"][idx] - price) / atr
    meta["atr_ratio"] = atr / p["atr100"][idx]
    meta["candle_atr"] = (p["high"][idx] - p["low"][idx]) / atr
    meta["chan_far_atr"] = np.where(ds > 0, price - p["chan_bot"][idx], p["chan_top"][idx] - price) / atr
    # 診斷：有利/不利方向走到 k 倍 ATR 的第一根
    fav, adv, sfav, sadv = first_passage(hi, lo, isc, starts, ds, price, atr, LEVELS, MAX_BARS)
    return res, meta, (fav, adv, sfav, sadv)


def _work(args):
    coin, interval, fee, slip, source, sub = args
    data = prepare(coin, interval, source, sub)
    if data is None:
        return coin, None
    vs = variants()
    sig = run_set(data, data["sigs"], vs, interval, fee, slip, sub)
    rnd = run_set(data, data["rnd"], vs, interval, fee, slip, sub)
    return coin, (sig, rnd, [str(t) for t in data["p"]["time"][[0, -1]]])


@njit(cache=True)
def busy_filter(coin_id, i, j, kind):
    """同一個幣同時只持有一張：前一張出場那根之前的新訊號跳過（和 exit_backtest 相同）。
    資料結束還沒出場的不算，也不擋後面的訊號。訊號要先依（幣, i）排序。"""
    take = np.zeros(len(i), np.bool_)
    busy = -1
    prev = -1
    for s in range(len(i)):
        if coin_id[s] != prev:
            busy = -1
            prev = coin_id[s]
        if i[s] + 1 < busy or kind[s] == -1:
            continue
        take[s] = True
        busy = j[s]
    return take


def cluster_stats(x: np.ndarray, key: np.ndarray) -> tuple[float, float, float, float, int]:
    """平均、群集穩健標準誤（同一天同方向算一群）、群集等權平均和它的標準誤、群數。"""
    n = len(x)
    if n == 0:
        return np.nan, np.nan, np.nan, np.nan, 0
    s = pd.Series(x).groupby(key)
    sums, cnt = s.sum(), s.count()
    mean = x.mean()
    se = np.sqrt(((sums - cnt * mean) ** 2).sum()) / n
    cm = sums / cnt
    return mean, se, cm.mean(), (cm.std() / np.sqrt(len(cm)) if len(cm) > 1 else np.nan), len(cm)


def summarize(r, kind, hit, bars, t_in, t_out, key, risk_pct, risk=0.0025) -> dict:
    if len(r) == 0:
        return {"trades": 0}
    order = np.argsort(t_out, kind="stable")
    eq = np.cumprod(1 + risk * r[order])
    dd = (eq / np.maximum.accumulate(eq) - 1).min()
    years = max((t_out.max() - t_in.min()) / np.timedelta64(1, "D") / 365.25, 0.1)
    mean, se_cl, cmean, cse, ncl = cluster_stats(r, key)
    wins, losses = r[r > 0].sum(), -r[r <= 0].sum()
    stopped = kind == 1
    # 每日損益（依出場日入帳，沒交易的日子算 0）算年化 Sharpe；可以換算成「把每筆風險調到一樣的回撤」來比較
    day = pd.Series(risk * r, index=pd.DatetimeIndex(t_out).floor("D")).groupby(level=0).sum()
    day = day.reindex(pd.date_range(pd.Timestamp(t_in.min()).floor("D"), day.index.max(), freq="D"), fill_value=0)
    sharpe = day.mean() / day.std() * np.sqrt(365) if day.std() > 0 else np.nan
    return {"sharpe": sharpe, "calmar": ((eq[-1] ** (1 / years) - 1) / -dd) if dd < 0 else np.nan,
        "trades": len(r), "win_rate": (r > 0).mean(), "exp_r": mean, "se_r": r.std() / np.sqrt(len(r)),
        "se_r_cluster": se_cl, "cluster_r": cmean, "cluster_se": cse, "clusters": ncl,
        "pf": wins / losses if losses else np.inf, "tp1_rate": (hit >= 1).mean(), "stop_rate": stopped.mean(),
        "close_exit_rate": (kind == 4).mean(), "avg_stop_loss_r": r[stopped].mean() if stopped.any() else np.nan,
        "worst_r": r.min(), "p01_r": np.quantile(r, 0.01),
        "total_pct": (eq[-1] - 1) * 100, "ann_pct": (eq[-1] ** (1 / years) - 1) * 100, "max_dd_pct": dd * 100,
        "median_stop_pct": np.median(risk_pct) * 100, "avg_bars": bars.mean(),
    }


def periods(t: pd.Series) -> dict[str, np.ndarray]:
    return {"全期": np.ones(len(t), bool), "2020-23": (t < SPLIT).to_numpy(), "2024+": (t >= SPLIT).to_numpy()}


def cluster_key(meta: pd.DataFrame) -> np.ndarray:
    return (meta["time"].dt.floor("D").astype(str) + meta["side"].astype(str)).to_numpy()


def make_grid(meta, res, rmeta, rres, names, groups) -> pd.DataFrame:
    """每個設定 × 期間：帳戶績效（同幣同時一張）＋ 同一批訊號和實盤的成對差 ＋ 隨機進場的成對差。"""
    coin_id = pd.factorize(meta["coin"])[0]
    key, rkey = cluster_key(meta), cluster_key(rmeta)
    t_in = meta["time"].to_numpy(dtype="datetime64[ns]")
    per, rper = periods(meta["time"]), periods(rmeta["time"])
    base_r, base_k = res["r"][0], res["kind"][0]
    rows = []
    for vi, name in enumerate(names):
        r, kind = res["r"][vi], res["kind"][vi].astype(np.int64)
        j = res["j"][vi].astype(np.int64)
        take = busy_filter(coin_id, meta["i"].to_numpy(np.int64), j, kind)
        bars = j - meta["i"].to_numpy()
        both = (kind != -1) & (base_k != -1)
        rboth = (rres["kind"][vi] != -1) & (rres["kind"][0] != -1)
        for pname, pm in per.items():
            m = take & pm
            s = summarize(r[m], kind[m], res["hit"][vi][m], bars[m], t_in[m], res["exit_t"][vi][m], key[m],
                          res["risk"][vi][m])
            dm = both & pm
            d_mean, d_se, d_cm, d_cse, _ = cluster_stats(r[dm] - base_r[dm], key[dm])
            a_mean, a_se, a_cm, a_cse, _ = cluster_stats(r[dm], key[dm])
            rm = rboth & rper[pname]
            rr = rres["r"][vi][rm]
            q_mean, q_se, _, _, _ = cluster_stats(rr - rres["r"][0][rm], rkey[rm])
            rows.append({"stop": name, "group": groups[vi], "period": pname, **s,
                         "all_signals": int(dm.sum()), "all_exp_r": a_mean, "all_cluster_r": a_cm,
                         "all_cluster_se": a_cse, "paired_diff": d_mean, "paired_se": d_se,
                         "paired_cluster_diff": d_cm, "paired_cluster_se": d_cse,
                         "random_exp_r": rr.mean() if len(rr) else np.nan, "random_diff": q_mean,
                         "random_diff_se": q_se, "signal_minus_random": a_mean - (rr.mean() if len(rr) else np.nan)})
    return pd.DataFrame(rows)


# ---------- 第二步：讀第一步的輸出做診斷和檢驗（python -m research.stop_research --interval 4H --analyze） ----------

def race_table(tag: str, out: Path) -> pd.DataFrame:
    """止損 x 倍 ATR、目標 y 倍 ATR，哪個先到（同一根 K 棒都碰到算止損先到）。訊號 vs 趨勢一致的隨機進場。"""
    z = np.load(out / f"stop_paths_{tag}.npz")
    lv = [round(float(x), 2) for x in z["levels"]]
    rows = []
    for src, pre in (("訊號", ""), ("隨機", "rnd_")):
        fav, adv = z[pre + "sfav"], z[pre + "sadv"]
        for x in (0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0):
            a = adv[:, lv.index(x)]
            row = {"entries": src, "stop_atr": x}
            for y in (1.0, 1.5, 3.0, 4.5):
                f = fav[:, lv.index(y)]
                p_ = ((f > 0) & ((a == 0) | (f < a))).mean()
                row[f"p_target_{y}"] = p_
                row[f"edge_r_{y}"] = (p_ * y - (1 - p_) * x) / x  # 不含手續費
            rows.append(row)
        f15, a15 = fav[:, lv.index(1.5)], adv[:, lv.index(1.5)]
        stopped = (a15 > 0) & ((f15 == 0) | (a15 <= f15))
        for y in (1.5, 3.0):
            fy = fav[:, lv.index(y)]
            rows.append({"entries": src, "stop_atr": 1.5, "note": f"止損後 {MAX_BARS} 根內回到 +{y}ATR 的比例",
                         "value": (stopped & (fy > 0) & (fy >= a15))[stopped].mean()})
        reach = f15 > 0
        for x in (1.0, 1.5, 2.0):
            ax = adv[:, lv.index(x)]
            rows.append({"entries": src, "stop_atr": x, "note": "最後碰到 +1.5ATR 的單裡，先逆向超過 x ATR 的比例",
                         "value": (reach & (ax > 0) & (ax < f15))[reach].mean()})
    return pd.DataFrame(rows)


def selection_check(grid: pd.DataFrame) -> pd.DataFrame:
    """用一段時期挑 Sharpe / 每筆 R 前 5 名，看另一段時期有沒有贏實盤（兩個方向都做）。"""
    a = grid[grid.period == "2020-23"].set_index("stop")
    b = grid[grid.period == "2024+"].set_index("stop")
    live = grid["stop"].iloc[0]
    rows = []
    for crit in ("sharpe", "exp_r"):
        for lab, x, y in (("挑 2020-23 → 看 2024+", a, b), ("挑 2024+ → 看 2020-23", b, a)):
            top = x[crit].sort_values(ascending=False).index[:5]
            rows.append({"criterion": crit, "direction": lab, "rank_corr": a[crit].rank().corr(b[crit].rank()),
                         "top1": top[0], "top1_pick": x.loc[top[0], crit], "top1_test": y.loc[top[0], crit],
                         "top5_test_avg": y.loc[top, crit].mean(), "top5_beat_live": int((y.loc[top, crit] > y.loc[live, crit]).sum()),
                         "live_test": y.loc[live, crit], "live_rank_in_pick": int(x[crit].rank(ascending=False)[live])})
    return pd.DataFrame(rows)


def daily_pnl(z, meta, vi: int, risk: float = 0.0025) -> pd.Series:
    coin_id = pd.factorize(meta["coin"])[0]
    r, kind, j = z["r"][vi], z["kind"][vi].astype(np.int64), z["j"][vi].astype(np.int64)
    take = busy_filter(coin_id, meta["i"].to_numpy(np.int64), j, kind)
    return pd.Series(risk * r[take], index=pd.DatetimeIndex(z["exit_t"][vi][take]).floor("D")).groupby(level=0).sum()


def sharpe_bootstrap(z, meta, names: list[str], picks: list[str], n: int = 2000, block: int = 30) -> pd.DataFrame:
    """同一段每日帳戶損益，30 天一塊重抽，算（設定 − 實盤）Sharpe 差的 90% 區間。"""
    def sh(x):
        return x.mean() / x.std() * np.sqrt(365)
    rng = np.random.default_rng(7)
    base = daily_pnl(z, meta, 0)
    split = SPLIT.tz_localize(None)
    rows = []
    for nm in picks:
        v = daily_pnl(z, meta, names.index(nm))
        idx = pd.date_range(min(base.index.min(), v.index.min()), max(base.index.max(), v.index.max()), freq="D")
        a0, b0 = base.reindex(idx, fill_value=0).to_numpy(), v.reindex(idx, fill_value=0).to_numpy()
        for per, m in (("全期", np.ones(len(idx), bool)), ("2020-23", idx < split), ("2024+", idx >= split)):
            a, b = a0[m], b0[m]
            T = len(a)
            d = []
            for _ in range(n):
                ii = (rng.integers(0, T - block, size=T // block + 1)[:, None] + np.arange(block)).ravel()[:T]
                d.append(sh(b[ii]) - sh(a[ii]))
            d = np.array(d)
            rows.append({"stop": nm, "period": per, "sharpe_diff": sh(b) - sh(a), "ci05": np.quantile(d, 0.05),
                         "ci95": np.quantile(d, 0.95), "p_positive": (d > 0).mean()})
    return pd.DataFrame(rows)


def _reentry_coin(args):
    """被止損（TP1 前）後 N 根內收盤回到原進場價之外，就用同樣規則再進場一次。"""
    coin, interval, slip = args
    data = prepare(coin, interval, "binance", False)
    if data is None:
        return None
    p = data["p"]
    got = run_set(data, data["sigs"], variants()[:1], interval, 0.0005, slip, False)
    if got is None:
        return None
    res, meta, _ = got
    tp_r, portions, trail_k = PLANS[interval]
    op, hi, lo, cl = p["open"], p["high"], p["low"], p["close"]
    isc = np.ones(len(hi), np.bool_)
    rows = []
    for s in range(len(meta)):
        if res["kind"][0][s] != 1:
            continue
        i, d, j = int(meta["i"].iat[s]), int(meta["side"].iat[s]), int(res["j"][0][s])
        entry = cl[i]
        for n_ in (6, 12, 24):
            k = next((kk for kk in range(j, min(j + n_ + 1, len(cl) - 1)) if d * (cl[kk] - entry) > 0), None)
            if k is None:
                continue
            price = cl[k]
            risk = max(LIVE_ATR * p["atr"][k], MIN_STOP_PCT * price)
            tps = np.array([[price + d * x * risk for x in tp_r]])
            r, _, jj, kind, _ = sim_many(op, hi, lo, cl, isc, np.array([k + 1]), np.array([d]), np.array([price]),
                                         np.array([risk]), np.array([price - d * risk]), tps, np.array(portions), 0,
                                         np.array([trail_k * p["atr"][k]]), 0.0005, slip, MAX_BARS, 0, 0,
                                         np.array([0.0]), 0.0, 0.0, np.array([0.0]), 0, cl, cl, False)
            if kind[0] != -1:
                rows.append({"coin": coin, "window": n_, "time": p["time"][k + 1], "side": d, "r": r[0]})
    return rows


def reentry_study(interval: str, slip: float, workers: int) -> pd.DataFrame:
    coins = [c for c in TOP100 if c.isascii()]
    rows = []
    with ProcessPoolExecutor(workers) as ex:
        for got in ex.map(_reentry_coin, [(c, interval, slip) for c in coins]):
            rows += got or []
    df = pd.DataFrame(rows)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    key = cluster_key(df)
    out = []
    for n_, g in df.groupby("window"):
        m0 = (df["window"] == n_).to_numpy()
        for per, pm in periods(df["time"]).items():
            m = m0 & pm
            mean, se, cm, cse, ncl = cluster_stats(df["r"].to_numpy()[m], key[m])
            out.append({"window": n_, "period": per, "trades": int(m.sum()), "exp_r": mean, "se_r_cluster": se,
                        "cluster_r": cm, "cluster_se": cse})
        yr = g.groupby(g["time"].dt.year)["r"].mean()
        out.append({"window": n_, "period": "by_year", **{str(y): v for y, v in yr.items()}})
    return pd.DataFrame(out)


def analyze(interval: str, out: Path, workers: int) -> None:
    pd.set_option("display.width", 250)
    tag = interval
    grid = pd.read_csv(out / f"stop_grid_{tag}.csv")
    z = np.load(out / f"stop_signals_{tag}.npz", allow_pickle=True)
    meta = pd.read_csv(out / f"stop_signals_meta_{tag}.csv.gz", parse_dates=["time"])
    names = list(z["names"])
    race = race_table(tag, out)
    race.to_csv(out / f"stop_race_{tag}.csv", index=False)
    sel = pd.concat([selection_check(pd.read_csv(f)).assign(scenario=f.stem.replace("stop_grid_", ""))
                     for f in sorted(out.glob(f"stop_grid_{interval}*.csv")) if "pionex" not in f.stem])
    sel.to_csv(out / f"stop_selection_{tag}.csv", index=False)
    picks = [n for n in ("ATR1.75", "ATR2.0", "ATR1.0 TP價固定", "ATR1.25 TP價固定", "ATR1.75 TP價固定", "ATR2.0 TP價固定",
                         "1.5×max(ATR14,ATR50)", "收盤破 1.5ATR 出場，硬止損 2.5ATR", "TP1 後止損不動",
                         "TP1 後止損移到 +0.25R", "最後一段追蹤 3.5ATR") if n in names]
    boots = []
    for f in sorted(out.glob(f"stop_signals_{interval}*.npz")):
        if "pionex" in f.stem:  # 派網期間太短，不分兩段
            continue
        sc = f.stem.replace("stop_signals_", "")
        zz = np.load(f, allow_pickle=True)
        mm = pd.read_csv(out / f"stop_signals_meta_{sc}.csv.gz", parse_dates=["time"])
        boots.append(sharpe_bootstrap(zz, mm, list(zz["names"]), picks).assign(scenario=sc))
    boot = pd.concat(boots)
    boot.to_csv(out / f"stop_bootstrap_{tag}.csv", index=False)
    re = reentry_study(interval, 0.0, workers)
    re.to_csv(out / f"stop_reentry_{tag}.csv", index=False)
    print(race.round(3).to_string())
    print(sel.round(3).to_string())
    print(boot.round(3).to_string())
    print(re.round(3).to_string())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", default="4H", choices=["4H", "2H"])
    ap.add_argument("--fee", type=float, default=0.0005, help="每邊手續費")
    ap.add_argument("--slip", type=float, default=0.0, help="進場和止損/收盤出場另外加的滑價（每邊），TP 掛單不加")
    ap.add_argument("--source", default="binance", choices=["binance", "pionex"])
    ap.add_argument("--sub", action="store_true", help="用更細的 K 線判斷同一根 K 棒裡的先後（只限幣安）")
    ap.add_argument("--coins", nargs="*")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="output")
    ap.add_argument("--analyze", action="store_true", help="讀已經跑好的結果做診斷、挑選檢驗、Sharpe 重抽和再進場")
    args = ap.parse_args()
    if args.analyze:
        analyze(args.interval, Path(args.out), args.workers)
        return 0
    coins = args.coins or [c for c in TOP100 if c.isascii()]
    vs = variants()
    names = [v["name"] for v in vs]
    groups = [v["group"] for v in vs]
    parts = {"sig": [], "rnd": []}
    metas = {"sig": [], "rnd": []}
    paths = {"sig": [], "rnd": []}
    jobs = [(c, args.interval, args.fee, args.slip, args.source, args.sub) for c in coins]
    with ProcessPoolExecutor(args.workers) as ex:
        for coin, got in ex.map(_work, jobs):
            if got is None:
                continue
            sig, rnd, _ = got
            for key, val in (("sig", sig), ("rnd", rnd)):
                if val is None:
                    continue
                res, meta, fp = val
                parts[key].append(res)
                metas[key].append(meta)
                paths[key].append(fp)
            print(coin, file=sys.stderr)
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    tag = args.interval + ("_sub" if args.sub else "") + (f"_slip{args.slip}" if args.slip else "") \
        + ("_pionex" if args.source == "pionex" else "")
    cat = {}
    for key in ("sig", "rnd"):
        meta = pd.concat(metas[key], ignore_index=True)
        meta["time"] = pd.to_datetime(meta["time"], utc=True)
        res = {f: np.concatenate([p_[f] for p_ in parts[key]], axis=1) for f in parts[key][0]}
        cat[key] = (meta, res)
    meta, res = cat["sig"]
    rmeta, rres = cat["rnd"]
    np.savez_compressed(out / f"stop_signals_{tag}.npz", names=np.array(names), groups=np.array(groups),
                        **{k: v for k, v in res.items()}, rnd_r=rres["r"], rnd_kind=rres["kind"])
    meta.to_csv(out / f"stop_signals_meta_{tag}.csv.gz", index=False)
    rmeta.to_csv(out / f"stop_random_meta_{tag}.csv.gz", index=False)
    fp = [np.concatenate([x[k] for x in paths["sig"]]) for k in range(4)]
    rp = [np.concatenate([x[k] for x in paths["rnd"]]) for k in range(4)]
    np.savez_compressed(out / f"stop_paths_{tag}.npz", levels=LEVELS, fav=fp[0], adv=fp[1], sfav=fp[2], sadv=fp[3],
                        rnd_fav=rp[0], rnd_adv=rp[1], rnd_sfav=rp[2], rnd_sadv=rp[3])
    grid = make_grid(meta, res, rmeta, rres, names, groups)
    grid.to_csv(out / f"stop_grid_{tag}.csv", index=False)
    show = grid[grid["period"] != "全期"].pivot(index="stop", columns="period", values=["exp_r", "paired_diff"])
    print(show.loc[names].round(3).to_string())
    print(json.dumps({"signals": len(meta), "random": len(rmeta), "variants": len(vs)}), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
