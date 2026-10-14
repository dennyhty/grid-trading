"""vegas_trend 進場條件與選幣研究（只做分析，不改實盤規則）。

python -m research.entry_research --interval 4H [--source pionex] [--slip 0.0005]   第一步：訊號、特徵、實盤出場的 R
python -m research.entry_research --interval 4H --analyze                         第二步：分組、過濾、挑選驗證、選幣
python -m research.entry_research --interval 4H --timing                          進場時機：延遲、回踩掛單、等下一根確認
python -m research.entry_research --interval 4H --delay                           收到訊號晚幾分鐘下單：0/1/2/3/5/10/15 分鐘（1 分鐘 K 線）

需要 numba（和 stop_research.py 共用模擬程式）。

做法：
- 候選 = 趨勢條件成立（價格和 EMA12 在通道同一側、通道方向一致）且 MACD 柱翻轉的 K 棒，不管 RSI 和離通道多遠；
  「實盤」= 再加上 RSI 54-70/30-46、離通道 3 ATR 以內，和 VegasTrend.check 逐筆相同。
- 每個候選都用實盤的出場（4H：1R 平一半、保本、3 ATR 追蹤；2H：1/3/5R 各平 1/4、最後 1/4 追蹤 3 ATR）各自模擬一次。
- 每個訊號在進場那根 K 棒收盤時算特徵：趨勢強度、波動、K 棒形態、成交量、動能、大週期趨勢、長通道（EMA576/676）、
  BTC 和大盤狀態、同時出訊號的幣數、資金費率、幣的強弱排名、成交額、和 BTC 的相關、這個幣和整個策略最近的表現。
  全部只用當下已經知道的資料。
- 第二步用 2020-23 定分組門檻，看 2024 年後；再用一段挑過濾條件、另一段驗證（兩個方向都做）。

輸出（output/）：
- entry_signals_{週期}{標記}.csv.gz：每個候選的特徵、是否實盤訊號、實盤出場的 R、出場時間等。
- entry_quintiles_{週期}.csv、entry_filters_{週期}.csv、entry_selection_{週期}.csv、entry_model_{週期}.csv、
  entry_coins_{週期}.csv：第二步的結果。
- entry_timing_{週期}.csv：進場時機。
- entry_delay_{週期}.csv：晚幾分鐘下單（訊號收盤後的前 15 分鐘用幣安 1 分鐘 K 線，之後用 15 分鐘 K 線）。
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import zipfile
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from grid_trading import indicators as ind
from grid_trading.binance_history import FILES, binance_symbols, load
from grid_trading.signals.strategy import LONG
from grid_trading.signals.vegas_trend import VegasTrend
from grid_trading.universe import TOP100
from research.stop_research import (LIVE_ATR, MAX_BARS, MIN_STOP_PCT, PARENT_MS, PLANS, SPLIT, busy_filter, cluster_key,
                           cluster_stats, load_frame, periods, sim_many, summarize)

BARS_PER_DAY = {"4H": 6, "2H": 12}
FUNDING_DIR = Path("data/binance/funding")
MAX_EXT = 6.0  # 候選最多離通道幾倍 ATR（實盤是 3）


def run_length(state: np.ndarray) -> np.ndarray:
    """state 連續為 True 的第幾根（不成立時是 0）。"""
    out = np.zeros(len(state), np.int64)
    k = 0
    for i, s in enumerate(state):
        k = k + 1 if s else 0
        out[i] = k
    return out


def simulate(p: dict, idx: np.ndarray, ds: np.ndarray, interval: str, fee: float, slip: float,
             entry: np.ndarray | None = None, starts: np.ndarray | None = None, ref: np.ndarray | None = None) -> dict:
    """用實盤出場模擬每個進場（每個各自一次）。entry/starts 不給就是訊號 K 棒收盤進場、下一根開始算。
    ref 給的話，止損和 TP 用 ref 價格算（＝照網站上訊號的價格掛單），1R 是實際成交價到止損的距離。"""
    price = p["close"][idx] if entry is None else entry
    atr = p["atr"][idx]
    tp_r, portions, trail_k = PLANS[interval]
    if ref is None:
        risk = np.maximum(LIVE_ATR * atr, MIN_STOP_PCT * price)
        stop0 = price - ds * risk
        tps = np.stack([price + ds * k * risk for k in tp_r], axis=1)
    else:
        r0 = np.maximum(LIVE_ATR * atr, MIN_STOP_PCT * ref)
        stop0 = ref - ds * r0
        tps = np.stack([ref + ds * k * r0 for k in tp_r], axis=1)
        risk = np.maximum(ds * (price - stop0), MIN_STOP_PCT * price)
    zero = np.zeros(len(idx))
    hi, lo, cl, op = p["high"], p["low"], p["close"], p["open"]
    isc = p.get("is_close", np.ones(len(hi), np.bool_))
    st = idx + 1 if starts is None else starts
    r, hit, j, kind, mae = sim_many(op, hi, lo, cl, isc, st, ds.astype(np.int64), price, risk, stop0, tps,
                                    np.array(portions), 0, trail_k * atr, fee, slip, MAX_BARS, 0, 0, zero, 0.0, 0.0,
                                    zero, 0, cl, cl, False)
    return {"r": r, "hit": hit, "j": j, "kind": kind, "risk_pct": risk / price}


def coin_frame(coin: str, interval: str, source: str):
    df = load_frame(coin, interval, source)
    strat = VegasTrend()
    if len(df) < strat.min_bars + 50:
        return None, None
    return strat.prepare(df).reset_index(drop=True), strat


def _work(args):
    coin, interval, fee, slip, source = args
    p, strat = coin_frame(coin, interval, source)
    if p is None:
        return None
    bpd = BARS_PER_DAY[interval]
    c, o, h, l, v = (p[k].to_numpy(float) for k in ("close", "open", "high", "low", "volume"))
    atr = p["atr"].to_numpy(float)
    e12, e144, e169 = (p[k].to_numpy(float) for k in ("ema12", "ema144", "ema169"))
    top, bot = np.maximum(e144, e169), np.minimum(e144, e169)
    up = (c > top) & (e12 > top) & (e144 > e169)
    dn = (c < bot) & (e12 < bot) & (e144 < e169)
    hist = p["hist"].to_numpy(float)
    prev = np.r_[np.nan, hist[:-1]]
    flip_up = (prev <= 0) & (hist > 0)
    flip_dn = (prev >= 0) & (hist < 0)
    n = len(p)
    ok = (np.arange(n) >= strat.min_bars) & (np.arange(n) + 1 < n) & np.isfinite(atr) & (atr > 0)
    side = np.where(up & flip_up & ok, 1, np.where(dn & flip_dn & ok, -1, 0))
    ext = np.where(side > 0, c - top, bot - c) / atr
    idx = np.flatnonzero((side != 0) & (ext <= MAX_EXT))
    if len(idx) == 0:
        return None
    ds = side[idx]
    rsi = p["rsi"].to_numpy(float)
    rsi_ok = np.where(ds > 0, (rsi[idx] >= 54) & (rsi[idx] <= 70), (rsi[idx] >= 30) & (rsi[idx] <= 46))
    ext_ok = ext[idx] <= 3.0
    arr = {"open": o, "high": h, "low": l, "close": c, "atr": atr}
    sim = simulate(arr, idx, ds, interval, fee, slip)

    # ---- 特徵（都在訊號 K 棒收盤時就知道） ----
    s = pd.Series
    tr = ind.true_range(p)
    atr100 = tr.ewm(alpha=1 / 100, adjust=False).mean().to_numpy(float)
    adx = ind.adx(p).to_numpy(float)
    e576, e676 = ind.ema(p["close"], 576).to_numpy(float), ind.ema(p["close"], 676).to_numpy(float)
    ema_d50 = ind.ema(p["close"], 50 * bpd).to_numpy(float)
    ema_d200 = ind.ema(p["close"], 200 * bpd).to_numpy(float)
    run_up, run_dn = run_length(up), run_length(dn)
    # 這段趨勢裡第幾次同向 MACD 翻轉（含這次）
    nth = np.zeros(n, np.int64)
    k_up = k_dn = 0
    for i in range(n):
        k_up = (k_up + (1 if flip_up[i] else 0)) if up[i] else 0
        k_dn = (k_dn + (1 if flip_dn[i] else 0)) if dn[i] else 0
        nth[i] = k_up if up[i] else (k_dn if dn[i] else 0)
    lo10 = s(l).rolling(10).min().to_numpy()
    hi10 = s(h).rolling(10).max().to_numpy()
    hi20 = s(h).rolling(20).max().to_numpy()
    lo20 = s(l).rolling(20).min().to_numpy()
    vol_ma = s(v).rolling(20).mean().shift(1).to_numpy()
    qvol30 = s(v * c).rolling(30 * bpd).mean().to_numpy()
    rng = np.where(h - l > 0, h - l, np.nan)
    t = p["time"]
    close_time = t + pd.Timedelta(milliseconds=PARENT_MS[interval])
    f = pd.DataFrame({
        "coin": coin, "i": idx, "side": ds, "time": t.iloc[idx + 1].to_numpy(), "signal_close": close_time.iloc[idx].to_numpy(),
        "live": rsi_ok & ext_ok, "rsi_ok": rsi_ok, "ext_ok": ext_ok,
        "r": sim["r"], "hit": sim["hit"], "j": sim["j"], "kind": sim["kind"], "risk_pct": sim["risk_pct"],
        "exit_t": t.iloc[np.minimum(sim["j"], n - 1).astype(int)].to_numpy(),
        "ext_atr": ext[idx],
        "rsi_o": np.where(ds > 0, rsi[idx], 100 - rsi[idx]),
        "hist_atr": np.abs(hist[idx]) / atr[idx],
        "macd_o": ds * p["macd"].to_numpy(float)[idx] / atr[idx],
        "adx": adx[idx],
        "atr_pct": atr[idx] / c[idx] * 100,
        "atr_ratio": atr[idx] / atr100[idx],
        "chan_slope": ds * (e144[idx] - e144[idx - 20]) / atr[idx],
        "chan_width": np.abs(e144[idx] - e169[idx]) / atr[idx],
        "long_chan": ds * ((e144[idx] + e169[idx]) / 2 - (e576[idx] + e676[idx]) / 2) / atr[idx],
        "trend_age": np.where(ds > 0, run_up[idx], run_dn[idx]) / bpd,
        "nth_flip": nth[idx],
        "pullback": np.where(ds > 0, lo10[idx] - top[idx], bot[idx] - hi10[idx]) / atr[idx],
        "from_extreme": np.where(ds > 0, hi20[idx] - c[idx], c[idx] - lo20[idx]) / atr[idx],
        "body_o": ds * (c[idx] - o[idx]) / atr[idx],
        "range_atr": (h[idx] - l[idx]) / atr[idx],
        "close_pos": np.where(ds > 0, c[idx] - l[idx], h[idx] - c[idx]) / rng[idx],
        "vol_ratio": v[idx] / vol_ma[idx],
        "ret7_o": ds * (c[idx] / c[np.maximum(idx - 7 * bpd, 0)] - 1) * 100,
        "ret30_o": ds * (c[idx] / c[np.maximum(idx - 30 * bpd, 0)] - 1) * 100,
        "htf50_o": ds * (c[idx] / ema_d50[idx] - 1) * 100,
        "htf200_o": ds * (c[idx] / ema_d200[idx] - 1) * 100,
        "hour": close_time.iloc[idx].dt.hour.to_numpy(),
        "weekday": close_time.iloc[idx].dt.weekday.to_numpy(),
        "age_days": (t.iloc[idx] - t.iloc[0]).dt.days.to_numpy(),
    })
    # 長週期的均線要有足夠的 K 棒才準，不夠的當缺值
    for col, need in (("long_chan", 676), ("htf50_o", 50 * bpd), ("htf200_o", 200 * bpd), ("atr_ratio", 100)):
        f[col] = np.where(idx >= need, f[col], np.nan)
    f["ret7_o"] = np.where(idx >= 7 * bpd, f["ret7_o"], np.nan)
    f["ret30_o"] = np.where(idx >= 30 * bpd, f["ret30_o"], np.nan)
    state = np.where(up, 1, np.where(dn, -1, 0))
    panel = pd.DataFrame({"time": t.to_numpy(), "close": c, "state": state, "qvol30": qvol30})
    return coin, f, panel


def load_funding(coin: str) -> pd.DataFrame | None:
    fp = FUNDING_DIR / f"{coin}.csv.gz"
    if not fp.exists():
        return None
    d = pd.read_csv(fp).sort_values("calc_time")
    d["time"] = pd.to_datetime(d["calc_time"], unit="ms", utc=True)
    d["daily"] = d["last_funding_rate"] * 24 / d["funding_interval_hours"].clip(lower=1) * 100  # 每日 %
    return d[["time", "daily"]]


def add_cross_features(f: pd.DataFrame, panels: dict[str, pd.DataFrame], interval: str) -> pd.DataFrame:
    """需要所有幣一起算的特徵：大盤、BTC、強弱排名、成交額排名、和 BTC 的相關、同時出訊號數、資金費率、最近表現。"""
    bpd = BARS_PER_DAY[interval]
    close = pd.DataFrame({k: v.set_index("time")["close"] for k, v in panels.items()}).sort_index()
    state = pd.DataFrame({k: v.set_index("time")["state"] for k, v in panels.items()}).reindex(close.index)
    qvol = pd.DataFrame({k: v.set_index("time")["qvol30"] for k, v in panels.items()}).reindex(close.index)
    active = close.notna()
    breadth = ((state == 1).sum(axis=1) - (state == -1).sum(axis=1)) / active.sum(axis=1)
    r7 = close / close.shift(7 * bpd) - 1
    r30 = close / close.shift(30 * bpd) - 1
    rk7, rk30 = r7.rank(axis=1, pct=True), r30.rank(axis=1, pct=True)
    rkq = qvol.rank(axis=1, pct=True)
    ret = np.log(close).diff()
    corr = ret.rolling(90 * bpd, min_periods=30 * bpd).corr(ret["BTC"]) if "BTC" in ret else None
    btc = close["BTC"]
    btc_state = state["BTC"]
    btc_r7, btc_r30 = r7["BTC"], r30["BTC"]
    btc_d200 = btc / ind.ema(btc.ffill(), 200 * bpd) - 1
    # 每個訊號在它那根 K 棒（time 是下一根的開盤 = 訊號 K 棒收盤）的位置
    sig_bar = pd.DatetimeIndex(f["time"]) - pd.Timedelta(milliseconds=PARENT_MS[interval])
    pos = close.index.get_indexer(sig_bar)
    col = close.columns.get_indexer(f["coin"])
    sd = f["side"].to_numpy()
    take = lambda df_: df_.to_numpy()[pos, col]
    f["breadth_o"] = sd * breadth.to_numpy()[pos]
    f["btc_state_o"] = sd * btc_state.to_numpy()[pos]
    f["btc_r7_o"] = sd * btc_r7.to_numpy()[pos] * 100
    f["btc_r30_o"] = sd * btc_r30.to_numpy()[pos] * 100
    f["btc_d200_o"] = sd * btc_d200.to_numpy()[pos] * 100
    f["rank7_o"] = np.where(sd > 0, take(rk7), 1 - take(rk7))
    f["rank30_o"] = np.where(sd > 0, take(rk30), 1 - take(rk30))
    f["rs30_o"] = sd * (take(r30) - r30["BTC"].to_numpy()[pos]) * 100
    f["qvol_rank"] = take(rkq)
    f["corr_btc"] = take(corr) if corr is not None else np.nan
    f["n_active"] = active.sum(axis=1).to_numpy()[pos]
    f.loc[f["coin"] == "BTC", ["rs30_o", "corr_btc"]] = np.nan
    # 同一根 K 棒、同方向的實盤訊號有幾個（不含自己）
    lv = f[f["live"]]
    cnt = lv.groupby(["time", "side"]).size()
    f["crowd"] = [cnt.get((a, b), 0) - (1 if x else 0) for a, b, x in zip(f["time"], f["side"], f["live"])]
    # 資金費率：訊號收盤前最後一次，和最近 3 天平均（每日 %，乘上方向：正 = 同方向的人在付錢 = 擁擠）
    f["funding_o"] = np.nan
    f["funding3d_o"] = np.nan
    for coin, g in f.groupby("coin"):
        fd = load_funding(coin)
        if fd is None or fd.empty:
            continue
        fd = fd.set_index("time")["daily"]
        f3 = fd.rolling("3D").mean()
        q = pd.DatetimeIndex(g["signal_close"])
        k = fd.index.searchsorted(q, side="right") - 1
        okk = k >= 0
        last = np.where(okk, fd.to_numpy()[np.maximum(k, 0)], np.nan)
        avg3 = np.where(okk, f3.to_numpy()[np.maximum(k, 0)], np.nan)
        f.loc[g.index, "funding_o"] = g["side"].to_numpy() * last
        f.loc[g.index, "funding3d_o"] = g["side"].to_numpy() * avg3
    # 最近表現：整個策略最近 30 筆、這個幣過去一年（只算訊號時間之前已經出場的實盤訊號）
    lv = f[f["live"] & (f["kind"] != -1)].sort_values("exit_t")
    ex_t = lv["exit_t"].to_numpy(dtype="datetime64[ns]")
    cs = np.r_[0, np.cumsum(lv["r"].to_numpy())]
    q = f["signal_close"].to_numpy(dtype="datetime64[ns]")
    k = np.searchsorted(ex_t, q, side="left")  # 出場那根要在訊號收盤前開盤＝已經收完
    lo_ = np.maximum(k - 30, 0)
    f["strat_recent"] = np.where(k >= 10, (cs[k] - cs[lo_]) / np.maximum(k - lo_, 1), np.nan)
    f["coin_trail"] = np.nan
    f["coin_trail_n"] = 0
    for coin, g in f.groupby("coin"):
        h_ = lv[lv["coin"] == coin]
        if h_.empty:
            continue
        et = h_["exit_t"].to_numpy(dtype="datetime64[ns]")
        rr = h_["r"].to_numpy()
        qq = g["signal_close"].to_numpy(dtype="datetime64[ns]")
        vals, cnts = [], []
        for x in qq:
            m = (et < x) & (et > x - np.timedelta64(365, "D"))
            cnts.append(int(m.sum()))
            vals.append(rr[m].mean() if m.sum() >= 3 else np.nan)
        f.loc[g.index, "coin_trail"] = vals
        f.loc[g.index, "coin_trail_n"] = cnts
    f["mcap_rank"] = f["coin"].map({c: k for k, c in enumerate(TOP100)})
    return f


def build(interval: str, fee: float, slip: float, source: str, workers: int, out: Path) -> pd.DataFrame:
    coins = [c for c in TOP100 if c.isascii()]
    feats, panels = [], {}
    with ProcessPoolExecutor(workers) as ex:
        for got in ex.map(_work, [(c, interval, fee, slip, source) for c in coins]):
            if got is None:
                continue
            coin, f, panel = got
            feats.append(f)
            panels[coin] = panel
            print(coin, len(f), int(f["live"].sum()), file=sys.stderr)
    f = pd.concat(feats, ignore_index=True)
    for col in ("time", "signal_close", "exit_t"):
        f[col] = pd.to_datetime(f[col], utc=True)
    f = add_cross_features(f, panels, interval)
    f = f.sort_values(["coin", "i"]).reset_index(drop=True)
    tag = interval + (f"_slip{slip}" if slip else "") + ("_pionex" if source == "pionex" else "")
    f.to_csv(out / f"entry_signals_{tag}.csv.gz", index=False)
    lv = f[f["live"]]
    print(json.dumps({"candidates": len(f), "live": len(lv), "live_r": round(float(lv["r"].mean()), 4)}), file=sys.stderr)
    return f


# ---------------- 第二步 ----------------

FEATURES = {  # 特徵 → 中文名稱（_o 結尾的都已經乘上方向：越大 = 越順著訊號方向）
    "ext_atr": "離通道幾倍 ATR", "rsi_o": "RSI（空單用 100−RSI）", "hist_atr": "MACD 柱翻轉力道", "macd_o": "MACD 線位置",
    "adx": "ADX 趨勢強度", "atr_pct": "ATR 佔價格 %", "atr_ratio": "ATR 相對長期 ATR", "chan_slope": "通道斜率",
    "chan_width": "通道寬度", "long_chan": "和長通道（EMA576/676）的距離", "trend_age": "趨勢已持續幾天",
    "nth_flip": "這段趨勢第幾次翻轉", "pullback": "回踩離通道多近", "from_extreme": "離前 20 根高/低點",
    "body_o": "訊號 K 棒實體", "range_atr": "訊號 K 棒大小", "close_pos": "收盤在 K 棒的位置", "vol_ratio": "成交量相對 20 根平均",
    "ret7_o": "幣 7 天漲跌", "ret30_o": "幣 30 天漲跌", "htf50_o": "離日線 EMA50", "htf200_o": "離日線 EMA200",
    "breadth_o": "大盤趨勢廣度", "btc_state_o": "BTC 通道方向", "btc_r7_o": "BTC 7 天漲跌", "btc_r30_o": "BTC 30 天漲跌",
    "btc_d200_o": "BTC 離日線 EMA200", "rank7_o": "7 天強弱排名", "rank30_o": "30 天強弱排名", "rs30_o": "30 天相對 BTC 強弱",
    "qvol_rank": "成交額排名", "corr_btc": "和 BTC 的相關", "crowd": "同時同向訊號數", "funding_o": "資金費率（同向付費為正）",
    "funding3d_o": "資金費率 3 天平均", "strat_recent": "策略最近 30 筆平均 R", "coin_trail": "這個幣過去一年平均 R",
    "mcap_rank": "市值排名（越小越大）", "age_days": "上市天數", "hour": "訊號收盤時間（UTC）", "weekday": "星期",
}
FILTERS = {  # 名稱 → 留下哪些五分位（1 最小、5 最大）
    "去掉最低 1/5": (2, 3, 4, 5), "去掉最低 2/5": (3, 4, 5), "去掉最高 1/5": (1, 2, 3, 4), "去掉最高 2/5": (1, 2, 3),
    "去掉頭尾 1/5": (2, 3, 4),
}


def qcut_is(x: pd.Series, is_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """用 2020-23 的五分位門檻把全期分組（1~5，缺值 0）。"""
    edges = np.unique(np.nanquantile(x[is_mask], [0.2, 0.4, 0.6, 0.8]))
    q = np.where(x.isna(), 0, np.searchsorted(edges, x.to_numpy(), side="right") + 1)
    return q, edges


def account(f: pd.DataFrame, m: np.ndarray) -> dict:
    """只做 m 選到的訊號（同一個幣同時一張）的帳戶績效。"""
    g = f[m & (f["kind"] != -1).to_numpy()]
    if g.empty:
        return {"trades": 0}
    cid = pd.factorize(g["coin"])[0]
    take = busy_filter(cid, g["i"].to_numpy(np.int64), g["j"].to_numpy(np.int64), g["kind"].to_numpy(np.int64))
    g = g[take]
    t_in = g["time"].to_numpy(dtype="datetime64[ns]")
    return summarize(g["r"].to_numpy(), g["kind"].to_numpy(), g["hit"].to_numpy(), (g["j"] - g["i"]).to_numpy(),
                     t_in, g["exit_t"].to_numpy(dtype="datetime64[ns]"), cluster_key(g), g["risk_pct"].to_numpy())


def quintile_table(f: pd.DataFrame) -> pd.DataFrame:
    lv = f[f["live"] & (f["kind"] != -1)].reset_index(drop=True)
    per = periods(lv["time"])
    key = cluster_key(lv)
    rows = []
    for feat, name in FEATURES.items():
        if feat not in lv or lv[feat].notna().sum() < 100:
            continue
        q, edges = qcut_is(lv[feat], per["2020-23"])
        for pname, pm in per.items():
            for k in range(1, len(edges) + 2):
                m = pm & (q == k)
                mean, se, cm, cse, ncl = cluster_stats(lv["r"].to_numpy()[m], key[m])
                rows.append({"feature": feat, "name": name, "period": pname, "q": k,
                             "lo": edges[k - 2] if k > 1 else -np.inf, "hi": edges[k - 1] if k <= len(edges) else np.inf,
                             "n": int(m.sum()), "exp_r": mean, "se": se, "cluster_r": cm, "cluster_se": cse,
                             "win_rate": (lv["r"].to_numpy()[m] > 0).mean() if m.any() else np.nan})
            # 單調性：R 對五分位序號的斜率（群集穩健）
            m = pm & (q > 0)
            x, y, kk = q[m].astype(float), lv["r"].to_numpy()[m], key[m]
            xc = x - x.mean()
            b = (xc * y).sum() / (xc ** 2).sum()
            u = pd.Series(xc * (y - y.mean() - b * xc)).groupby(kk).sum()
            rows.append({"feature": feat, "name": name, "period": pname, "q": "slope", "exp_r": b,
                         "se": np.sqrt((u ** 2).sum()) / (xc ** 2).sum(), "n": int(m.sum())})
    return pd.DataFrame(rows)


def filter_table(f: pd.DataFrame) -> pd.DataFrame:
    """每個特徵 × 每種去掉方式：帳戶績效（兩段時期），和被去掉那些訊號的平均 R。"""
    f = f.reset_index(drop=True)
    live = (f["live"] & (f["kind"] != -1)).to_numpy()
    per = periods(f["time"])
    key = cluster_key(f)
    rows = []
    base = {pn: account(f, live & pm) for pn, pm in per.items()}
    for pn in per:
        rows.append({"feature": "live", "name": "實盤（不過濾）", "filter": "—", "period": pn, **base[pn]})
    for feat, name in FEATURES.items():
        if feat not in f or f.loc[live, feat].notna().sum() < 100:
            continue
        q, _ = qcut_is(f[feat], live & per["2020-23"])
        for fname, keep in FILTERS.items():
            km = np.isin(q, keep) | (q == 0)  # 缺值的留著
            for pn, pm in per.items():
                a = account(f, live & km & pm)
                dm = live & ~km & pm
                d_mean, d_se, d_cm, d_cse, _ = cluster_stats(f["r"].to_numpy()[dm], key[dm])
                rows.append({"feature": feat, "name": name, "filter": fname, "period": pn, **a,
                             "kept_share": (live & km & pm).sum() / max((live & pm).sum(), 1),
                             "dropped_r": d_mean, "dropped_se": d_se, "dropped_cluster_r": d_cm, "dropped_cluster_se": d_cse})
    return pd.DataFrame(rows)


def selection(ft: pd.DataFrame) -> pd.DataFrame:
    """用一段挑過濾條件（Sharpe 或群集 R 前 5 名，至少留下 50% 訊號），看另一段。"""
    rows = []
    cand = ft[ft["feature"] != "live"]
    for crit in ("sharpe", "cluster_r", "exp_r"):
        for lab, a, b in (("挑 2020-23 → 看 2024+", "2020-23", "2024+"), ("挑 2024+ → 看 2020-23", "2024+", "2020-23")):
            x = cand[(cand.period == a) & (cand.kept_share >= 0.5)].set_index(["feature", "filter"])
            y = cand[cand.period == b].set_index(["feature", "filter"])
            live_b = ft[(ft.feature == "live") & (ft.period == b)].iloc[0]
            live_a = ft[(ft.feature == "live") & (ft.period == a)].iloc[0]
            top = x[crit].sort_values(ascending=False).index[:5]
            for rank, ix in enumerate(top, 1):
                rows.append({"criterion": crit, "direction": lab, "rank": rank, "feature": ix[0], "filter": ix[1],
                             "name": x.loc[ix, "name"], "pick_value": x.loc[ix, crit], "pick_live": live_a[crit],
                             "test_value": y.loc[ix, crit], "test_live": live_b[crit],
                             "test_trades": y.loc[ix, "trades"], "test_live_trades": live_b["trades"]})
            xr, yr = x[crit], y[crit].reindex(x.index)
            rows.append({"criterion": crit, "direction": lab, "rank": "rank_corr", "pick_value": xr.rank().corr(yr.rank())})
    return pd.DataFrame(rows)


def model_check(f: pd.DataFrame) -> pd.DataFrame:
    """所有特徵一起：在一段用五分位序號做嶺迴歸預測 R，到另一段看預測分數高低的實際 R。"""
    lv = f[f["live"] & (f["kind"] != -1)].reset_index(drop=True)
    per = periods(lv["time"])
    feats = [c for c in FEATURES if c in lv and lv[c].notna().mean() > 0.7]
    rows = []
    key = cluster_key(lv)
    for lab, a, b in (("2020-23 → 2024+", "2020-23", "2024+"), ("2024+ → 2020-23", "2024+", "2020-23")):
        X = np.column_stack([qcut_is(lv[c], per[a])[0] for c in feats]).astype(float)
        X[X == 0] = 3.0  # 缺值當中間
        X = (X - 3.0) / 1.414
        ya = lv["r"].to_numpy()[per[a]]
        Xa = X[per[a]]
        lam = 50.0 * len(feats)
        beta = np.linalg.solve(Xa.T @ Xa + lam * np.eye(len(feats)), Xa.T @ (ya - ya.mean()))
        score = X @ beta
        sb = score[per[b]]
        edges = np.quantile(score[per[a]], [0.2, 0.4, 0.6, 0.8])
        qb = np.searchsorted(edges, sb, side="right") + 1
        rb, kb = lv["r"].to_numpy()[per[b]], key[per[b]]
        for k in range(1, 6):
            m = qb == k
            mean, se, cm, cse, _ = cluster_stats(rb[m], kb[m])
            rows.append({"direction": lab, "score_q": k, "n": int(m.sum()), "exp_r": mean, "se": se,
                         "cluster_r": cm, "cluster_se": cse})
        for c_, w in sorted(zip(feats, beta), key=lambda z: -abs(z[1]))[:8]:
            rows.append({"direction": lab, "score_q": "weight", "feature": c_, "exp_r": w})
        # 只做分數前 60%：帳戶績效
        full = np.zeros(len(lv), bool)
        full[np.flatnonzero(per[b])[qb >= 3]] = True
        acc = account(lv, full)
        base = account(lv, per[b].copy())
        rows.append({"direction": lab, "score_q": "top60_account", **{k: acc.get(k) for k in ("trades", "exp_r", "sharpe", "max_dd_pct", "cluster_r")},
                     "live_sharpe": base.get("sharpe"), "live_exp_r": base.get("exp_r"), "live_cluster_r": base.get("cluster_r")})
    return pd.DataFrame(rows)


def coin_table(f: pd.DataFrame) -> pd.DataFrame:
    """每個幣兩段時期的平均 R：過去好的幣以後會不會比較好。"""
    lv = f[f["live"] & (f["kind"] != -1)]
    per = periods(lv["time"])
    a = lv[per["2020-23"]].groupby("coin")["r"].agg(["mean", "count"])
    b = lv[per["2024+"]].groupby("coin")["r"].agg(["mean", "count"])
    t = a.join(b, lsuffix="_2020_23", rsuffix="_2024", how="outer")
    t["mcap_rank"] = t.index.map({c: k for k, c in enumerate(TOP100)})
    return t.reset_index()


def analyze(interval: str, out: Path, tag: str) -> None:
    f = pd.read_csv(out / f"entry_signals_{tag}.csv.gz", parse_dates=["time", "signal_close", "exit_t"])
    f = f.sort_values(["coin", "i"]).reset_index(drop=True)
    qt = quintile_table(f)
    qt.to_csv(out / f"entry_quintiles_{tag}.csv", index=False)
    ft = filter_table(f)
    ft.to_csv(out / f"entry_filters_{tag}.csv", index=False)
    sel = selection(ft)
    sel.to_csv(out / f"entry_selection_{tag}.csv", index=False)
    mc = model_check(f)
    mc.to_csv(out / f"entry_model_{tag}.csv", index=False)
    coin_table(f).to_csv(out / f"entry_coins_{tag}.csv", index=False)
    pd.set_option("display.width", 250)
    s = qt[qt.q == "slope"].pivot(index="feature", columns="period", values=["exp_r", "se"])
    print(s.round(4).to_string())
    print(sel.round(3).to_string())
    print(mc.round(3).to_string())


# ---------------- 進場時機 ----------------

def _timing_coin(args):
    """實盤訊號：收盤直接進場 vs 延遲 15/30/60 分鐘、回踩掛單、等下一根確認。用 15 分鐘 K 線模擬。"""
    coin, interval, fee, slip = args
    p, strat = coin_frame(coin, interval, "binance")
    if p is None:
        return None
    sdf = load(coin, "15m")
    if sdf.empty:
        return None
    pm = PARENT_MS[interval]
    t15 = sdf["time"].to_numpy(dtype="datetime64[ms]").astype(np.int64)
    pid_time = (t15 // pm) * pm
    ptime = p["time"].to_numpy(dtype="datetime64[ms]").astype(np.int64)
    keep = np.isin(pid_time, ptime)
    sdf, pid_time = sdf[keep].reset_index(drop=True), pid_time[keep]
    pidx = np.searchsorted(ptime, pid_time)
    sub = {k: sdf[k].to_numpy(float) for k in ("open", "high", "low", "close")}
    sub["is_close"] = np.r_[pid_time[1:] != pid_time[:-1], True]
    first = np.searchsorted(pidx, np.arange(len(p)))
    sigs = []
    for i in np.flatnonzero(strat.candidates(p)):
        hit_ = strat.check(p, int(i)) if i >= strat.min_bars else None
        if hit_ is not None:
            sigs.append((int(i), 1 if hit_[0] == LONG else -1))
    if not sigs:
        return None
    idx = np.array([s_[0] for s_ in sigs])
    ds = np.array([s_[1] for s_ in sigs])
    ok = (idx + 2 < len(p)) & (first[np.minimum(idx + 1, len(p) - 1)] < len(pidx))
    ok &= pidx[np.minimum(first[np.minimum(idx + 1, len(p) - 1)], len(pidx) - 1)] == idx + 1
    idx, ds = idx[ok], ds[ok]
    if len(idx) == 0:
        return None
    parr = {"close": p["close"].to_numpy(float), "atr": p["atr"].to_numpy(float)}
    s0 = first[idx + 1]
    arr = {"open": sub["open"], "high": sub["high"], "low": sub["low"], "close": sub["close"], "atr": parr["atr"],
           "is_close": sub["is_close"]}
    price0 = parr["close"][idx]
    rows = []

    def run(name, entry, starts, filled, ref=None):
        e = np.where(filled, entry, price0)
        st = np.where(filled, starts, s0)
        if ref is not None:  # 照訊號價格掛止損：成交價已經在止損外就不進場
            filled = filled & (ds * (e - (price0 - ds * LIVE_ATR * parr["atr"][idx])) > 0)
            e = np.where(filled, e, price0)
        res = simulate(arr, idx, ds, interval, fee, slip, entry=e, starts=st, ref=ref)
        j = pidx[np.minimum(res["j"], len(pidx) - 1)]
        for k_ in range(len(idx)):
            rows.append({"coin": coin, "i": int(idx[k_]), "side": int(ds[k_]), "time": p["time"].iat[idx[k_] + 1],
                         "variant": name, "filled": bool(filled[k_]), "r": res["r"][k_] if filled[k_] else 0.0,
                         "kind": int(res["kind"][k_]), "hit": int(res["hit"][k_]), "j": int(j[k_]),
                         "exit_t": p["time"].iat[min(int(j[k_]), len(p) - 1)], "risk_pct": res["risk_pct"][k_],
                         "entry_move_atr": ds[k_] * (e[k_] - price0[k_]) / parr["atr"][idx[k_]]})

    allt = np.ones(len(idx), bool)
    run("收盤直接進場（實盤）", price0, s0, allt)
    for d_min in (15, 30, 60):
        k = d_min // 15
        st = s0 + k
        okk = (st < len(pidx)) & (pidx[np.minimum(st - 1, len(pidx) - 1)] == idx + 1)
        ep = sub["close"][np.minimum(st - 1, len(pidx) - 1)]
        run(f"延遲 {d_min} 分鐘（止損跟著成交價）", ep, np.minimum(st, len(pidx) - 1), okk)
        run(f"延遲 {d_min} 分鐘（照訊號價格掛止損和 TP）", ep, np.minimum(st, len(pidx) - 1), okk, ref=price0)
    atr_i = parr["atr"][idx]
    for kk in (0.25, 0.5, 1.0):
        for nb in (1, 3, 6):
            lim = price0 - ds * kk * atr_i
            filled = np.zeros(len(idx), bool)
            fst = np.zeros(len(idx), np.int64)
            for q in range(len(idx)):
                a = s0[q]
                end = first[min(idx[q] + 1 + nb, len(p) - 1)] if idx[q] + 1 + nb < len(p) else len(pidx)
                seg_lo = sub["low"][a:end] if ds[q] > 0 else -sub["high"][a:end]
                hit_ = np.flatnonzero(seg_lo <= ds[q] * lim[q])
                if len(hit_):
                    filled[q] = True
                    fst[q] = a + hit_[0]  # 成交那根 15 分鐘 K 棒本身也算（保守：同一根碰到止損就算止損）
            run(f"回踩 {kk} ATR 掛單，{nb} 根內（止損跟著成交價）", lim, fst, filled)
            run(f"回踩 {kk} ATR 掛單，{nb} 根內（止損和 TP 照訊號價格）", lim, fst, filled, ref=price0)
    # 等下一根收盤仍順方向（收盤比訊號收盤更有利）才進場
    nxt = parr["close"][idx + 1]
    conf = ds * (nxt - price0) > 0
    run("等下一根收盤確認", nxt, first[np.minimum(idx + 2, len(p) - 1)], conf & (idx + 2 < len(p)))
    return rows


def timing(interval: str, fee: float, slip: float, workers: int, out: Path) -> pd.DataFrame:
    coins = [c for c in TOP100 if c.isascii()]
    rows = []
    with ProcessPoolExecutor(workers) as ex:
        for got in ex.map(_timing_coin, [(c, interval, fee, slip) for c in coins]):
            rows += got or []
            print(len(rows), file=sys.stderr)
    d = pd.DataFrame(rows)
    d["time"] = pd.to_datetime(d["time"], utc=True)
    d["exit_t"] = pd.to_datetime(d["exit_t"], utc=True)
    base = d[d["variant"] == "收盤直接進場（實盤）"].set_index(["coin", "i"])["r"]
    res = []
    for v, g in d.groupby("variant", sort=False):
        g = g.sort_values(["coin", "i"]).reset_index(drop=True)
        k = cluster_key(g)
        diff = g["r"].to_numpy() - base.reindex(pd.MultiIndex.from_frame(g[["coin", "i"]])).to_numpy()
        for pn, pm in periods(g["time"]).items():
            fm = pm & g["filled"].to_numpy() & (g["kind"] != -1).to_numpy()
            acc = account(g.assign(live=True), fm)
            dm = pm & (g["kind"] != -1).to_numpy()
            d_mean, d_se, d_cm, d_cse, _ = cluster_stats(diff[dm], k[dm])
            res.append({"variant": v, "period": pn, "fill_rate": g["filled"].to_numpy()[pm].mean(),
                        "filled_r": g["r"].to_numpy()[fm].mean(), "entry_move_atr": g["entry_move_atr"].to_numpy()[fm].mean(),
                        "per_signal_diff": d_mean, "per_signal_se": d_se, "per_signal_cluster_diff": d_cm,
                        "per_signal_cluster_se": d_cse, **acc})
    t = pd.DataFrame(res)
    t.to_csv(out / f"entry_timing_{interval}{'_slip' + str(slip) if slip else ''}.csv", index=False)
    print(t[["variant", "period", "fill_rate", "filled_r", "per_signal_diff", "per_signal_se", "trades", "exp_r",
             "sharpe", "max_dd_pct"]].round(3).to_string())
    return t


DELAYS = (0, 1, 2, 3, 5, 10, 15)  # 分鐘
M1_DIR = Path("data/binance/m1")


def _live_signals(coin: str, interval: str):
    """實盤訊號（含 2H 的 ADX 規則）：(K 線, 訊號索引, 方向)。"""
    p, strat = coin_frame(coin, interval, "binance")
    if p is None:
        return None, None, None
    sigs = []
    for i in np.flatnonzero(strat.candidates(p)):
        if i < strat.min_bars or i + 2 >= len(p):
            continue
        hit_ = strat.check(p, int(i))
        if hit_ is not None and strat.accept(p, int(i), interval):
            sigs.append((int(i), 1 if hit_[0] == LONG else -1))
    if not sigs:
        return p, None, None
    return p, np.array([q[0] for q in sigs]), np.array([q[1] for q in sigs])


def _m1_day(coin: str, day: str) -> pd.DataFrame:
    """某個幣某一天的幣安 1 分鐘 K 線（日檔），快取在 data/binance/m1/{幣}/{日期}.csv.gz。"""
    path = M1_DIR / coin / f"{day}.csv.gz"
    if path.exists():
        return pd.read_csv(path)
    import requests
    for sym in reversed(binance_symbols(coin)):
        key = f"data/futures/um/daily/klines/{sym}/1m/{sym}-1m-{day}.zip"
        for _ in range(3):
            try:
                resp = requests.get(FILES + key, timeout=60)
                break
            except requests.RequestException:
                resp = None
        if resp is None or resp.status_code != 200:
            continue
        with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
            raw = z.read(z.namelist()[0]).decode()
        df = pd.read_csv(io.StringIO(raw), header=None, usecols=range(5), names=["time", "open", "high", "low", "close"])
        df = df[pd.to_numeric(df["time"], errors="coerce").notna()].astype(float)
        df["time"] = df["time"].astype("int64")
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        return df
    return pd.DataFrame(columns=["time", "open", "high", "low", "close"])


def _delay_coin(args):
    """某個幣的實盤訊號，晚 0~15 分鐘下單各模擬一次。訊號收盤後前 15 分鐘的那根 15 分鐘 K 棒換成 15 根 1 分鐘 K 棒。"""
    coin, interval, fee, slip = args
    p, idx, ds = _live_signals(coin, interval)
    if idx is None:
        return None
    sdf = load(coin, "15m")
    if sdf.empty:
        return None
    pm = PARENT_MS[interval]
    t15 = sdf["time"].to_numpy(dtype="datetime64[ms]").astype(np.int64)
    pid_time = (t15 // pm) * pm
    ptime = p["time"].to_numpy(dtype="datetime64[ms]").astype(np.int64)
    keep = np.isin(pid_time, ptime)
    sdf, t15, pid_time = sdf[keep].reset_index(drop=True), t15[keep], pid_time[keep]
    pidx = np.searchsorted(ptime, pid_time)
    is_close = np.r_[pid_time[1:] != pid_time[:-1], True]
    first = np.searchsorted(pidx, np.arange(len(p)))
    close_ms = ptime[idx] + pm  # 訊號 K 棒收盤
    s0 = first[idx + 1]
    ok = (s0 < len(t15)) & (t15[np.minimum(s0, len(t15) - 1)] == close_ms) & (s0 + 1 < len(t15))
    ok &= t15[np.minimum(s0 + 1, len(t15) - 1)] == close_ms + 900_000
    # 1 分鐘 K 線：收盤後 15 根都要有
    windows = {}
    for q in np.flatnonzero(ok):
        day = pd.Timestamp(int(close_ms[q]), unit="ms", tz="UTC").strftime("%Y-%m-%d")
        m1 = _m1_day(coin, day)
        w = m1[(m1["time"] >= close_ms[q]) & (m1["time"] < close_ms[q] + 900_000)]
        if len(w) == 15:
            windows[q] = w
    sel = np.array(sorted(windows), dtype=np.int64)
    if len(sel) == 0:
        return None
    idx, ds, s0, close_ms = idx[sel], ds[sel], s0[sel], close_ms[sel]
    # 把每個訊號收盤後的第一根 15 分鐘 K 棒換成 15 根 1 分鐘 K 棒（同一根被兩個訊號用到時只換一次）
    cols = ("open", "high", "low", "close")
    sub = {k: sdf[k].to_numpy(float) for k in cols}
    rep = sorted(set(int(x) for x in s0))
    by_pos = {int(s0[k]): windows[int(sel[k])] for k in range(len(sel))}
    pieces = {k: [] for k in cols}
    isc_p, last = [], 0
    for pos in rep:
        for k in cols:
            pieces[k] += [sub[k][last:pos], by_pos[pos][k].to_numpy(float)]
        isc_p += [is_close[last:pos], np.zeros(15, bool)]
        last = pos + 1
    for k in cols:
        pieces[k].append(sub[k][last:])
    isc_p.append(is_close[last:])
    arr = {k: np.concatenate(pieces[k]) for k in cols}
    arr["is_close"] = np.concatenate(isc_p)
    rep_arr = np.array(rep)
    block = s0 + 14 * np.searchsorted(rep_arr, s0)  # 這個訊號的第一根 1 分鐘 K 棒在新陣列的位置
    atr = p["atr"].to_numpy(float)
    arr["atr"] = atr
    price0 = p["close"].to_numpy(float)[idx]
    rows = []
    for d in DELAYS:
        st = block + d  # d = 15 時就是下一根 15 分鐘 K 棒
        e = arr["open"][st]
        for frame, ref in (("照訊號價格", price0), ("止損跟著成交價", None)):
            filled = np.ones(len(idx), bool)
            if ref is not None:  # 成交價已經在止損外就不進場
                filled = ds * (e - (price0 - ds * LIVE_ATR * atr[idx])) > 0
            ee = np.where(filled, e, price0)
            res = simulate(arr, idx, ds, interval, fee, slip, entry=ee, starts=st, ref=ref)
            for k_ in range(len(idx)):
                rows.append({"coin": coin, "i": int(idx[k_]), "side": int(ds[k_]), "time": p["time"].iat[idx[k_] + 1],
                             "delay": d, "frame": frame, "filled": bool(filled[k_]),
                             "r": res["r"][k_] if filled[k_] else 0.0, "kind": int(res["kind"][k_]),
                             "hit": int(res["hit"][k_]), "j_sub": int(res["j"][k_]), "risk_pct": res["risk_pct"][k_],
                             "move_atr": ds[k_] * (e[k_] - price0[k_]) / atr[idx[k_]],
                             "move_r": ds[k_] * (e[k_] - price0[k_]) / (LIVE_ATR * atr[idx[k_]])})
    # 出場時間：換回原本的 K 棒序號（給帳戶模擬的「同一個幣同時一張」用）
    newpid = np.concatenate([np.concatenate([pidx[(rep[n - 1] + 1) if n else 0:pos], np.full(15, pidx[pos])])
                             for n, pos in enumerate(rep)] + [pidx[rep[-1] + 1:]])
    for r_ in rows:
        j = min(r_.pop("j_sub"), len(newpid) - 1)
        r_["j"] = int(newpid[j])
        r_["exit_t"] = p["time"].iat[min(r_["j"], len(p) - 1)]
    return rows


def delay(interval: str, fee: float, slip: float, workers: int, out: Path) -> pd.DataFrame:
    coins = [c for c in TOP100 if c.isascii()]
    # 先把要用的 1 分鐘日檔抓好（多執行緒），模擬時直接讀快取
    need = []
    for c in coins:
        p, idx, _ = _live_signals(c, interval)
        if idx is None:
            continue
        days = {t.strftime("%Y-%m-%d") for t in p["time"].iloc[idx + 1]}  # 訊號 K 棒收盤 = 下一根開盤
        need += [(c, d) for d in sorted(days) if not (M1_DIR / c / f"{d}.csv.gz").exists()]
    print(f"下載 {len(need)} 個 1 分鐘日檔", file=sys.stderr)
    with ThreadPoolExecutor(16) as ex:
        for n, _ in enumerate(ex.map(lambda cd: _m1_day(*cd), need)):
            if n % 500 == 0:
                print(n, file=sys.stderr)
    rows = []
    with ProcessPoolExecutor(workers) as ex:
        for got in ex.map(_delay_coin, [(c, interval, fee, slip) for c in coins]):
            rows += got or []
    d = pd.DataFrame(rows)
    d["time"] = pd.to_datetime(d["time"], utc=True)
    d["exit_t"] = pd.to_datetime(d["exit_t"], utc=True)
    tag = f"{interval}{'_slip' + str(slip) if slip else ''}"
    d.to_csv(out / f"entry_delay_trades_{tag}.csv.gz", index=False)
    res = []
    for frame, gf in d.groupby("frame", sort=False):
        base = gf[gf["delay"] == 0].set_index(["coin", "i"])["r"]
        for dl, g in gf.groupby("delay"):
            g = g.sort_values(["coin", "i"]).reset_index(drop=True)
            k = cluster_key(g)
            diff = g["r"].to_numpy() - base.reindex(pd.MultiIndex.from_frame(g[["coin", "i"]])).to_numpy()
            for pn, pmask in periods(g["time"]).items():
                fm = pmask & g["filled"].to_numpy() & (g["kind"] != -1).to_numpy()
                acc = account(g.assign(live=True), fm)
                dm = pmask & (g["kind"] != -1).to_numpy()
                d_mean, d_se, d_cm, d_cse, _ = cluster_stats(diff[dm], k[dm])
                res.append({"frame": frame, "delay": dl, "period": pn, "signals": int(pmask.sum()),
                            "fill_rate": g["filled"].to_numpy()[pmask].mean(),
                            "move_r": g["move_r"].to_numpy()[pmask].mean(),
                            "per_signal_diff": d_mean, "per_signal_se": d_se, "per_signal_cluster_diff": d_cm,
                            "per_signal_cluster_se": d_cse, **acc})
    t = pd.DataFrame(res)
    t.to_csv(out / f"entry_delay_{tag}.csv", index=False)
    print(t[["frame", "delay", "period", "fill_rate", "move_r", "per_signal_diff", "per_signal_se", "trades", "exp_r",
             "win_rate", "sharpe", "max_dd_pct"]].round(4).to_string())
    return t


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", default="4H", choices=["4H", "2H"])
    ap.add_argument("--fee", type=float, default=0.0005)
    ap.add_argument("--slip", type=float, default=0.0)
    ap.add_argument("--source", default="binance", choices=["binance", "pionex"])
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--out", default="output")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--timing", action="store_true")
    ap.add_argument("--delay", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    tag = args.interval + (f"_slip{args.slip}" if args.slip else "") + ("_pionex" if args.source == "pionex" else "")
    if args.analyze:
        analyze(args.interval, out, tag)
    elif args.timing:
        timing(args.interval, args.fee, args.slip, args.workers, out)
    elif args.delay:
        delay(args.interval, args.fee, args.slip, args.workers, out)
    else:
        build(args.interval, args.fee, args.slip, args.source, args.workers, out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
