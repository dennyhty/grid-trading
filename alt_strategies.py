"""其他主流策略的回測：python alt_strategies.py --family meanrev|vwap|vegas|momentum [--tf 4h] [--out output]

- meanrev：布林通道 + RSI 極端值的均值回歸，可加「盤整」過濾（ADX、維加斯通道走平），目標價是布林中線。
- vwap：週/月錨定 VWAP（含成交量加權標準差帶）的偏離回歸與順勢回踩；成交量分布（POC、價值區 VAL/VAH）的
  「回到價值區」（80% 規則）。
- vegas：現行的 4H 維加斯趨勢（收緊 RSI），拿來當基準。
- momentum：橫斷面強弱輪動，固定日期再平衡，做多最強 k 個、做空最弱 k 個。

每個設定都輸出全部交易到 CSV；選參數、驗證、隨機對照在 alt_report.py。
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from grid_trading import indicators as ind
from grid_trading import signals
from grid_trading.altsim import Bars, coins, load_bars, run_signals, to_frame

DAY = 86_400_000_000_000
WEEK_ANCHOR = pd.Timestamp("2023-01-02", tz="UTC").value  # 週一 00:00 UTC


# ---------------------------------------------------------------- 共用指標
def base_ind(b: Bars) -> dict:
    df = b.df
    c = df["close"]
    x = {"atr": ind.atr(df).to_numpy(), "rsi": ind.rsi(c).to_numpy(), "adx": ind.adx(df).to_numpy(),
         "ema144": ind.ema(c, 144).to_numpy(), "ema169": ind.ema(c, 169).to_numpy(),
         "sma20": c.rolling(20).mean().to_numpy(), "std20": c.rolling(20).std(ddof=0).to_numpy()}
    x["flat"] = np.abs(x["ema169"] - pd.Series(x["ema169"]).shift(12).to_numpy()) < 1.5 * x["atr"]
    return x


# ---------------------------------------------------------------- 均值回歸
def meanrev_grid():
    for k, rsi, filt, stop, confirm in itertools.product(
            (2.0, 2.5), ((30, 70), (20, 80)), ("none", "adx20", "adx25", "flat", "adx25+flat"), (1.5, 2.5), (0, 1)):
        yield f"meanrev|k{k}|rsi{rsi[0]}-{rsi[1]}|{filt}|sl{stop}|conf{confirm}", dict(
            k=k, rsi=rsi, filt=filt, stop=stop, confirm=confirm)


def range_ok(x: dict, filt: str) -> np.ndarray:
    ok = np.ones(len(x["atr"]), bool)
    if "adx20" in filt:
        ok &= x["adx"] < 20
    if "adx25" in filt:
        ok &= x["adx"] < 25
    if "flat" in filt:
        ok &= x["flat"]
    return ok


def meanrev_signals(b: Bars, x: dict, p: dict):
    c, mid, sd = b.c, x["sma20"], x["std20"]
    up, lo = mid + p["k"] * sd, mid - p["k"] * sd
    lo_rsi, hi_rsi = p["rsi"]
    ext_long = (c < lo) & (x["rsi"] < lo_rsi)
    ext_short = (c > up) & (x["rsi"] > hi_rsi)
    if p["confirm"]:  # 前一根在帶外、這根收回帶內
        prev_l = np.r_[False, ext_long[:-1]]
        prev_s = np.r_[False, ext_short[:-1]]
        long_ = prev_l & (c >= lo)
        short_ = prev_s & (c <= up)
    else:
        long_, short_ = ext_long, ext_short
    ok = range_ok(x, p["filt"]) & np.isfinite(mid) & (np.arange(len(c)) >= 200) & (x["atr"] > 0)
    out = []
    for i in np.flatnonzero((long_ | short_) & ok):
        d = 1 if long_[i] else -1
        target = mid[i]
        if d * (target - c[i]) <= 0:
            continue
        out.append((int(i), d, p["stop"] * x["atr"][i], [float(target)] * 3))
    return out


# ---------------------------------------------------------------- VWAP
def anchored_vwap(b: Bars, anchor: str):
    """週（週一 00:00 UTC 起）或月錨定的 VWAP 和成交量加權標準差，以及該期間已經過幾根 K 線。"""
    if anchor == "week":
        grp = (b.t - WEEK_ANCHOR) // (7 * DAY)
    else:
        grp = pd.to_datetime(b.t, unit="ns", utc=True).tz_localize(None).to_period("M").astype("int64").to_numpy()
    tp = (b.h + b.l + b.c) / 3
    s = pd.DataFrame({"g": grp, "pv": tp * b.v, "v": b.v, "p2v": tp * tp * b.v})
    g = s.groupby("g")
    cv, cpv, cp2v = g["v"].cumsum(), g["pv"].cumsum(), g["p2v"].cumsum()
    vwap = (cpv / cv).to_numpy()
    sd = np.sqrt(np.maximum((cp2v / cv).to_numpy() - vwap**2, 0))
    return vwap, sd, g.cumcount().to_numpy()


def vwap_grid():
    for k, filt, stop, confirm in itertools.product((1.5, 2.0, 2.5), ("none", "adx25"), (1.5, 2.5), (0, 1)):
        yield f"vwapRev|week|k{k}|{filt}|sl{stop}|conf{confirm}", dict(kind="rev", anchor="week", k=k, filt=filt,
                                                                       stop=stop, confirm=confirm)
    for anchor, stop in itertools.product(("week", "month"), (1.5, 2.5)):
        yield f"vwapPull|{anchor}|sl{stop}", dict(kind="pull", anchor=anchor, stop=stop)


def vwap_signals(b: Bars, x: dict, p: dict, cache: dict):
    key = p["anchor"] if "anchor" in p else "week"
    if key not in cache:
        cache[key] = anchored_vwap(b, key)
    vwap, sd, age = cache[key]
    c, h, l, atr = b.c, b.h, b.l, x["atr"]
    n = len(c)
    ok = (age >= 12) & (np.arange(n) >= 200) & (atr > 0) & (sd > 0)
    out = []
    if p["kind"] == "rev":
        up, lo = vwap + p["k"] * sd, vwap - p["k"] * sd
        ext_l, ext_s = c < lo, c > up
        if p["confirm"]:
            long_ = np.r_[False, ext_l[:-1]] & (c >= lo)
            short_ = np.r_[False, ext_s[:-1]] & (c <= up)
        else:
            long_, short_ = ext_l, ext_s
        ok &= range_ok(x, p["filt"])
        for i in np.flatnonzero((long_ | short_) & ok):
            d = 1 if long_[i] else -1
            if d * (vwap[i] - c[i]) <= 0:
                continue
            out.append((int(i), d, p["stop"] * atr[i], [float(vwap[i])] * 3))
    else:  # 順勢回踩：維加斯通道同向，K 線下探（上探）VWAP 後收回其上（下）
        top, bot = np.maximum(x["ema144"], x["ema169"]), np.minimum(x["ema144"], x["ema169"])
        up_t = (c > top) & (x["ema144"] > x["ema169"])
        dn_t = (c < bot) & (x["ema144"] < x["ema169"])
        long_ = up_t & (l <= vwap) & (c > vwap) & (b.c > b.o)
        short_ = dn_t & (h >= vwap) & (c < vwap) & (b.c < b.o)
        for i in np.flatnonzero((long_ | short_) & ok):
            d = 1 if long_[i] else -1
            r = p["stop"] * atr[i]
            out.append((int(i), d, r, [c[i] + d * k * r for k in (1, 2, 3)]))
    return out


# ---------------------------------------------------------------- 成交量分布
def vp_levels(h1: pd.DataFrame, bars: Bars, days: int, nbins: int = 80):
    """每天 00:00 UTC 用前 days 天的 1H 資料算 POC 和價值區（70%），整天沿用，對應到 bars 的每一根。"""
    t = h1["time"].dt.as_unit("ns").astype("int64").to_numpy()
    tp = ((h1["high"] + h1["low"] + h1["close"]) / 3).to_numpy()
    v = h1["volume"].to_numpy()
    hi, lo = h1["high"].to_numpy(), h1["low"].to_numpy()
    day0 = t[0] // DAY
    day_of_bar = bars.t // DAY
    first_day, last_day = int(day_of_bar.min()), int(day_of_bar.max())
    days_idx = np.arange(first_day, last_day + 1)
    starts = np.searchsorted(t, days_idx * DAY)           # 當天第一根 1H
    poc = np.full(len(days_idx), np.nan)
    val, vah = poc.copy(), poc.copy()
    for k, d in enumerate(days_idx):
        e = starts[k]
        s = np.searchsorted(t, (d - days) * DAY)
        if e - s < days * 24 * 0.9 or d < day0 + days:
            continue
        a, z = lo[s:e].min(), hi[s:e].max()
        if z <= a:
            continue
        idx = np.clip(((tp[s:e] - a) / (z - a) * nbins).astype(int), 0, nbins - 1)
        hist = np.bincount(idx, weights=v[s:e], minlength=nbins)
        pk = int(hist.argmax())
        lo_i = hi_i = pk
        acc, tot = hist[pk], hist.sum()
        while acc < 0.7 * tot and (lo_i > 0 or hi_i < nbins - 1):
            below = hist[lo_i - 1] if lo_i > 0 else -1
            above = hist[hi_i + 1] if hi_i < nbins - 1 else -1
            if above >= below:
                hi_i += 1
                acc += above
            else:
                lo_i -= 1
                acc += below
        bw = (z - a) / nbins
        poc[k], val[k], vah[k] = a + (pk + 0.5) * bw, a + lo_i * bw, a + (hi_i + 1) * bw
    pos = day_of_bar - first_day
    return poc[pos], val[pos], vah[pos]


def vp_grid():
    for days, target, stop in itertools.product((10, 20, 30), ("poc", "edge"), (1.5, 2.5)):
        yield f"vpReenter|d{days}|{target}|sl{stop}", dict(days=days, target=target, stop=stop)


def vp_signals(b: Bars, x: dict, p: dict, cache: dict, h1: pd.DataFrame):
    key = ("vp", p["days"])
    if key not in cache:
        cache[key] = vp_levels(h1, b, p["days"])
    poc, val, vah = cache[key]
    c, atr = b.c, x["atr"]
    prev_c = np.r_[np.nan, c[:-1]]
    ok = np.isfinite(poc) & (np.arange(len(c)) >= 200) & (atr > 0)
    long_ = ok & (prev_c < val) & (c >= val) & (c < vah)       # 前一根收在價值區下方，這根收回區內
    short_ = ok & (prev_c > vah) & (c <= vah) & (c > val)
    out = []
    for i in np.flatnonzero(long_ | short_):
        d = 1 if long_[i] else -1
        target = (poc[i] if p["target"] == "poc" else (vah[i] if d > 0 else val[i]))
        if d * (target - c[i]) <= 0:
            continue
        out.append((int(i), d, p["stop"] * atr[i], [float(target)] * 3))
    return out


# ---------------------------------------------------------------- 基準：現行 4H 維加斯趨勢（收緊 RSI）
def vegas_signals(b: Bars, strict: bool):
    s = signals.get("vegas_trend", **({"rsi_long": (54, 70), "rsi_short": (30, 46)} if strict else {}))
    out = []
    for g in s.scan(b.df, b.sym, b.tf):
        i = int(np.searchsorted(b.t, g.time.value)) - 1
        out.append((i, g.direction, abs(g.price - g.sl), g.tps))
    return out


# ---------------------------------------------------------------- 排程
def work(args):
    coin, family, tf = args
    b = load_bars(coin, tf)
    if b is None or len(b.t) < 400:
        return []
    x = base_ind(b)
    rows: list[tuple] = []
    max_bars = {"4h": 30, "2h": 48, "1d": 10}[tf] if family != "vegas" else 240
    if family == "meanrev":
        for cfg, p in meanrev_grid():
            rows += run_signals(b, cfg, meanrev_signals(b, x, p), max_bars=max_bars)
    elif family == "vwap":
        cache: dict = {}
        h1 = load_bars(coin, "1h").df
        for cfg, p in vwap_grid():
            mb = max_bars if p["kind"] == "rev" else 240
            rows += run_signals(b, cfg, vwap_signals(b, x, p, cache), max_bars=mb)
        for cfg, p in vp_grid():
            rows += run_signals(b, cfg, vp_signals(b, x, p, cache, h1), max_bars=max_bars)
    elif family == "vegas":
        rows += run_signals(b, "vegas|strict", vegas_signals(b, True), max_bars=240)
        rows += run_signals(b, "vegas|current", vegas_signals(b, False), max_bars=240)
    return rows


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--family", required=True, choices=["meanrev", "vwap", "vegas"])
    p.add_argument("--tf", default="4h", choices=["2h", "4h", "1d"])
    p.add_argument("--out", default="output")
    a = p.parse_args()
    jobs = [(c, a.family, a.tf) for c in coins()]
    rows = []
    with ProcessPoolExecutor(os.cpu_count() or 2) as ex:
        for r in ex.map(work, jobs):
            rows += r
    t = to_frame(rows)
    Path(a.out).mkdir(exist_ok=True)
    path = Path(a.out) / f"alt_{a.family}_{a.tf}_trades.csv.gz"
    t.to_csv(path, index=False)
    print(a.family, a.tf, len(t), "trades ->", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
