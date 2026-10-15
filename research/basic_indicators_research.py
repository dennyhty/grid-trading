"""基礎指標策略研究（只做分析，不改實盤規則）。

python -m research.basic_indicators_research --interval 4H [--source pionex]
python -m research.basic_indicators_research --interval 4H --robust     第二步：候選家族換參數（穩健性）

需要 numba（和 stop_research.py 共用模擬程式）。

做法：
- 每個「觸發」都是單一、常見的指標事件（MA/EMA 交叉、布林帶、維加斯通道、MFI、RSI、量價背離、
  MACD、唐奇安通道、超級趨勢、SAR、KD、CCI、肯特納通道、一目均衡表、DMI）。條件第一次成立那根收盤進場。
- 每個觸發再配三種方向過濾：不過濾、EMA200（價格在 EMA200 上方只做多）、維加斯（價格和 EMA12 在
  EMA144/169 通道同一側且通道同向）。
- 出場一律用實盤的出場（止損 1.5 ATR = 1R；4H：1R 平一半、保本、3 ATR 追蹤；2H：1/3/5R 各平 1/4、
  最後 1/4 追蹤 3 ATR），只比較「進場」的好壞。手續費每邊 0.05%。
- 同一個幣同時只持有一張（busy_filter）。另外算群集績效：同一天同方向的訊號算一注。
- 2020-23 和 2024 年後分開看；挑最好的只能用 2020-23，2024 年後當驗證。

輸出（output/）：
- basic_grid_{週期}{標記}.csv：每個觸發 × 過濾 × 期間的績效。
- basic_grid_{週期}_robust.csv：--robust 的結果。basic_trades_*.csv.gz：每筆訊號的明細。
"""

from __future__ import annotations

import argparse
import sys
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from grid_trading import indicators as ind
from grid_trading.universe import TOP100
from research.stop_research import (LIVE_ATR, MAX_BARS, MIN_STOP_PCT, PLANS, SPLIT, busy_filter, cluster_stats,
                           load_frame, sim_many, summarize)

FEE = 0.0005
WARMUP = 220
FILTERS = ("none", "ema200", "vegas")


# ---------- 指標 ----------

def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n).mean()


def mfi(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """資金流量指標（MFI）：用典型價 × 成交量算的 RSI。"""
    tp = (df["high"] + df["low"] + df["close"]) / 3
    flow = tp * df["volume"]
    d = tp.diff()
    pos = flow.where(d > 0, 0.0).rolling(n).sum()
    neg = flow.where(d < 0, 0.0).rolling(n).sum()
    return 100 - 100 / (1 + pos / neg.replace(0, np.nan))


def obv(df: pd.DataFrame) -> pd.Series:
    return (np.sign(df["close"].diff()).fillna(0) * df["volume"]).cumsum()


def supertrend(df: pd.DataFrame, n: int = 10, k: float = 3.0) -> np.ndarray:
    """超級趨勢的方向：1 多、-1 空。"""
    hl2 = ((df["high"] + df["low"]) / 2).to_numpy()
    a = ind.atr(df, n).to_numpy()
    c = df["close"].to_numpy()
    up, dn = hl2 - k * a, hl2 + k * a
    out = np.ones(len(c))
    fu, fd = up[0], dn[0]
    for i in range(1, len(c)):
        fu = max(up[i], fu) if c[i - 1] > fu else up[i]
        fd = min(dn[i], fd) if c[i - 1] < fd else dn[i]
        if out[i - 1] == 1:
            out[i] = -1 if c[i] < fu else 1
        else:
            out[i] = 1 if c[i] > fd else -1
    return out


def stoch(df: pd.DataFrame, n: int = 14, k: int = 3, d: int = 3) -> tuple[pd.Series, pd.Series]:
    lo, hi = df["low"].rolling(n).min(), df["high"].rolling(n).max()
    raw = 100 * (df["close"] - lo) / (hi - lo).replace(0, np.nan)
    kk = raw.rolling(k).mean()
    return kk, kk.rolling(d).mean()


def cci(df: pd.DataFrame, n: int = 20) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3
    m = tp.rolling(n).mean()
    md = tp.rolling(n).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    return (tp - m) / (0.015 * md)


def dmi(df: pd.DataFrame, n: int = 14) -> tuple[pd.Series, pd.Series]:
    up = df["high"].diff()
    down = -df["low"].diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)
    tr = ind.atr(df, n)
    return (100 * plus_dm.ewm(alpha=1 / n, adjust=False).mean() / tr,
            100 * minus_dm.ewm(alpha=1 / n, adjust=False).mean() / tr)


def cross_up(a, b) -> np.ndarray:
    n = max(np.size(a), np.size(b))
    a = pd.Series(np.broadcast_to(np.asarray(a, float), n))
    b = pd.Series(np.broadcast_to(np.asarray(b, float), n))
    return ((a > b) & (a.shift(1) <= b.shift(1))).to_numpy()


def edge(x) -> np.ndarray:
    """條件第一次成立的那根。"""
    x = pd.Series(np.asarray(x, bool))
    return (x & ~x.shift(1, fill_value=False)).to_numpy()


def divergence(price_lo, price_hi, osc, look: int = 30, recent: int = 5):
    """背離：最近 recent 根創 look 根新低（高），但指標的低點（高點）比前一段高（低）。回傳（看多, 看空）。"""
    plo, phi, o = pd.Series(price_lo), pd.Series(price_hi), pd.Series(osc)
    p_rl, p_pl = plo.rolling(recent).min(), plo.shift(recent).rolling(look - recent).min()
    o_rl, o_pl = o.rolling(recent).min(), o.shift(recent).rolling(look - recent).min()
    p_rh, p_ph = phi.rolling(recent).max(), phi.shift(recent).rolling(look - recent).max()
    o_rh, o_ph = o.rolling(recent).max(), o.shift(recent).rolling(look - recent).max()
    bull = (plo == p_rl) & (p_rl < p_pl) & (o_rl > o_pl)
    bear = (phi == p_rh) & (p_rh > p_ph) & (o_rh < o_ph)
    return bull.to_numpy(), bear.to_numpy()


def triggers(df: pd.DataFrame) -> dict[str, tuple[np.ndarray, np.ndarray, str]]:
    """{代號: (做多觸發, 做空觸發, 類型)}。類型：trend 順勢、revert 逆勢。"""
    c, h, l, v = df["close"], df["high"], df["low"], df["volume"]
    out: dict[str, tuple[np.ndarray, np.ndarray, str]] = {}

    def add(name, lo, sh, kind):
        out[name] = (np.asarray(lo, bool), np.asarray(sh, bool), kind)

    # MA / EMA
    for f, s, fn in ((9, 21, ind.ema), (20, 50, ind.ema), (50, 200, ind.ema), (20, 50, sma), (50, 200, sma)):
        a, b = fn(c, f), fn(c, s)
        tag = "EMA" if fn is ind.ema else "MA"
        add(f"{tag}{f}/{s} 交叉", cross_up(a, b), cross_up(b, a), "trend")
    for n in (20, 50, 200):
        e = ind.ema(c, n)
        add(f"價格穿越 EMA{n}", cross_up(c, e), cross_up(e, c), "trend")
    e20, e50 = ind.ema(c, 20), ind.ema(c, 50)
    # 均線回踩：EMA20 > EMA50 時碰到 EMA20 後收盤站回
    add("EMA20 回踩", (e20 > e50) & (l <= e20) & (c > e20) & (c > df["open"]),
        (e20 < e50) & (h >= e20) & (c < e20) & (c < df["open"]), "trend")

    # 布林帶（20, 2）
    mid, sd = sma(c, 20), c.rolling(20).std()
    up_b, lo_b = mid + 2 * sd, mid - 2 * sd
    add("布林突破", cross_up(c, up_b), cross_up(lo_b, c), "trend")
    add("布林回歸", cross_up(c, lo_b), cross_up(up_b, c), "revert")  # 跌破下軌後收回
    bw = (up_b - lo_b) / mid
    squeeze = (bw <= bw.rolling(120).quantile(0.2)).shift(1, fill_value=False)
    add("布林收縮後突破", cross_up(c, up_b) & squeeze, cross_up(lo_b, c) & squeeze, "trend")
    add("布林中軌穿越", cross_up(c, mid), cross_up(mid, c), "trend")

    # 維加斯通道（EMA144/169）
    e144, e169 = ind.ema(c, 144), ind.ema(c, 169)
    top, bot = np.maximum(e144, e169), np.minimum(e144, e169)
    add("維加斯通道突破", cross_up(c, top), cross_up(bot, c), "trend")
    add("維加斯回踩反彈", (e144 > e169) & (c > top) & (l <= top * 1.0) & (c > df["open"]),
        (e144 < e169) & (c < bot) & (h >= bot) & (c < df["open"]), "trend")
    e12 = ind.ema(c, 12)
    add("EMA12 穿越維加斯", cross_up(e12, top), cross_up(bot, e12), "trend")

    # MFI / RSI
    m = mfi(df)
    add("MFI 20/80 回轉", cross_up(m, 20), cross_up(80, m), "revert")
    add("MFI 50 穿越", cross_up(m, 50), cross_up(50, m), "trend")
    r = ind.rsi(c)
    add("RSI 30/70 回轉", cross_up(r, 30), cross_up(70, r), "revert")
    add("RSI 50 穿越", cross_up(r, 50), cross_up(50, r), "trend")
    add("RSI 40/60 回檔", cross_up(r, 40), cross_up(60, r), "trend")  # 搭配趨勢過濾＝順勢回檔
    add("RSI 70/30 突破", cross_up(r, 70), cross_up(30, r), "trend")

    # 量價背離
    o = obv(df)
    b1, s1 = divergence(l, h, o)
    add("OBV 量價背離", edge(b1), edge(s1), "revert")
    b2, s2 = divergence(l, h, r)
    add("RSI 背離", edge(b2), edge(s2), "revert")
    b3, s3 = divergence(l, h, m)
    add("MFI 背離", edge(b3), edge(s3), "revert")
    vol_hi = v > 2 * sma(v, 20).shift(1)
    hh, ll = h.rolling(20).max().shift(1), l.rolling(20).min().shift(1)
    add("放量突破 20 根高低", edge((c > hh) & vol_hi), edge((c < ll) & vol_hi), "trend")
    # 價漲量縮：創 20 根新高但成交量低於均量 → 看空；反之看多
    vol_lo = sma(v, 5) < 0.8 * sma(v, 20)
    add("價創新高量縮（反向）", edge((c < ll) & vol_lo), edge((c > hh) & vol_lo), "revert")

    # 其他常見指標
    mc = ind.macd(c)
    add("MACD 金叉死叉", cross_up(mc["macd"], mc["signal"]), cross_up(mc["signal"], mc["macd"]), "trend")
    add("MACD 零軸穿越", cross_up(mc["macd"], 0), cross_up(0, mc["macd"]), "trend")
    for n in (20, 55):
        add(f"唐奇安 {n} 突破", edge(c > h.rolling(n).max().shift(1)), edge(c < l.rolling(n).min().shift(1)), "trend")
    st = supertrend(df)
    add("超級趨勢翻轉", cross_up(st, 0), cross_up(0, st), "trend")
    ps = ind.psar(df)
    add("SAR 翻轉", cross_up(c, ps), cross_up(ps, c), "trend")
    k, d = stoch(df)
    add("KD 超買超賣交叉", cross_up(k, d) & (d < 20), cross_up(d, k) & (d > 80), "revert")
    cc = cci(df)
    add("CCI ±100 突破", cross_up(cc, 100), cross_up(-100, cc), "trend")
    add("CCI ±100 回轉", cross_up(cc, -100), cross_up(100, cc), "revert")
    a = ind.atr(df)
    kc_u, kc_l = ind.ema(c, 20) + 2 * a, ind.ema(c, 20) - 2 * a
    add("肯特納通道突破", cross_up(c, kc_u), cross_up(kc_l, c), "trend")
    tenkan = (h.rolling(9).max() + l.rolling(9).min()) / 2
    kijun = (h.rolling(26).max() + l.rolling(26).min()) / 2
    span_a = ((tenkan + kijun) / 2).shift(26)
    span_b = ((h.rolling(52).max() + l.rolling(52).min()) / 2).shift(26)
    ctop, cbot = np.maximum(span_a, span_b), np.minimum(span_a, span_b)
    add("一目轉換線交叉（雲上/雲下）", cross_up(tenkan, kijun) & (c > ctop), cross_up(kijun, tenkan) & (c < cbot), "trend")
    add("一目突破雲", cross_up(c, ctop), cross_up(cbot, c), "trend")
    pdi, mdi = dmi(df)
    adx = ind.adx(df)
    add("DMI 交叉（ADX>20）", cross_up(pdi, mdi) & (adx > 20), cross_up(mdi, pdi) & (adx > 20), "trend")
    return out


def robust_triggers(df: pd.DataFrame) -> dict[str, tuple[np.ndarray, np.ndarray, str]]:
    """第二步：第一步在兩段時期群集都為正的幾個家族，換參數看是不是只有一組剛好好。"""
    c, h, l = df["close"], df["high"], df["low"]
    out: dict[str, tuple[np.ndarray, np.ndarray, str]] = {}
    for n in (14, 20, 30, 50):
        for k in (1.5, 2.0, 2.5):
            mid, sd = sma(c, n), c.rolling(n).std()
            up_b, lo_b = mid + k * sd, mid - k * sd
            out[f"布林回歸 n{n} k{k}"] = (cross_up(c, lo_b), cross_up(up_b, c), "revert")
            # 碰到下軌（不必收在外面）後收紅 K
            out[f"布林觸軌收紅 n{n} k{k}"] = (edge((l <= lo_b) & (c > df["open"])),
                                          edge((h >= up_b) & (c < df["open"])), "revert")
    r = ind.rsi(c)
    for look in (20, 30, 50):
        for rec in (3, 5, 8):
            b, s_ = divergence(l, h, r, look, rec)
            out[f"RSI 背離 look{look} rec{rec}"] = (edge(b), edge(s_), "revert")
    for lo_t, hi_t in ((25, 75), (30, 70), (35, 65), (40, 60)):
        out[f"RSI {lo_t}/{hi_t} 回轉"] = (cross_up(r, lo_t), cross_up(hi_t, r), "revert")
    for f, s2 in ((9, 26), (7, 22), (12, 30)):
        tk = (h.rolling(f).max() + l.rolling(f).min()) / 2
        kj = (h.rolling(s2).max() + l.rolling(s2).min()) / 2
        out[f"轉換/基準線交叉 {f}/{s2}"] = (cross_up(tk, kj), cross_up(kj, tk), "trend")
    return out


def trend_filters(df: pd.DataFrame) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    c = df["close"]
    e200 = ind.ema(c, 200)
    e12, e144, e169 = ind.ema(c, 12), ind.ema(c, 144), ind.ema(c, 169)
    top, bot = np.maximum(e144, e169), np.minimum(e144, e169)
    n = len(df)
    return {
        "none": (np.ones(n, bool), np.ones(n, bool)),
        "ema200": ((c > e200).to_numpy(), (c < e200).to_numpy()),
        "vegas": (((c > top) & (e12 > top) & (e144 > e169)).to_numpy(),
                  ((c < bot) & (e12 < bot) & (e144 < e169)).to_numpy()),
    }


# ---------- 回測 ----------

def simulate(df: pd.DataFrame, idx: np.ndarray, ds: np.ndarray, interval: str, atr: np.ndarray) -> dict:
    price = df["close"].to_numpy()[idx]
    a = atr[idx]
    tp_r, portions, trail_k = PLANS[interval]
    risk = np.maximum(LIVE_ATR * a, MIN_STOP_PCT * price)
    stop0 = price - ds * risk
    tps = np.stack([price + ds * k * risk for k in tp_r], axis=1)
    zero = np.zeros(len(idx))
    hi, lo, cl, op = (df[k].to_numpy() for k in ("high", "low", "close", "open"))
    isc = np.ones(len(hi), np.bool_)
    r, hit, j, kind, _ = sim_many(op, hi, lo, cl, isc, idx + 1, ds.astype(np.int64), price, risk, stop0, tps,
                                  np.array(portions), 0, trail_k * a, FEE, 0.0, MAX_BARS, 0, 0, zero, 0.0, 0.0,
                                  zero, 0, cl, cl, False)
    return {"r": r, "hit": hit, "j": j, "kind": kind, "risk_pct": risk / price}


def _coin(args):
    coin, interval, source, rule_filter, robust = args
    df = load_frame(coin, interval, source)
    if len(df) < WARMUP + 100:
        return None
    df = df.reset_index(drop=True)
    trig = robust_triggers(df) if robust else triggers(df)
    filt = trend_filters(df)
    atr = ind.atr(df).to_numpy()
    rows = []
    valid = np.arange(len(df)) >= WARMUP
    valid[-1] = False
    for tname, (tl, ts, tkind) in trig.items():
        for fname, (fl, fs) in filt.items():
            if rule_filter is not None and (tname, fname) not in rule_filter:
                continue
            lo = tl & fl & valid
            sh = ts & fs & valid
            idx = np.concatenate([np.nonzero(lo)[0], np.nonzero(sh)[0]])
            ds = np.concatenate([np.ones(lo.sum()), -np.ones(sh.sum())])
            if not len(idx):
                continue
            o = np.argsort(idx, kind="stable")
            idx, ds = idx[o], ds[o]
            res = simulate(df, idx, ds, interval, atr)
            rows.append(pd.DataFrame({
                "trigger": tname, "kind_t": tkind, "filter": fname, "coin": coin, "i": idx, "side": ds.astype(int),
                "time": df["time"].to_numpy()[idx], "exit_time": df["time"].to_numpy()[res["j"]],
                "r": res["r"], "hit": res["hit"], "j": res["j"], "kind": res["kind"], "risk_pct": res["risk_pct"]}))
    # 基準：隨機進場（每 30 根抽一根、方向隨機）和目前的實盤 vegas_trend（4H 無 ADX 條件、2H 有）
    # 隨機進場：每 30 根抽一根、方向隨機；有過濾時只留和過濾同方向的（＝只靠趨勢過濾能賺多少）
    rng = np.random.default_rng(zlib.crc32(coin.encode()))
    pick = valid & (rng.random(len(df)) < 1 / 30)
    side = rng.choice([-1.0, 1.0], len(df))
    for fname, (fl, fs) in filt.items():
        m = pick & (((side > 0) & fl) | ((side < 0) & fs))
        ridx = np.nonzero(m)[0]
        rds = side[ridx]
        res = simulate(df, ridx, rds, interval, atr)
        rows.append(pd.DataFrame({"trigger": "隨機進場", "kind_t": "bench", "filter": fname, "coin": coin,
                                  "i": ridx, "side": rds.astype(int), "time": df["time"].to_numpy()[ridx],
                                  "exit_time": df["time"].to_numpy()[res["j"]], "r": res["r"], "hit": res["hit"],
                                  "j": res["j"], "kind": res["kind"], "risk_pct": res["risk_pct"]}))
    # 目前實盤的 vegas_trend（向量化，條件和 VegasTrend.check / accept 相同）
    c = df["close"]
    e12, e144, e169 = ind.ema(c, 12), ind.ema(c, 144), ind.ema(c, 169)
    top, bot = np.maximum(e144, e169), np.minimum(e144, e169)
    hist = ind.macd(c)["hist"]
    prev = hist.shift(1)
    r = ind.rsi(c)
    a = pd.Series(atr)
    lo = (c > top) & (e12 > top) & (e144 > e169) & (prev <= 0) & (hist > 0) & r.between(54, 70) & (c - top <= 3 * a)
    sh = (c < bot) & (e12 < bot) & (e144 < e169) & (prev >= 0) & (hist < 0) & r.between(30, 46) & (bot - c <= 3 * a)
    if interval == "2H":
        ok = ~(ind.adx(df) >= 24)
        lo, sh = lo & ok, sh & ok
    lo, sh = lo.to_numpy() & valid, sh.to_numpy() & valid
    vidx = np.nonzero(lo | sh)[0]
    vds = np.where(lo[vidx], 1.0, -1.0)
    res = simulate(df, vidx, vds, interval, atr)
    rows.append(pd.DataFrame({"trigger": "實盤 vegas_trend", "kind_t": "bench", "filter": "vegas", "coin": coin,
                              "i": vidx, "side": vds.astype(int), "time": df["time"].to_numpy()[vidx],
                              "exit_time": df["time"].to_numpy()[res["j"]], "r": res["r"], "hit": res["hit"],
                              "j": res["j"], "kind": res["kind"], "risk_pct": res["risk_pct"]}))
    return pd.concat(rows) if rows else None


def collect(interval: str, source: str, workers: int, rule_filter=None, robust: bool = False) -> pd.DataFrame:
    with ProcessPoolExecutor(workers) as ex:
        parts = [p for p in ex.map(_coin, [(c, interval, source, rule_filter, robust) for c in TOP100])
                 if p is not None]
    t = pd.concat(parts, ignore_index=True)
    t["time"] = pd.to_datetime(t["time"], utc=True)
    t["exit_time"] = pd.to_datetime(t["exit_time"], utc=True)
    return t


def grid(t: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (tname, fname), g in t.groupby(["trigger", "filter"], sort=False):
        g = g.sort_values(["coin", "i"], kind="stable")
        coin_id = pd.factorize(g["coin"])[0]
        take = busy_filter(coin_id, g["i"].to_numpy(np.int64), g["j"].to_numpy(np.int64),
                           g["kind"].to_numpy(np.int64))
        g = g[take]
        for pname, pm in (("全期", np.ones(len(g), bool)), ("2020-23", (g["time"] < SPLIT).to_numpy()),
                          ("2024+", (g["time"] >= SPLIT).to_numpy())):
            x = g[pm]
            if len(x) < 30:
                continue
            key = (x["time"].dt.floor("D").astype(str) + x["side"].astype(str)).to_numpy()
            s = summarize(x["r"].to_numpy(), x["kind"].to_numpy(), x["hit"].to_numpy(),
                          (x["j"] - x["i"]).to_numpy(), x["time"].to_numpy(dtype="datetime64[ns]"),
                          x["exit_time"].to_numpy(dtype="datetime64[ns]"), key, x["risk_pct"].to_numpy())
            lo, sh = x[x["side"] > 0]["r"], x[x["side"] < 0]["r"]
            yr = x.groupby(x["time"].dt.year)["r"].mean()
            rows.append({"trigger": tname, "type": x["kind_t"].iat[0], "filter": fname, "period": pname, **s,
                         "long_r": lo.mean(), "short_r": sh.mean(), "years_pos": int((yr > 0).sum()),
                         "years": len(yr), "t_cluster": s["cluster_r"] / s["cluster_se"] if s["cluster_se"] else np.nan})
    return pd.DataFrame(rows)


def add_crowd(t: pd.DataFrame, hours: int) -> pd.DataFrame:
    """crowd：同一個觸發 × 過濾、同方向，過去 hours 小時內（含這根）有幾個幣出過訊號（含自己）。
    掃描時這根收盤就知道，不用到未來的資料。"""
    t = t.sort_values("time", kind="stable").copy()
    out = np.zeros(len(t), np.int64)
    pos = {k: v for k, v in zip(t.index, range(len(t)))}
    for _, g in t.groupby(["trigger", "filter", "side"], sort=False):
        tm = g["time"].to_numpy(dtype="datetime64[ns]").astype(np.int64)
        hi = np.searchsorted(tm, tm, side="right")
        lo = np.searchsorted(tm, tm - hours * 3_600_000_000_000, side="right")
        out[[pos[k] for k in g.index]] = hi - lo
    t["crowd"] = out
    return t


def crowd_grid(t: pd.DataFrame, hours: int = 24) -> pd.DataFrame:
    """依 crowd 分組：只做「同時出訊號的幣很少」或「很多」的訊號。"""
    t = add_crowd(t, hours)
    rows = []
    for lab, m in (("全部", None), ("crowd≤1", t["crowd"] <= 1), ("crowd≤2", t["crowd"] <= 2),
                   ("crowd≤3", t["crowd"] <= 3), ("crowd≤5", t["crowd"] <= 5), ("crowd≥3", t["crowd"] >= 3),
                   ("crowd≥6", t["crowd"] >= 6)):
        x = t if m is None else t[m]
        g = grid(x)
        g.insert(2, "crowd", lab)
        rows.append(g)
    return pd.concat(rows, ignore_index=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", default="4H", choices=["4H", "2H"])
    ap.add_argument("--source", default="binance", choices=["binance", "pionex"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--robust", action="store_true", help="第二步：候選家族換參數")
    ap.add_argument("--crowd", action="store_true", help="讀第一步的明細，依同時出訊號的幣數分組")
    args = ap.parse_args()
    out = Path("output")
    out.mkdir(exist_ok=True)
    tag = ("_robust" if args.robust else "") + ("" if args.source == "binance" else "_pionex")
    if args.crowd:
        t = pd.read_csv(out / f"basic_trades_{args.interval}{tag}.csv.gz", parse_dates=["time", "exit_time"])
        cg = crowd_grid(t)
        cg.to_csv(out / f"basic_crowd_{args.interval}{tag}.csv", index=False)
        return 0
    t = collect(args.interval, args.source, args.workers, robust=args.robust)
    t.to_csv(out / f"basic_trades_{args.interval}{tag}.csv.gz", index=False)
    g = grid(t)
    g.to_csv(out / f"basic_grid_{args.interval}{tag}.csv", index=False)
    show = g[g["period"] != "全期"].pivot_table(index=["trigger", "filter"], columns="period",
                                                 values=["exp_r", "cluster_r", "win_rate", "trades"])
    with pd.option_context("display.width", 250, "display.max_rows", 500):
        print(show.round(3).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
