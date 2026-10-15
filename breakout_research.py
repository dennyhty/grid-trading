"""突破追多的過濾研究（只做分析，不改實盤規則）。

python breakout_research.py collect      # 產生所有基礎訊號、特徵和各種出場的結果 → output/breakout_signals.pkl
python breakout_research.py analyze      # 單一過濾的分組表、挑選（只用 2020-23）、2024+ 驗證

需要 numba（和 stop_research.py 共用模擬程式）。

做法：
- 基礎訊號（寬鬆版，用來測過濾）：1H 收盤突破前 20 根高點、站上 1H EMA20/50/144/169、4H 收盤在維加斯通道之上。
  停損放在最近的 1H 波段低點（左右各 3 根）下方 0.3%，距離 2%~15%。空單完全鏡像。
- 外部訊號群的原版 = 基礎訊號 + 1H RSI 60~80 + 4H MACD 柱為正且放大 + 停損距離 3%~12%。
- 每個訊號記下進場當下的特徵（全部只用已收盤的 K 線），再用幾種出場各跑一次：
  grp：1R 平一半、保本、2R 平剩下（外部訊號群的做法）；tp1：1R 全出；trail：1R 平一半、保本、剩下追蹤 3 倍 4H ATR；
  q135：1/3/5R 各平 1/4、最後 1/4 追蹤 3 倍 4H ATR。手續費每邊 0.05%，最多抱 7 天（追蹤版 14 天）。
- 同一個幣同時只持有一張（先過濾、再套 busy_filter）。
- 隨機基準：同樣的趨勢條件（站上 1H 均線、4H 在通道上）但不要求突破，每 10 根抽一根，同樣的停損和出場。
- 過濾只能用 2020-23 挑，2024 年後當驗證；看每筆 R 和群集穩健標準誤（同一天同方向算一群）。
"""

from __future__ import annotations

import sys
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from grid_trading import indicators as ind
from grid_trading.binance_history import load
from grid_trading.history import resample
from grid_trading.universe import TOP100
from stop_research import SPLIT, busy_filter, cluster_stats, sim_many

OUT = Path("output")
FEE = 0.0005
WARMUP = 200
LOOKBACK = 20
PIVOT_K = 3
BUFFER = 0.003
RISK_MIN, RISK_MAX = 0.02, 0.15
EXITS = {  # 名稱: (TP 的 R 倍數, 平倉比例, 追蹤幾倍 4H ATR, 最多抱幾根 1H)
    "grp": ((1.0, 2.0), (0.5, 0.5), 0.0, 168),
    "tp1": ((1.0,), (1.0,), 0.0, 168),
    "trail": ((1.0,), (0.5,), 3.0, 336),
    "q135": ((1.0, 3.0, 5.0), (0.25, 0.25, 0.25), 3.0, 336),
}


# ---------- 大週期 ----------

def warm(s: pd.Series, n: int) -> pd.Series:
    """均線還沒累積 n 根之前當作沒有資料。"""
    return s.where(np.arange(len(s)) >= n)


def above(x: pd.Series, y: pd.Series, d: int) -> pd.Series:
    """順著方向在上方為 1、反之為 0，沒有資料為 NaN。"""
    return pd.Series(np.where(x.isna() | y.isna(), np.nan, (d * (x - y) > 0).astype(float)), index=x.index)


def htf(df1: pd.DataFrame, rule: str, hours: int) -> pd.DataFrame:
    """合成大週期並算指標；avail = 這根收盤的時間（之後的 1H 才能用）。"""
    h = resample(df1, rule, origin="epoch")
    c = h["close"]
    e12, e144, e169 = ind.ema(c, 12), ind.ema(c, 144), ind.ema(c, 169)
    top, bot = np.maximum(e144, e169), np.minimum(e144, e169)
    hist = ind.macd(c)["hist"]
    atr = ind.atr(h)
    return pd.DataFrame({
        "avail": h["time"] + pd.Timedelta(hours=hours), "close": c, "top": top, "bot": bot,
        "up": c > top, "down": c < bot,
        "strict_up": (c > top) & (e12 > top) & (e144 > e169),
        "strict_dn": (c < bot) & (e12 < bot) & (e144 < e169),
        "hist": hist, "hist_prev": hist.shift(1), "atr": atr, "rsi": ind.rsi(c), "adx": ind.adx(h),
        "ema20": ind.ema(c, 20), "ema50": warm(ind.ema(c, 50), 50), "ema200": warm(ind.ema(c, 200), 200),
        "ret6": c / c.shift(6) - 1,
    })


def asof(df: pd.DataFrame, h: pd.DataFrame, prefix: str, cols: list[str]) -> pd.DataFrame:
    m = pd.merge_asof(df[["bar_close"]], h[["avail"] + cols].sort_values("avail"),
                      left_on="bar_close", right_on="avail", direction="backward")
    for col in cols:
        df[f"{prefix}{col}"] = m[col].to_numpy()
    return df


# ---------- 每個幣 ----------

def trend_series(coin: str) -> pd.DataFrame | None:
    df = load(coin, "1h")
    if len(df) < WARMUP * 4:
        return None
    h = htf(df, "4h", 4)
    return pd.DataFrame({"avail": h["avail"], coin: np.where(h["up"], 1, np.where(h["down"], -1, 0))}).iloc[169:]


def features(df: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    c, hi, lo, vol = df["close"], df["high"], df["low"], df["volume"]
    df["bar_close"] = df["time"] + pd.Timedelta(hours=1)
    for n in (20, 50, 144, 169):
        df[f"ema{n}"] = ind.ema(c, n)
    df["rsi"] = ind.rsi(c)
    df["atr"] = ind.atr(df)
    df["adx"] = ind.adx(df)
    df["prior_high"] = hi.shift(1).rolling(LOOKBACK).max()
    df["prior_low"] = lo.shift(1).rolling(LOOKBACK).min()
    k = PIVOT_K
    df["swing_low"] = lo.where(lo == lo.rolling(2 * k + 1, center=True).min()).ffill().shift(k)
    df["swing_high"] = hi.where(hi == hi.rolling(2 * k + 1, center=True).max()).ffill().shift(k)
    df["low24"] = lo.rolling(24).min()
    df["high24"] = hi.rolling(24).max()
    df["vol_ratio"] = vol / vol.shift(1).rolling(20).mean()
    df["squeeze"] = (df["prior_high"] - df["prior_low"]) / df["atr"]          # 突破前 20 根的區間寬度（ATR）
    df["ret24"] = c / c.shift(24) - 1
    df["ret168"] = c / c.shift(168) - 1
    df["close_pos"] = ((c - lo) / (hi - lo)).fillna(0.5)                       # 收在 K 棒的哪裡（1 = 最高）
    df["body_atr"] = (c - df["open"]) / df["atr"]
    # 突破了多久以來的高點：往回數幾根，最高價都還低於這根收盤（上限 720 根 = 30 天）
    arr_h, arr_l, arr_c = hi.to_numpy(), lo.to_numpy(), c.to_numpy()
    nh = np.zeros(len(df))
    nl = np.zeros(len(df))
    for w in (20, 50, 100, 200, 400, 720):
        mh = hi.shift(1).rolling(w).max().to_numpy()
        ml = lo.shift(1).rolling(w).min().to_numpy()
        nh = np.where(arr_c > mh, w, nh)
        nl = np.where(arr_c < ml, w, nl)
    df["new_high_n"], df["new_low_n"] = nh, nl
    df["age_days"] = (df["time"] - df["time"].iloc[0]).dt.total_seconds() / 86400

    h4 = htf(df[["time", "open", "high", "low", "close", "volume"]], "4h", 4)
    df = asof(df, h4, "h4_", ["close", "top", "bot", "up", "down", "strict_up", "strict_dn", "hist", "hist_prev",
                              "atr", "rsi", "adx", "ema200", "ret6"])
    d1 = htf(df[["time", "open", "high", "low", "close", "volume"]], "1D", 24)
    df = asof(df, d1, "d1_", ["close", "ema20", "ema50", "ema200", "rsi", "top", "bot", "adx"])
    m = pd.merge_asof(df[["bar_close"]], market, left_on="bar_close", right_on="avail", direction="backward")
    for col in market.columns:
        if col != "avail":
            df[col] = m[col].to_numpy()
    return df


def side_features(df: pd.DataFrame, d: int) -> pd.DataFrame:
    """把多空統一成「順著交易方向為正」的特徵。"""
    c = df["close"]
    f = pd.DataFrame(index=df.index)
    f["rsi_dir"] = df["rsi"] if d > 0 else 100 - df["rsi"]
    f["h4_rsi_dir"] = df["h4_rsi"] if d > 0 else 100 - df["h4_rsi"]
    f["d1_rsi_dir"] = df["d1_rsi"] if d > 0 else 100 - df["d1_rsi"]
    hist, prev = df["h4_hist"] * d, df["h4_hist_prev"] * d
    f["h4_macd_pos"] = hist > 0
    f["h4_macd_rising"] = (hist > 0) & (hist > prev)
    f["h4_strict"] = (df["h4_strict_up"] if d > 0 else df["h4_strict_dn"]).fillna(False).astype(bool)
    edge = df["h4_top"] if d > 0 else df["h4_bot"]
    f["ext_h4"] = d * (c - edge) / df["h4_atr"]                       # 離 4H 通道幾個 4H ATR
    f["ext_ema20"] = d * (c - df["ema20"]) / df["atr"]                # 離 1H EMA20 幾個 1H ATR
    f["brk_atr"] = d * (c - (df["prior_high"] if d > 0 else df["prior_low"])) / df["atr"]
    f["new_n"] = df["new_high_n"] if d > 0 else df["new_low_n"]
    f["ret24_dir"] = d * df["ret24"]
    f["ret168_dir"] = d * df["ret168"]
    f["h4_ret6_dir"] = d * df["h4_ret6"]
    f["close_pos_dir"] = df["close_pos"] if d > 0 else 1 - df["close_pos"]
    f["body_dir"] = d * df["body_atr"]
    f["d1_above_ema50"] = above(df["d1_close"], df["d1_ema50"], d)
    f["d1_above_ema200"] = above(df["d1_close"], df["d1_ema200"], d)
    f["d1_vegas"] = above(df["d1_close"], warm(df["d1_top"] if d > 0 else df["d1_bot"], 0), d).where(
        df["age_days"] >= 170)
    f["h4_above_ema200"] = above(df["h4_close"], df["h4_ema200"], d)
    f["btc_trend_dir"] = d * df["btc_trend"]                          # BTC 4H：+1 順向在通道外，-1 反向
    f["btc_ret24_dir"] = d * df["btc_ret24"]
    f["breadth_dir"] = df["breadth_up"] if d > 0 else df["breadth_dn"]  # 全部幣裡 4H 在通道同一側的比例
    for col in ("vol_ratio", "squeeze", "adx", "h4_adx", "d1_adx", "age_days"):
        f[col] = df[col]
    f["hour"] = df["bar_close"].dt.hour
    return f


def simulate(df: pd.DataFrame, idx: np.ndarray, d: int, stop: np.ndarray, atr4: np.ndarray) -> dict:
    price = df["close"].to_numpy()[idx]
    risk = np.abs(price - stop)
    hi, lo, cl, op = (df[k].to_numpy() for k in ("high", "low", "close", "open"))
    isc = np.ones(len(hi), np.bool_)
    zero = np.zeros(len(idx))
    ds = np.full(len(idx), d, np.int64)
    out = {}
    for name, (tp_r, portions, trail_k, max_bars) in EXITS.items():
        tps = np.stack([price + d * k * risk for k in tp_r], axis=1)
        r, hit, j, kind, mae = sim_many(op, hi, lo, cl, isc, idx + 1, ds, price, risk, stop.astype(float), tps,
                                        np.array(portions), 0, trail_k * atr4, FEE, 0.0, max_bars, 0, 0, zero,
                                        0.0, 0.0, zero, 0, cl, cl, False)
        out[f"r_{name}"], out[f"hit_{name}"], out[f"j_{name}"], out[f"kind_{name}"] = r, hit, j, kind
    out["risk_pct"] = risk / price
    return out


def _coin(args):
    coin, market = args
    raw = load(coin, "1h")
    if len(raw) < WARMUP * 4:
        return None
    df = features(raw.reset_index(drop=True), market)
    c = df["close"]
    valid = (np.arange(len(df)) >= WARMUP * 4) & df["h4_top"].notna().to_numpy()  # 4H 均線要先累積
    valid[-1] = False
    rng = np.random.default_rng(zlib.crc32(coin.encode()))
    parts = []
    for d in (1, -1):
        if d > 0:
            regime = (c > df[["ema20", "ema50", "ema144", "ema169"]].max(axis=1)) & df["h4_up"].fillna(False).astype(bool)
            brk = c > df["prior_high"]
            swing = df["swing_low"].where(df["swing_low"] < c, df["low24"])
            stop = swing * (1 - BUFFER)
        else:
            regime = (c < df[["ema20", "ema50", "ema144", "ema169"]].min(axis=1)) & df["h4_down"].fillna(False).astype(bool)
            brk = c < df["prior_low"]
            swing = df["swing_high"].where(df["swing_high"] > c, df["high24"])
            stop = swing * (1 + BUFFER)
        risk_pct = (d * (c - stop) / c).to_numpy()
        ok_risk = (risk_pct >= RISK_MIN) & (risk_pct <= RISK_MAX)
        base = regime.to_numpy() & ok_risk & valid
        rand = base & (rng.random(len(df)) < 0.1)
        sf = side_features(df, d)
        for kind_name, mask in (("breakout", base & brk.to_numpy()), ("random", rand)):
            idx = np.nonzero(mask)[0]
            if not len(idx):
                continue
            res = simulate(df, idx, d, stop.to_numpy()[idx], df["h4_atr"].to_numpy()[idx])
            part = sf.iloc[idx].reset_index(drop=True)
            part.insert(0, "set", kind_name)
            part.insert(1, "coin", coin)
            part.insert(2, "side", d)
            part.insert(3, "i", idx)
            part.insert(4, "time", df["bar_close"].to_numpy()[idx])
            for key, val in res.items():
                part[key] = val
            parts.append(part)
    return pd.concat(parts, ignore_index=True) if parts else None


def market_frame(workers: int) -> pd.DataFrame:
    with ProcessPoolExecutor(workers) as ex:
        series = [s for s in ex.map(trend_series, TOP100) if s is not None]
    m = series[0]
    for s in series[1:]:
        m = m.merge(s, on="avail", how="outer")
    m = m.sort_values("avail").reset_index(drop=True)
    vals = m.drop(columns="avail")
    n = vals.notna().sum(axis=1)
    out = pd.DataFrame({"avail": m["avail"],
                        "breadth_up": (vals == 1).sum(axis=1) / n, "breadth_dn": (vals == -1).sum(axis=1) / n})
    btc = load("BTC", "1h")
    bh = htf(btc, "4h", 4)
    b = pd.DataFrame({"avail": bh["avail"], "btc_trend": np.where(bh["up"], 1, np.where(bh["down"], -1, 0)),
                      "btc_ret24": bh["close"] / bh["close"].shift(6) - 1})
    out = pd.merge_asof(out, b, on="avail", direction="backward")
    return out[n.to_numpy() >= 5].reset_index(drop=True)


def collect(workers: int = 4) -> pd.DataFrame:
    market = market_frame(workers)
    with ProcessPoolExecutor(workers) as ex:
        parts = [p for p in ex.map(_coin, [(c, market) for c in TOP100]) if p is not None]
    t = pd.concat(parts, ignore_index=True)
    t["time"] = pd.to_datetime(t["time"], utc=True)
    OUT.mkdir(exist_ok=True)
    t.to_pickle(OUT / "breakout_signals.pkl")
    return t


# ---------- 分析 ----------

def take(t: pd.DataFrame, exit_name: str) -> pd.DataFrame:
    """同一個幣同時只持有一張；資料結束還沒出場的不算。"""
    t = t.sort_values(["coin", "side", "i"]).reset_index(drop=True)
    out = []
    for _, g in t.groupby(["coin", "side"], sort=False):
        # 多空分開算（實盤同一個幣不會同時有多空，但兩邊的訊號本來就不會同時成立）
        keep = busy_filter(np.zeros(len(g), np.int64), g["i"].to_numpy(), g[f"j_{exit_name}"].to_numpy(),
                           g[f"kind_{exit_name}"].to_numpy())
        out.append(g[keep])
    return pd.concat(out, ignore_index=True) if out else t.iloc[:0]


def stats(t: pd.DataFrame, exit_name: str) -> dict:
    r = t[f"r_{exit_name}"].to_numpy()
    if len(r) == 0:
        return {"n": 0}
    key = (t["time"].dt.floor("D").astype(str) + t["side"].astype(str)).to_numpy()
    mean, se, _, _, ncl = cluster_stats(r, key)
    return {"n": len(r), "win": (r > 0).mean(), "tp1": (t[f"hit_{exit_name}"] >= 1).mean(), "exp_r": mean,
            "se_cl": se, "t_cl": mean / se if se > 0 else np.nan, "clusters": ncl}


def by_period(t: pd.DataFrame, exit_name: str) -> dict:
    t = take(t, exit_name)
    out = {}
    for label, m in (("all", np.ones(len(t), bool)), ("is", (t["time"] < SPLIT).to_numpy()),
                     ("oos", (t["time"] >= SPLIT).to_numpy())):
        for k, v in stats(t[m], exit_name).items():
            out[f"{label}_{k}"] = v
    return out




CONT = ["rsi_dir", "h4_rsi_dir", "d1_rsi_dir", "ext_h4", "ext_ema20", "brk_atr", "ret24_dir", "ret168_dir",
        "h4_ret6_dir", "close_pos_dir", "body_dir", "btc_ret24_dir", "breadth_dir", "vol_ratio", "squeeze", "adx",
        "h4_adx", "d1_adx", "age_days", "risk_pct", "new_n"]
BOOL = ["h4_macd_pos", "h4_macd_rising", "h4_strict", "d1_above_ema50", "d1_above_ema200", "d1_vegas",
        "h4_above_ema200", "btc_trend_dir"]


def group_original(t: pd.DataFrame) -> pd.Series:
    """外部訊號群原版的條件（加在基礎訊號上）。"""
    return t["rsi_dir"].between(60, 80) & t["h4_macd_rising"].astype(bool) & t["risk_pct"].between(0.03, 0.12)


def baseline_table(t: pd.DataFrame) -> pd.DataFrame:
    rows = []
    sets = {"基礎突破（寬鬆）": t[t["set"] == "breakout"],
            "外部訊號群原版": t[(t["set"] == "breakout") & group_original(t)],
            "隨機進場（同趨勢條件）": t[t["set"] == "random"]}
    for name, s in sets.items():
        for side_name, sd in (("多", s[s["side"] == 1]), ("空", s[s["side"] == -1]), ("多空", s)):
            for e in EXITS:
                rows.append({"set": name, "side": side_name, "exit": e, **by_period(sd, e)})
    return pd.DataFrame(rows)


def quintiles(t: pd.DataFrame, exit_name: str) -> pd.DataFrame:
    """在已經套用「一個幣一張」的基礎訊號上，每個特徵用 2020-23 的五分位切開，看各組的 R 和勝率。"""
    base = take(t[t["set"] == "breakout"], exit_name)
    is_ = base["time"] < SPLIT
    rows = []
    for col in CONT + BOOL:
        x = base[col].astype(float)
        if col in BOOL:
            bins = x
            labels = sorted(x.dropna().unique())
        else:
            edges = np.unique(np.nanquantile(x[is_], [0, 0.2, 0.4, 0.6, 0.8, 1]))
            if len(edges) < 3:
                continue
            edges[0], edges[-1] = -np.inf, np.inf
            bins = pd.cut(x, edges, labels=False)
            labels = range(len(edges) - 1)
        for q in labels:
            m = (bins == q).to_numpy()
            lo_, hi_ = (q, q) if col in BOOL else (edges[q], edges[q + 1])
            row = {"feature": col, "bin": q, "lo": lo_, "hi": hi_}
            for label, pm in (("is", is_.to_numpy()), ("oos", ~is_.to_numpy())):
                for k, v in stats(base[m & pm], exit_name).items():
                    row[f"{label}_{k}"] = v
            rows.append(row)
    return pd.DataFrame(rows)


def threshold_scan(t: pd.DataFrame, exit_name: str) -> pd.DataFrame:
    """每個特徵試「只留 >= 門檻」和「只留 <= 門檻」（門檻 = 2020-23 的 20/40/60/80 分位），重新套一個幣一張。"""
    b = t[t["set"] == "breakout"]
    is_ = b["time"] < SPLIT
    rows = []
    ref = by_period(b, exit_name)
    for col in CONT + BOOL:
        x = b[col].astype(float)
        if col in BOOL:
            cands = [(">=", 1.0), ("<=", 0.0)] if col != "btc_trend_dir" else [(">=", 1.0), (">=", 0.0), ("<=", 0.0)]
        else:
            qs = np.nanquantile(x[is_], [0.2, 0.4, 0.6, 0.8])
            cands = [(">=", q) for q in qs] + [("<=", q) for q in qs]
        for op, thr in cands:
            m = (x >= thr) if op == ">=" else (x <= thr)
            res = by_period(b[m.to_numpy()], exit_name)
            rows.append({"feature": col, "op": op, "thr": thr, **res,
                         "is_gain": res.get("is_exp_r", np.nan) - ref["is_exp_r"],
                         "oos_gain": res.get("oos_exp_r", np.nan) - ref["oos_exp_r"]})
    return pd.DataFrame(rows)


RULES = {
    "外部訊號群原版": group_original,
    "改良A：RSI>=75＋量>=2倍": lambda s: (s["rsi_dir"] >= 75) & (s["vol_ratio"] >= 2),
    "改良B：改良A＋4H MACD 柱放大": lambda s: (s["rsi_dir"] >= 75) & (s["vol_ratio"] >= 2)
    & s["h4_macd_rising"].astype(bool),
}


def max_positions(s: pd.DataFrame, n: int | None, exit_name: str = "grp") -> pd.DataFrame:
    """同時最多持有 n 張（所有幣合計）：已經滿了就跳過新訊號。出場時間用持有根數換算（1H）。"""
    s = s.sort_values("time")
    if n is None:
        return s
    ex = (s["time"] + pd.to_timedelta(s[f"j_{exit_name}"] - s["i"], unit="h")).to_numpy()
    tm = s["time"].to_numpy()
    keep = np.zeros(len(s), bool)
    held: list = []
    for k in range(len(s)):
        held = [x for x in held if x > tm[k]]
        if len(held) < n:
            keep[k] = True
            held.append(ex[k])
    return s[keep]


def first_per_day(s: pd.DataFrame, k: int) -> pd.DataFrame:
    """每個台北日只接最早的 k 筆。"""
    s = s.sort_values("time")
    return s.groupby((s["time"] + pd.Timedelta(hours=8)).dt.floor("D")).head(k)


def rule_tables(t: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, years = [], []
    for sides, label in (((1,), "多"), ((-1,), "空"), ((1, -1), "多空")):
        for name, f in RULES.items():
            for e in EXITS:
                parts = []
                for d in sides:
                    s = t[(t["set"] == "breakout") & (t["side"] == d)]
                    parts.append(take(s[f(s).to_numpy()], e))
                s = pd.concat(parts)
                caps = [("全部接", s)] + [(f"同時最多{n}張", max_positions(s, n, e)) for n in (3, 5, 10)] \
                    + [(f"每天最早{k}筆", first_per_day(s, k)) for k in (1, 3)]
                for cap, c in caps:
                    is_ = (c["time"] < SPLIT).to_numpy()
                    row = {"side": label, "rule": name, "exit": e, "cap": cap}
                    for p, m in (("is", is_), ("oos", ~is_)):
                        for k, v in stats(c[m], e).items():
                            row[f"{p}_{k}"] = v
                    rows.append(row)
                if e == "grp":
                    for y, g in s.groupby(s["time"].dt.year):
                        years.append({"side": label, "rule": name, "year": y, **stats(g, e)})
    return pd.DataFrame(rows), pd.DataFrame(years)


def pionex_frame(coin: str) -> pd.DataFrame:
    path = Path("data") / f"{coin}_USDT_PERP_60M.csv.gz"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True)
    return df.reset_index(drop=True)


def collect_pionex(workers: int = 4) -> pd.DataFrame:
    """派網 1H（約 14 個月）重跑同樣的訊號。"""
    global load
    load = lambda coin, interval: pionex_frame(coin)  # noqa: E731（_coin、market_frame 都用模組層級的 load）
    market = market_frame(workers)
    with ProcessPoolExecutor(workers, initializer=_use_pionex) as ex:
        parts = [p for p in ex.map(_coin, [(c, market) for c in TOP100]) if p is not None]
    t = pd.concat(parts, ignore_index=True)
    t["time"] = pd.to_datetime(t["time"], utc=True)
    t.to_pickle(OUT / "breakout_signals_pionex.pkl")
    return t


def _use_pionex():
    global load
    load = lambda coin, interval: pionex_frame(coin)  # noqa: E731


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "collect"
    if cmd == "collect":
        t = collect()
        print(t.groupby(["set", "side"]).size())
    elif cmd == "analyze":
        t = pd.read_pickle(OUT / "breakout_signals.pkl")
        baseline_table(t).to_csv(OUT / "breakout_baseline.csv", index=False)
        long_ = t[t["side"] == 1]
        quintiles(long_, "grp").to_csv(OUT / "breakout_quintiles_long.csv", index=False)
        threshold_scan(long_, "grp").to_csv(OUT / "breakout_thresholds_long.csv", index=False)
        rt, yr = rule_tables(t)
        rt.to_csv(OUT / "breakout_rules.csv", index=False)
        yr.to_csv(OUT / "breakout_rules_by_year.csv", index=False)
    elif cmd == "pionex":
        t = collect_pionex()
        rt, yr = rule_tables(t)
        rt.to_csv(OUT / "breakout_rules_pionex.csv", index=False)
