"""美股/商品代幣為什麼維加斯做不好：用現貨（股票本身、期貨）的歷史價格補長，再回測（只做分析，不改實盤規則）。

python -m research.rwa_underlying fetch      # 從 Yahoo Finance 抓現貨 1 小時（約 2 年）和日線（上市以來），快取到 data/yahoo/
python -m research.rwa_underlying run        → output/rwa_underlying.csv、rwa_underlying_trades.csv.gz、rwa_token_bars.csv
python -m research.rwa_underlying summary    （只重算彙總，不重抓派網）
python -m research.rwa_underlying random     → output/rwa_underlying_random.csv（隨機進場對照）

資料：
- 派網代幣（XXXX + X）對應的美股/ETF 現貨；商品用 COMEX/NYMEX 期貨連續合約（XAU→GC=F 等）。
- Yahoo 1 小時 K 線最多回查 730 天，所以 2H/4H 只能補到約 2 年；日線可以回到上市（大多 2000 年前後）。
- 美股只有正規交易時段（美東 9:30-16:00）。2H/4H K 棒在每個交易日內從開盤切：
  4H = 9:30-13:30、13:30-16:00；2H = 9:30-11:30、11:30-13:30、13:30-16:00。不跨日、不跨週末。
- 期貨幾乎 24 小時（每天休 1 小時、週末休市），照 UTC 切 2H/4H，沒有成交的時段不產生 K 棒。

規則和實盤一樣：RSI 54-70/30-46、離通道 3 ATR 內、2H 加 ADX < 24；止損 1.5 ATR、TP 1/3/5R 各平 1/4、
TP1 後保本、TP2 後移到 TP1、最後 1/4 追蹤 3 ATR；同一個標的同一個週期同時一張，2H/4H 同方向只做一張。
手續費每邊 0.05%。跳空直接穿過止損時用開盤價出場（比 1R 虧更多）。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

import research.entry_research as er
from research.exit_unify import account, dedupe, per_interval
from research.stop_research import LIVE_ATR, MAX_BARS, MIN_STOP_PCT, load_frame, sim_many
from grid_trading.signals.strategy import LONG
from grid_trading.signals.vegas_trend import VegasTrend
from grid_trading.universe import RWA_COMMODITIES, RWA_STOCKS

CACHE = Path("data/yahoo")
UA = {"User-Agent": "Mozilla/5.0"}
PLAN = ((1.0, 3.0, 5.0), (0.25, 0.25, 0.25), 3.0)  # 實盤出場（4H、2H 一樣）
FEE = 0.0005
# 派網代號 → Yahoo 代號（股票預設是去掉結尾的 X）
SPECIAL = {"XAU": "GC=F", "XAG": "SI=F", "XPT": "PL=F", "XPD": "PA=F", "WTI": "CL=F", "COPPER": "HG=F",
           "USOX": "USO", "SKHX": "000660.KS", "SPCX": "SPCX"}
FUTURES = {"GC=F", "SI=F", "PL=F", "PA=F", "CL=F", "HG=F"}


def ticker(coin: str) -> str:
    return SPECIAL.get(coin, coin[:-1] if coin.endswith("X") else coin)


def _get(tk: str, params: dict) -> pd.DataFrame:
    for attempt in range(4):
        r = requests.get(f"https://query1.finance.yahoo.com/v8/finance/chart/{tk}", params=params, headers=UA, timeout=30)
        if r.status_code == 429:
            time.sleep(5 * (attempt + 1))
            continue
        res = r.json()["chart"]["result"]
        if not res or "timestamp" not in res[0]:
            return pd.DataFrame()
        q = res[0]["indicators"]["quote"][0]
        df = pd.DataFrame({"time": pd.to_datetime(res[0]["timestamp"], unit="s", utc=True),
                           **{k: q[k] for k in ("open", "high", "low", "close", "volume")}})
        return df.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)
    return pd.DataFrame()


def fetch() -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    only = set(sys.argv[2:])
    for coin in RWA_COMMODITIES + RWA_STOCKS:
        if only and coin not in only:
            continue
        tk = ticker(coin)
        for name, params in (("1h", {"interval": "1h", "range": "730d", "includePrePost": "false"}),
                             ("1d", {"interval": "1d", "period1": 0, "period2": int(time.time())})):
            df = _get(tk, params)
            df.to_csv(CACHE / f"{coin}_{name}.csv.gz", index=False)
            print(coin, tk, name, len(df), df["time"].min() if len(df) else "-", file=sys.stderr)
            time.sleep(0.6)


def load_yahoo(coin: str, name: str) -> pd.DataFrame:
    path = CACHE / f"{coin}_{name}.csv.gz"
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    if df.empty:
        return df
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df


def _agg(df: pd.DataFrame, key) -> pd.DataFrame:
    g = df.groupby(key, sort=True)
    out = pd.DataFrame({"time": g["time"].first(), "open": g["open"].first(), "high": g["high"].max(),
                        "low": g["low"].min(), "close": g["close"].last(), "volume": g["volume"].sum()})
    return out.sort_values("time").reset_index(drop=True)


def session_bars(df: pd.DataFrame, interval: str, tz: str = "America/New_York", open_min: int = 570) -> pd.DataFrame:
    """股票：每個交易日內從開盤切 2H/4H，最後不足一根的併進前一根（美股 9:30 開盤；SK 海力士用首爾 9:00）。"""
    ny = df["time"].dt.tz_convert(tz)
    mins = (ny.dt.hour * 60 + ny.dt.minute - open_min)
    df = df[(mins >= 0) & (mins < 390)].copy()
    mins = mins[df.index]
    size = 240 if interval == "4H" else 120
    slot = (mins // size).clip(upper=390 // size - 1)
    key = ny[df.index].dt.strftime("%Y-%m-%d") + "_" + slot.astype(str)
    return _agg(df, key.to_numpy())


def futures_bars(df: pd.DataFrame, interval: str) -> pd.DataFrame:
    size = 4 if interval == "4H" else 2
    key = df["time"].dt.floor(f"{size}h")
    out = _agg(df, key.to_numpy())
    out["time"] = sorted(set(key))
    return out


def underlying_frame(coin: str, interval: str) -> pd.DataFrame:
    if interval == "1D":
        return load_yahoo(coin, "1d")
    h = load_yahoo(coin, "1h")
    if h.empty:
        return h
    if ticker(coin) in FUTURES:
        return futures_bars(h, interval)
    if ticker(coin).endswith(".KS"):
        return session_bars(h, interval, "Asia/Seoul", 540)
    return session_bars(h, interval)


def signals(coin: str, df: pd.DataFrame, interval: str, label: str) -> pd.DataFrame | None:
    strat = VegasTrend()
    if len(df) < strat.min_bars + 50:
        return None
    p = strat.prepare(df.reset_index(drop=True)).reset_index(drop=True)
    idx, ds = [], []
    for i in np.flatnonzero(strat.candidates(p)):
        if i < strat.min_bars or i + 1 >= len(p):
            continue
        hit = strat.check(p, int(i))
        if hit is None or not strat.accept(p, int(i), interval if interval != "1D" else "4H"):
            continue
        idx.append(int(i))
        ds.append(1 if hit[0] == LONG else -1)
    if not idx:
        return None
    idx, ds = np.array(idx), np.array(ds)
    arr = {k: p[k].to_numpy(float) for k in ("open", "high", "low", "close", "atr")}
    er.PLANS = {interval: PLAN}
    s = er.simulate(arr, idx, ds, interval, FEE, 0.0)
    r_gap = gap_sim(arr, idx, ds)
    t = pd.to_datetime(p["time"], utc=True)
    # 下一根開盤相對訊號收盤的跳空（以 1R 計）、這筆最差是否超過 1R
    gap_r = ds * (arr["open"][np.minimum(idx + 1, len(p) - 1)] - arr["close"][idx]) / (1.5 * arr["atr"][idx])
    return pd.DataFrame({"source": label, "coin": coin, "interval": interval, "i": idx, "side": ds,
                         "time": t.iloc[idx + 1].to_numpy(), "exit_t": t.iloc[np.minimum(s["j"], len(t) - 1)].to_numpy(),
                         "r": r_gap, "r_nogap": s["r"], "hit": s["hit"], "kind": s["kind"], "j": s["j"], "gap_r": gap_r,
                         "closed": closed_bar(t.iloc[idx], interval, coin in RWA_COMMODITIES).to_numpy(),
                         "atr_pct": arr["atr"][idx] / arr["close"][idx]})


def gap_sim(arr: dict, idx: np.ndarray, ds: np.ndarray) -> np.ndarray:
    """和 er.simulate 一樣，但開盤已經跳過止損時用開盤價成交（現貨隔夜、週末跳空）。"""
    price, atr = arr["close"][idx], arr["atr"][idx]
    tp_r, portions, trail_k = PLAN
    risk = np.maximum(LIVE_ATR * atr, MIN_STOP_PCT * price)
    stop0 = price - ds * risk
    tps = np.stack([price + ds * k * risk for k in tp_r], axis=1)
    zero = np.zeros(len(idx))
    cl = arr["close"]
    r, *_ = sim_many(arr["open"], arr["high"], arr["low"], cl, np.ones(len(cl), np.bool_), idx + 1,
                     ds.astype(np.int64), price, risk, stop0, tps, np.array(portions), 0, trail_k * atr, FEE, 0.0,
                     MAX_BARS, 0, 0, zero, 0.0, 0.0, zero, 0, cl, cl, True)
    return r


def closed_bar(t: pd.Series, interval: str, commodity: bool = False) -> pd.Series:
    """K 棒的中間時刻是不是現貨休市（美股：美東 9:30-16:00 以外或週末）。"""
    half = {"4H": 2, "2H": 1}.get(interval, 0)
    mid = t.dt.tz_convert("America/New_York") + pd.Timedelta(hours=half)
    if interval == "1D":
        return pd.Series(False, index=t.index)
    if commodity:  # 期貨：週五 17:00 到週日 18:00 休市
        wd, h = mid.dt.weekday, mid.dt.hour
        return (wd == 5) | ((wd == 4) & (h >= 17)) | ((wd == 6) & (h < 18))
    return ~((mid.dt.weekday < 5) & (mid.dt.hour * 60 + mid.dt.minute >= 570) & (mid.dt.hour < 16))


def bar_profile(df: pd.DataFrame, interval: str, commodity: bool) -> dict:
    """派網代幣的 K 棒裡有多少根落在現貨休市時段、休市時段的平均振幅相對開市時段、ATR 佔價格的比例。"""
    rng = (df["high"] - df["low"]) / df["close"]
    open_ = ~closed_bar(df["time"], interval, commodity)
    return {"bars": len(df), "share_closed": 1 - open_.mean(),
            "range_closed_vs_open": rng[~open_].mean() / rng[open_].mean() if open_.any() else np.nan}


def build() -> pd.DataFrame:
    parts, prof = [], []
    for coin in RWA_COMMODITIES + RWA_STOCKS:
        grp = "商品" if coin in RWA_COMMODITIES else "美股"
        for iv in ("4H", "2H", "1D"):
            got = signals(coin, underlying_frame(coin, iv), iv, "現貨")
            if got is not None:
                parts.append(got.assign(group=grp))
        for iv in ("4H", "2H"):
            try:
                tok = load_frame(coin, iv, "pionex")
            except Exception as e:  # noqa: BLE001
                print(coin, iv, "pionex", e, file=sys.stderr)
                continue
            if len(tok):
                tok["time"] = pd.to_datetime(tok["time"], utc=True)
                prof.append({"coin": coin, "group": grp, "interval": iv, **bar_profile(tok, iv, grp == "商品")})
                got = signals(coin, tok, iv, "派網代幣")
                if got is not None:
                    parts.append(got.assign(group=grp))
    allt = pd.concat(parts, ignore_index=True)
    for c in ("time", "exit_t"):
        allt[c] = pd.to_datetime(allt[c], utc=True)
    Path("output").mkdir(exist_ok=True)
    allt.to_csv("output/rwa_underlying_trades.csv.gz", index=False)
    pd.DataFrame(prof).to_csv("output/rwa_token_bars.csv", index=False)
    return allt


def book(tr: pd.DataFrame, intervals) -> pd.DataFrame:
    tr = tr[tr["interval"].isin(intervals)]
    if tr.empty:
        return tr
    return dedupe(pd.concat([per_interval(tr[tr["interval"] == iv]) for iv in intervals], ignore_index=True))


def summarize(allt: pd.DataFrame) -> pd.DataFrame:
    rows = []
    end = allt["time"].max() + pd.Timedelta(days=1)
    tok = allt[allt["source"] == "派網代幣"]
    tok_start = tok.groupby("coin")["time"].min()

    def add(name, tr, a, b, **kw):
        x = tr[(tr["time"] >= a) & (tr["time"] < b)] if len(tr) else tr
        acc = account(x, a, b) if len(x) else {"trades": 0}
        if acc.get("trades"):
            acc["gap_through"] = (x["r"] < -1.15).mean()
            acc["exp_r_nogap"] = x["r_nogap"].mean()
            acc["n_long"] = int((x["side"] > 0).sum())
            acc["long_r"] = x.loc[x["side"] > 0, "r"].mean()
            acc["short_r"] = x.loc[x["side"] < 0, "r"].mean()
            acc["atr_pct"] = x["atr_pct"].median() * 100
        rows.append({"test": name, **kw, **acc})

    for grp in ("美股", "商品"):
        g = allt[allt["group"] == grp]
        u, tk = g[g["source"] == "現貨"], g[g["source"] == "派網代幣"]
        u_start = u.loc[u["interval"] != "1D", "time"].min()
        # 只看同一批標的、各自在派網上市之後的現貨訊號
        both = u[u["coin"].isin(tok_start.index)]
        both = both[both["time"] >= both["coin"].map(tok_start)]
        ts = tk["time"].min()
        for ivs, label in ((("4H",), "4H"), (("2H",), "2H"), (("4H", "2H"), "4H+2H")):
            ub, bb, tb = book(u, ivs), book(both, ivs), book(tk, ivs)
            add("A 現貨（1 小時資料全部）", ub, u_start, end, group=grp, interval=label)
            add("B 現貨（同標的、代幣上市後）", bb, ts, end, group=grp, interval=label)
            add("C 派網代幣", tb, ts, end, group=grp, interval=label)
            if len(tb):
                add("C1 派網代幣：訊號在現貨開市時段", tb[~tb["closed"].astype(bool)], ts, end, group=grp, interval=label)
                add("C2 派網代幣：訊號在現貨休市時段", tb[tb["closed"].astype(bool)], ts, end, group=grp, interval=label)
            for y in range(u_start.year, end.year + 1):
                a, b = pd.Timestamp(f"{y}-01-01", tz="UTC"), pd.Timestamp(f"{y + 1}-01-01", tz="UTC")
                add(f"A 現貨 {y}", ub, max(a, u_start), min(b, end), group=grp, interval=label)
        d = book(u, ("1D",))
        if len(d):
            d0 = d["time"].min()
            add("D 現貨日線（上市以來）", d, d0, end, group=grp, interval="1D")
            for a, b, nm in ((d0, pd.Timestamp("2010-01-01", tz="UTC"), "-2009"),
                             (pd.Timestamp("2010-01-01", tz="UTC"), pd.Timestamp("2018-01-01", tz="UTC"), "2010-17"),
                             (pd.Timestamp("2018-01-01", tz="UTC"), pd.Timestamp("2024-01-01", tz="UTC"), "2018-23"),
                             (pd.Timestamp("2024-01-01", tz="UTC"), end, "2024+")):
                add(f"D 現貨日線 {nm}", d, a, b, group=grp, interval="1D")
    return pd.DataFrame(rows)


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    if cmd == "fetch":
        fetch()
        return
    if cmd == "random":
        random_bench()
        return
    if cmd == "run":
        allt = build()
    else:
        allt = pd.read_csv("output/rwa_underlying_trades.csv.gz")
        for c in ("time", "exit_t"):
            allt[c] = pd.to_datetime(allt[c], utc=True)
    out = summarize(allt)
    out.to_csv("output/rwa_underlying.csv", index=False)
    cols = ["test", "group", "interval", "trades", "win_rate", "exp_r", "se_cluster", "exp_r_nogap", "n_long", "long_r",
            "short_r", "gap_through", "atr_pct", "sum_r_per_year", "sharpe", "max_dd_pct"]
    with pd.option_context("display.width", 250, "display.max_rows", 300):
        print(out[[c for c in cols if c in out]].round(3).to_string())



def random_bench(n_per_coin: int = 400, seed: int = 0) -> pd.DataFrame:
    """美股做多是策略的功勞還是股票本來就漲：同一批股票、同樣出場，隨機 K 棒進場做多/做空，
    以及「只在通道上方且通道向上（策略的趨勢條件）時」隨機進場，和實際訊號比。"""
    rng = np.random.default_rng(seed)
    rows = []
    allt = pd.read_csv("output/rwa_underlying_trades.csv.gz")
    allt["time"] = pd.to_datetime(allt["time"], utc=True)
    for grp, coins in (("美股", RWA_STOCKS), ("商品", RWA_COMMODITIES)):
        for iv in ("1D", "4H", "2H"):
            acc = {"random_long": [], "random_short": [], "trend_long": [], "trend_short": []}
            for coin in coins:
                df = underlying_frame(coin, iv)
                if len(df) < 300:
                    continue
                p = VegasTrend().prepare(df.reset_index(drop=True)).reset_index(drop=True)
                arr = {k: p[k].to_numpy(float) for k in ("open", "high", "low", "close", "atr")}
                top = np.maximum(p["ema144"], p["ema169"]).to_numpy()
                bot = np.minimum(p["ema144"], p["ema169"]).to_numpy()
                up = (p["close"].to_numpy() > top) & (p["ema12"].to_numpy() > top) & (p["ema144"] > p["ema169"]).to_numpy()
                dn = (p["close"].to_numpy() < bot) & (p["ema12"].to_numpy() < bot) & (p["ema144"] < p["ema169"]).to_numpy()
                ok = np.arange(len(p)) >= 200
                ok[-2:] = False
                for name, mask, d in (("random_long", ok, 1), ("random_short", ok, -1),
                                      ("trend_long", ok & up, 1), ("trend_short", ok & dn, -1)):
                    cand = np.flatnonzero(mask)
                    if len(cand) == 0:
                        continue
                    idx = np.sort(rng.choice(cand, min(n_per_coin, len(cand)), replace=False))
                    acc[name].append(gap_sim(arr, idx, np.full(len(idx), d)))
            sig = allt[(allt["source"] == "現貨") & (allt["group"] == grp) & (allt["interval"] == iv)]
            row = {"group": grp, "interval": iv, "signal_long": sig.loc[sig["side"] > 0, "r"].mean(),
                   "signal_short": sig.loc[sig["side"] < 0, "r"].mean()}
            for k, v in acc.items():
                row[k] = np.concatenate(v).mean() if v else np.nan
            rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv("output/rwa_underlying_random.csv", index=False)
    print(out.round(3).to_string())
    return out


if __name__ == "__main__":
    main()
