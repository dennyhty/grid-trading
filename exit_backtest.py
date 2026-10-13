"""vegas_trend 止損 / 止盈設定研究（只做分析，不改實盤規則）。

python exit_backtest.py --interval 4H [--fee 0.0005]

需要另外 pip install numba（只有這支研究程式用到，網站不需要）。

訊號本身不變（和實盤同一套 vegas_trend 進場條件），只換出場方式：
- 止損：ATR 倍數、訊號 K 棒最低/最高價（可加緩衝或最小距離）、前 N 根最低/最高價。
- 止盈：TP 的 R 倍數、各段平倉比例、保護止損怎麼移、最後一段改成追蹤止損。

每種組合都照回測的規則跑：同一個幣同時只持有一張單、同一根 K 棒同時碰到止損和 TP 時先算止損、
最多持有 240 根。用幣安 USDT 永續月檔（2020 年起），2H 和實盤一樣由 1H 合成。

輸出 output/exit_grid_{週期}.csv（每個組合 × 期間的統計）、隨機進場對照 output/exit_random_{週期}.csv，
和幾個主要組合的逐筆交易 output/exit_trades_*.csv.gz。
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

from grid_trading.binance_history import load
from grid_trading.history import resample
from grid_trading.signals.strategy import LONG
from grid_trading.signals.vegas_trend import VegasTrend
from grid_trading.universe import TOP100

MAX_BARS = 240
SPLIT = pd.Timestamp("2024-01-01", tz="UTC")
MIN_STOP_PCT = 0.001  # 止損距離小於進場價 0.1% 時當成 0.1%，不然 R 會被手續費放大到沒意義

# 止損：(名稱, 種類, 參數)
STOPS = [
    ("ATR1.0", "atr", 1.0),
    ("ATR1.5", "atr", 1.5),  # 實盤
    ("ATR2.0", "atr", 2.0),
    ("ATR2.5", "atr", 2.5),
    ("K棒極值", "candle", 0.0),
    ("K棒極值+0.2ATR", "candle", 0.2),
    ("K棒極值+0.5ATR", "candle", 0.5),
    ("K棒極值(至少1ATR)", "candle_floor", 1.0),
    ("K棒極值(至少1.5ATR)", "candle_floor", 1.5),
    ("前3根極值", "swing", 3),
    ("前5根極值", "swing", 5),
    ("前10根極值", "swing", 10),
]

# 止盈：(名稱, TP 的 R 倍數, 各段比例, 保護止損模式, 追蹤止損 ATR 倍數)
# 保護模式 0：TP1 後移到保本、TP2 後移到 TP1（實盤）；1：TP2 後才移到保本；2：不移
EXITS = [
    ("1/2/3R 各1/3", (1, 2, 3), (1 / 3, 1 / 3, 1 / 3), 0, 0.0),  # 實盤
    ("1/2/4R 各1/3", (1, 2, 4), (1 / 3, 1 / 3, 1 / 3), 0, 0.0),
    ("1/3/5R 各1/3", (1, 3, 5), (1 / 3, 1 / 3, 1 / 3), 0, 0.0),
    ("1.5/3/4.5R 各1/3", (1.5, 3, 4.5), (1 / 3, 1 / 3, 1 / 3), 0, 0.0),
    ("2/4/6R 各1/3", (2, 4, 6), (1 / 3, 1 / 3, 1 / 3), 0, 0.0),
    ("0.5/1/2R 各1/3", (0.5, 1, 2), (1 / 3, 1 / 3, 1 / 3), 0, 0.0),
    ("1/2/3R 50/25/25", (1, 2, 3), (0.5, 0.25, 0.25), 0, 0.0),
    ("1/2/3R 不移保本", (1, 2, 3), (1 / 3, 1 / 3, 1 / 3), 2, 0.0),
    ("1/2/3R TP2後才保本", (1, 2, 3), (1 / 3, 1 / 3, 1 / 3), 1, 0.0),
    ("單一 TP 1.5R", (1.5,), (1.0,), 2, 0.0),
    ("單一 TP 2R", (2,), (1.0,), 2, 0.0),
    ("單一 TP 3R", (3,), (1.0,), 2, 0.0),
    ("1/2R + 追蹤3ATR", (1, 2), (1 / 3, 1 / 3), 0, 3.0),
    ("1R + 追蹤3ATR", (1,), (0.5,), 0, 3.0),
    ("只追蹤3ATR", (), (), 2, 3.0),
]


@njit(cache=True)
def track(high, low, close, start, d, entry, risk, tps, portions, protect, trail, atr, fee, max_bars):
    """和 tracker.track 同一套規則的快速版。回傳 (R, 碰到幾個 TP, 出場 K 棒索引, 是否結束)。"""
    n = len(high)
    stop = entry - d * risk
    be = entry * (1 + d * 2 * fee)
    remaining = 1.0
    r = 0.0
    hit = 0
    ntp = len(tps)
    extreme = entry
    bars = 0
    for j in range(start, min(n, start + max_bars)):
        bars += 1
        adverse = low[j] if d > 0 else high[j]
        if d * (adverse - stop) <= 0:
            r += remaining * (d * (stop - entry) - fee * (entry + stop)) / risk
            return r, hit, j, True
        favorable = high[j] if d > 0 else low[j]
        while hit < ntp and d * (favorable - tps[hit]) >= 0:
            p = portions[hit]
            r += p * (d * (tps[hit] - entry) - fee * (entry + tps[hit])) / risk
            remaining -= p
            if protect == 0:
                if hit == 0:
                    stop = be
                elif hit == 1:
                    stop = tps[0]
            elif protect == 1 and hit == 1:
                stop = be
            hit += 1
        if remaining <= 1e-9:
            return r, hit, j, True
        if trail > 0 and hit == ntp:
            if d * (favorable - extreme) > 0:
                extreme = favorable
            ts = extreme - d * trail * atr
            if d * (ts - stop) > 0:
                stop = ts
        if bars >= max_bars:
            r += remaining * (d * (close[j] - entry) - fee * (entry + close[j])) / risk
            return r, hit, j, True
    return r, hit, n - 1, False


def stop_distance(kind: str, arg: float, d: int, price: float, atr: float, hi: float, lo: float,
                  swing_hi: dict, swing_lo: dict) -> float:
    if kind == "atr":
        dist = arg * atr
    elif kind == "candle":
        dist = (price - lo if d > 0 else hi - price) + arg * atr
    elif kind == "candle_floor":
        dist = max(price - lo if d > 0 else hi - price, arg * atr)
    else:
        dist = price - swing_lo[int(arg)] if d > 0 else swing_hi[int(arg)] - price
    return max(dist, MIN_STOP_PCT * price)


def load_bars(coin: str, interval: str) -> pd.DataFrame:
    if interval == "4H":
        df = load(coin, "4h")
    else:
        df = load(coin, "1h")
        if len(df):
            df = resample(df, "2h", origin="epoch")
    return df.reset_index(drop=True)


def coin_signals(coin: str, interval: str) -> dict | None:
    df = load_bars(coin, interval)
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
    swings = {}
    for n in (3, 5, 10):
        swings[n] = (p["high"].rolling(n).max().to_numpy(), p["low"].rolling(n).min().to_numpy())
    return {"coin": coin, "time": p["time"].to_numpy(), "high": p["high"].to_numpy(float),
            "low": p["low"].to_numpy(float), "close": p["close"].to_numpy(float), "atr": p["atr"].to_numpy(float),
            "sigs": sigs, "swings": swings}


def run_variant(data: dict, stop: tuple, exit_: tuple, fee: float, idx_list=None, sides=None) -> list[dict]:
    """一個幣、一個出場組合的所有交易。idx_list/sides 給定時（隨機進場）不做同時只持有一張的限制。"""
    _, kind, arg = stop
    _, tp_r, portions, protect, trail = exit_
    h, l, c, a, t = data["high"], data["low"], data["close"], data["atr"], data["time"]
    sigs = data["sigs"] if idx_list is None else list(zip(idx_list, sides))
    busy = -1
    out = []
    port = np.array(portions, dtype=float)
    for i, d in sigs:
        if idx_list is None and i + 1 < busy:
            continue
        if i + 1 >= len(h) or not np.isfinite(a[i]):
            continue
        price = c[i]
        sw_hi = {n: data["swings"][n][0][i] for n in (3, 5, 10)}
        sw_lo = {n: data["swings"][n][1][i] for n in (3, 5, 10)}
        risk = stop_distance(kind, arg, d, price, a[i], h[i], l[i], sw_hi, sw_lo)
        tps = np.array([price + d * k * risk for k in tp_r], dtype=float)
        r, hit, j, done = track(h, l, c, i + 1, d, price, risk, tps, port, protect, trail, a[i], fee, MAX_BARS)
        if not done:
            continue
        busy = j
        out.append({"coin": data["coin"], "time": t[i + 1], "exit_time": t[j], "side": d, "r": r, "tps_hit": hit,
                    "bars": j - i, "stop_pct": risk / price, "stop_atr": risk / a[i]})
    return out


def summarize(tr: pd.DataFrame, risk: float = 0.0025) -> dict:
    if tr.empty:
        return {"trades": 0}
    r = tr["r"]
    eq = (1 + risk * tr.sort_values("exit_time")["r"]).cumprod()
    dd = (eq / eq.cummax() - 1).min()
    years = max((tr["exit_time"].max() - tr["time"].min()).days / 365.25, 0.1)
    cl = tr.assign(day=tr["time"].dt.floor("D")).groupby(["day", "side"])["r"].mean()
    wins, losses = r[r > 0].sum(), -r[r <= 0].sum()
    return {
        "trades": len(r), "win_rate": (r > 0).mean(), "exp_r": r.mean(), "se_r": r.std() / np.sqrt(len(r)),
        "pf": wins / losses if losses else np.inf, "tp1_rate": (tr["tps_hit"] >= 1).mean(),
        "total_pct": (eq.iloc[-1] - 1) * 100, "ann_pct": (eq.iloc[-1] ** (1 / years) - 1) * 100,
        "max_dd_pct": dd * 100, "clusters": len(cl), "cluster_r": cl.mean(), "cluster_se": cl.std() / np.sqrt(len(cl)),
        "median_stop_pct": tr["stop_pct"].median() * 100, "p10_stop_pct": tr["stop_pct"].quantile(0.1) * 100,
        "median_stop_atr": tr["stop_atr"].median(), "avg_bars": tr["bars"].mean(),
        "wide_stop_share": (tr["stop_pct"] > 0.0025 / 3).mean(),  # 100U 帳戶、0.25U 風險時每 1/3 不到 1U
    }


def periods(tr: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {"全期": tr, "2020-23": tr[tr["time"] < SPLIT], "2024+": tr[tr["time"] >= SPLIT]}


def _work(args):
    coin, interval, fee = args
    data = coin_signals(coin, interval)
    if data is None:
        return coin, None
    res = {}
    for st in STOPS:
        for ex in EXITS:
            res[(st[0], ex[0])] = run_variant(data, st, ex, fee)
    # 隨機進場：每個實際訊號在同一個幣隨機挑 5 根 K 棒、同方向，套同樣出場
    rng = np.random.default_rng(zlib.crc32(coin.encode()))
    n = len(data["high"])
    idx = rng.integers(200, n - 2, size=5 * len(data["sigs"])) if data["sigs"] else np.array([], int)
    sides = [d for _, d in data["sigs"] for _ in range(5)]
    rnd = {}
    for st in STOPS:
        for ex in EXITS[:1] + [e for e in EXITS if e[0] in RANDOM_EXITS]:
            rnd[(st[0], ex[0])] = run_variant(data, st, ex, fee, list(idx), sides)
    return coin, (res, rnd)


RANDOM_EXITS = {"1/2/4R 各1/3", "1/3/5R 各1/3", "單一 TP 2R", "1/2R + 追蹤3ATR", "只追蹤3ATR", "2/4/6R 各1/3"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", default="4H", choices=["4H", "2H"])
    ap.add_argument("--fee", type=float, default=0.0005, help="每邊成本；加 0.05%% 滑價就用 0.001")
    ap.add_argument("--coins", nargs="*")
    ap.add_argument("--out", default="output")
    args = ap.parse_args()
    coins = args.coins or [c for c in TOP100 if c.isascii()]
    allres: dict = {}
    allrnd: dict = {}
    trades_out = {}
    with ProcessPoolExecutor(4) as ex:
        for coin, got in ex.map(_work, [(c, args.interval, args.fee) for c in coins]):
            if got is None:
                continue
            res, rnd = got
            for k, v in res.items():
                allres.setdefault(k, []).extend(v)
            for k, v in rnd.items():
                allrnd.setdefault(k, []).extend(v)
            print(coin, file=sys.stderr)
    rows = []
    for (sname, ename), tr in allres.items():
        tr = pd.DataFrame(tr)
        if tr.empty:
            continue
        tr["time"] = pd.to_datetime(tr["time"], utc=True)
        tr["exit_time"] = pd.to_datetime(tr["exit_time"], utc=True)
        if sname in ("ATR1.5", "K棒極值", "K棒極值+0.2ATR") and ename == EXITS[0][0]:
            trades_out[sname] = tr
        for pname, sub in periods(tr).items():
            rows.append({"stop": sname, "exit": ename, "period": pname, **summarize(sub)})
    rrows = []
    for (sname, ename), tr in allrnd.items():
        tr = pd.DataFrame(tr)
        if tr.empty:
            continue
        tr["time"] = pd.to_datetime(tr["time"], utc=True)
        tr["exit_time"] = pd.to_datetime(tr["exit_time"], utc=True)
        for pname, sub in periods(tr).items():
            s = summarize(sub)
            rrows.append({"stop": sname, "exit": ename, "period": pname, "trades": s["trades"],
                          "exp_r": s["exp_r"], "se_r": s["se_r"], "win_rate": s["win_rate"]})
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    tag = f"{args.interval}" + ("" if args.fee == 0.0005 else f"_fee{args.fee}")
    pd.DataFrame(rows).to_csv(out / f"exit_grid_{tag}.csv", index=False)
    pd.DataFrame(rrows).to_csv(out / f"exit_random_{tag}.csv", index=False)
    for k, tr in trades_out.items():
        tr.to_csv(out / f"exit_trades_{tag}_{k}.csv.gz", index=False)
    print(json.dumps({"variants": len(allres), "rows": len(rows)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
