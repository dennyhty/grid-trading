"""vegas_trend：比較四組止盈階梯 1/2/3R、1/3/5R（現在）、2/4/6R、1/2/4R（只做分析，不改實盤規則）。

python -m research.tp_ladder --source binance_ext   （合約上市前接幣安現貨，2017-08 起，主）
python -m research.tp_ladder --source pionex        （派網 2023 起，輔）

除了三個止盈價位，其他都照實盤：止損 1.5 ATR = 1R，每個 TP 各平 1/4，TP1 後保本、TP2 後止損移到 TP1，
最後 1/4 在 TP3 後追蹤 3 ATR。同幣同週期同時一張、4H+2H 同幣同方向只做一張，每組各自套。
另外用現在（1/3/5R）帳戶做到的那批單算成對差和群集穩健標準誤。

輸出 output/tp_ladder_{source}.csv（帳戶）、tp_ladder_yearly_{source}.csv（逐年）、tp_ladder_paired_{source}.csv（成對差）。
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from grid_trading.signals.strategy import LONG
from grid_trading.signals.vegas_trend import VegasTrend
from grid_trading.universe import TOP100
from research.exit_unify import dedupe, per_interval
from research.stop_research import LIVE_ATR, MAX_BARS, MIN_STOP_PCT, cluster_key, cluster_stats, load_frame, sim_many

PORTIONS, TRAIL_K, PROTECT = (0.25, 0.25, 0.25), 3.0, 0  # protect 0 = 實盤：TP1 保本、TP2 移到 TP1
MODES = {  # 名稱 → 三個止盈（R）；第一個是現在實盤，成對差都跟它比
    "1/3/5R（現在）": (1.0, 3.0, 5.0),
    "1/2/3R": (1.0, 2.0, 3.0),
    "2/4/6R": (2.0, 4.0, 6.0),
    "1/2/4R": (1.0, 2.0, 4.0),
}
RISK = 0.0025


def _work(args):
    coin, interval, source, fee = args
    df = load_frame(coin, interval, source)
    strat = VegasTrend()
    if len(df) < strat.min_bars + 50:
        return None
    p = strat.prepare(df).reset_index(drop=True)
    idx, ds = [], []
    for i in np.flatnonzero(strat.candidates(p)):
        if i < strat.min_bars or i + 1 >= len(p):
            continue
        hit = strat.check(p, int(i))
        if hit is None or not strat.accept(p, int(i), interval):
            continue
        idx.append(int(i))
        ds.append(1 if hit[0] == LONG else -1)
    if not idx:
        return None
    idx, ds = np.array(idx), np.array(ds, np.int64)
    op, hi, lo, cl, atr = (p[k].to_numpy(float) for k in ("open", "high", "low", "close", "atr"))
    price, a = cl[idx], atr[idx]
    risk = np.maximum(LIVE_ATR * a, MIN_STOP_PCT * price)
    stop0 = price - ds * risk
    zero = np.zeros(len(idx))
    isc = np.ones(len(hi), np.bool_)
    t = pd.to_datetime(p["time"], utc=True)
    rows = []
    for name, tp_r in MODES.items():
        tps = np.stack([price + ds * k * risk for k in tp_r], axis=1)
        r, hit, j, kind, _ = sim_many(op, hi, lo, cl, isc, idx + 1, ds, price, risk, stop0, tps, np.array(PORTIONS),
                                      PROTECT, TRAIL_K * a, fee, 0.0, MAX_BARS, 0, 0, zero, 0.0, 0.0, zero, 0, cl, cl, False)
        rows.append(pd.DataFrame({"coin": coin, "interval": interval, "mode": name, "i": idx, "side": ds,
                                  "time": t.iloc[idx + 1].to_numpy(), "exit_t": t.iloc[np.minimum(j, len(t) - 1)].to_numpy(),
                                  "r": r, "hit": hit, "kind": kind, "j": j}))
    return pd.concat(rows)


def max_dd(tr: pd.DataFrame) -> float:
    eq = np.concatenate([[1.0], np.cumprod(1 + RISK * tr.sort_values("exit_t")["r"].to_numpy())])
    return (eq / np.maximum.accumulate(eq) - 1).min() * 100


def stats(tr: pd.DataFrame) -> dict:
    if tr.empty:
        return {"trades": 0}
    r = tr["r"].to_numpy()
    mean, se, *_ = cluster_stats(r, cluster_key(tr))
    day = (RISK * tr.set_index(tr["exit_t"].dt.floor("D"))["r"]).groupby(level=0).sum()
    day = day.reindex(pd.date_range(tr["time"].min().floor("D"), day.index.max(), freq="D"), fill_value=0)
    years = max((tr["exit_t"].max() - tr["time"].min()).days / 365.25, 0.1)
    return {"trades": len(tr), "win_rate": (r > 0).mean(), "exp_r": mean, "se_cluster": se, "t": mean / se,
            "sum_r": r.sum(), "simple_pct_per_year": r.sum() * RISK * 100 / years,
            "sharpe": day.mean() / day.std() * np.sqrt(365), "max_dd_pct": max_dd(tr),
            "tp1_rate": (tr["hit"] >= 1).mean(), "tp2_rate": (tr["hit"] >= 2).mean(), "tp3_rate": (tr["hit"] >= 3).mean(),
            "full_stop_rate": (tr["kind"] == 1).mean(),
            "tp1_then_be_or_worse": ((tr["hit"] == 1) & (tr["kind"] == 2)).mean()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="binance_ext", choices=["binance", "binance_ext", "pionex"])
    ap.add_argument("--fee", type=float, default=0.0005)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="output")
    args = ap.parse_args()
    coins = [c for c in TOP100 if c.isascii()]
    jobs = [(c, iv, args.source, args.fee) for iv in ("4H", "2H") for c in coins]
    with ProcessPoolExecutor(args.workers) as ex:
        parts = [g for g in ex.map(_work, jobs) if g is not None]
    allt = pd.concat(parts, ignore_index=True)
    for c in ("time", "exit_t"):
        allt[c] = pd.to_datetime(allt[c], utc=True)
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    allt.to_csv(out / f"tp_ladder_trades_{args.source}.csv.gz", index=False)

    t0, t1 = allt["time"].min(), allt["time"].max() + pd.Timedelta(days=1)
    ts = lambda s: pd.Timestamp(s, tz="UTC")
    if args.source == "pionex":
        pers = {"全期": (t0, t1), "2023-24": (t0, ts("2025-01-01")), "2025+": (ts("2025-01-01"), t1)}
    else:
        pers = {"全期": (t0, t1), "2017-19": (t0, ts("2020-01-01")), "2020-23": (ts("2020-01-01"), ts("2024-01-01")),
                "2024+": (ts("2024-01-01"), t1)}
    books = {}
    for mode in MODES:
        m = allt[allt["mode"] == mode]
        b4, b2 = per_interval(m[m["interval"] == "4H"]), per_interval(m[m["interval"] == "2H"])
        books[mode] = {"4H": b4, "2H": b2, "4H+2H（實盤）": dedupe(pd.concat([b4, b2], ignore_index=True))}

    rows, yrows = [], []
    for mode, bk in books.items():
        for book, tr in bk.items():
            for pn, (a, b) in pers.items():
                rows.append({"book": book, "mode": mode, "period": pn, **stats(tr[(tr["time"] >= a) & (tr["time"] < b)])})
            for y, g in tr.groupby(tr["exit_t"].dt.year):
                yrows.append({"book": book, "mode": mode, "year": y, "trades": len(g), "win_rate": (g["r"] > 0).mean(),
                              "exp_r": g["r"].mean(), "sum_r": g["r"].sum(), "simple_pct": g["r"].sum() * RISK * 100,
                              "max_dd_pct": max_dd(g)})

    # 成對差：用 A（實盤）帳戶實際做到的那批單，換成別的出場每筆差多少
    base = list(MODES)[0]
    key = ["coin", "interval", "i"]
    prow = []
    for book, tr in books[base].items():
        for mode in list(MODES)[1:]:
            other = allt[allt["mode"] == mode][key + ["r"]].rename(columns={"r": "r2"})
            m = tr.merge(other, on=key)
            for pn, (a, b) in pers.items():
                g = m[(m["time"] >= a) & (m["time"] < b)]
                d = (g["r2"] - g["r"]).to_numpy()
                mean, se, *_ = cluster_stats(d, cluster_key(g))
                prow.append({"book": book, "mode": mode, "vs": base, "period": pn, "trades": len(g), "diff_r": mean,
                             "se_cluster": se, "t": mean / se if se else np.nan,
                             "share_better": (d > 1e-9).mean(), "share_worse": (d < -1e-9).mean()})

    pd.DataFrame(rows).to_csv(out / f"tp_ladder_{args.source}.csv", index=False)
    pd.DataFrame(yrows).to_csv(out / f"tp_ladder_yearly_{args.source}.csv", index=False)
    pd.DataFrame(prow).to_csv(out / f"tp_ladder_paired_{args.source}.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.max_rows", 500):
        print(pd.DataFrame(rows).round(3).to_string(), file=sys.stderr)
        print(pd.DataFrame(prow).round(3).to_string(), file=sys.stderr)


if __name__ == "__main__":
    main()
