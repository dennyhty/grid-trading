"""vegas_trend：4H 和 2H 出場改成一樣好不好、同一個幣 2H/4H 同方向只做一張（只做分析，不改實盤規則）。

python -m research.exit_unify --source binance     （幣安 2020 起，主）
python -m research.exit_unify --source pionex      （派網 2023 起，輔）

需要 numba（和 stop_research.py 共用模擬程式）。

1. 每個實盤訊號（RSI 54-70/30-46、離通道 3 ATR 內、2H 加 ADX < 24）分別用兩種出場模擬：
   - 「4H 式」：1R 平一半、保本、剩下 3 ATR 追蹤
   - 「2H 式」：1/3/5R 各平 1/4、TP1 後保本、TP2 後移到 TP1、最後 1/4 追蹤 3 ATR
   止損都是 1.5 ATR。同一個幣同一個週期同時只持有一張。
2. 合併兩個週期成一個帳戶（每筆 0.25% 風險），比較「同一個幣 2H/4H 同方向已經有單就不做」前後。

輸出 output/exit_unify_{source}.csv（每個組合 × 期間）。
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

import research.entry_research as er
from research.stop_research import busy_filter, cluster_key, cluster_stats, load_frame
from grid_trading.signals.strategy import LONG
from grid_trading.signals.vegas_trend import VegasTrend
from grid_trading.universe import TOP100

STYLES = {"4H式": ((1.0,), (0.5,), 3.0), "2H式": ((1.0, 3.0, 5.0), (0.25, 0.25, 0.25), 3.0)}
LIVE_STYLE = {"4H": "4H式", "2H": "2H式"}
SPLIT = pd.Timestamp("2024-01-01", tz="UTC")


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
    idx, ds = np.array(idx), np.array(ds)
    arr = {k: p[k].to_numpy(float) for k in ("open", "high", "low", "close", "atr")}
    t = pd.to_datetime(p["time"], utc=True)
    rows = []
    for style, plan in STYLES.items():
        er.PLANS = {interval: plan}
        s = er.simulate(arr, idx, ds, interval, fee, 0.0)
        rows.append(pd.DataFrame({"coin": coin, "interval": interval, "style": style, "i": idx, "side": ds,
                                  "time": t.iloc[idx + 1].to_numpy(), "exit_t": t.iloc[np.minimum(s["j"], len(t) - 1)].to_numpy(),
                                  "r": s["r"], "hit": s["hit"], "kind": s["kind"], "j": s["j"]}))
    return pd.concat(rows)


def per_interval(tr: pd.DataFrame) -> pd.DataFrame:
    """同一個週期、同一個幣同時只持有一張。"""
    tr = tr[tr["kind"] != -1].sort_values(["coin", "i"]).reset_index(drop=True)
    take = busy_filter(pd.factorize(tr["coin"])[0], tr["i"].to_numpy(np.int64), tr["j"].to_numpy(np.int64),
                       tr["kind"].to_numpy(np.int64))
    return tr[take]


def dedupe(tr: pd.DataFrame) -> pd.DataFrame:
    """兩個週期合併：同一個幣、同方向已經有另一個週期的單還沒出場，就不做這張（先出的那張留下）。"""
    tr = tr.sort_values(["time", "interval"]).reset_index(drop=True)
    keep = np.ones(len(tr), bool)
    open_until: dict = {}
    for k, (c, s, t, e) in enumerate(zip(tr["coin"], tr["side"], tr["time"], tr["exit_t"])):
        if (c, s) in open_until and open_until[(c, s)] > t:
            keep[k] = False
            continue
        open_until[(c, s)] = e
    return tr[keep]


def account(tr: pd.DataFrame, start, end, risk=0.0025) -> dict:
    tr = tr[(tr["time"] >= start) & (tr["time"] < end)]
    if tr.empty:
        return {"trades": 0}
    order = tr.sort_values("exit_t")
    eq = np.cumprod(1 + risk * order["r"].to_numpy())
    dd = (eq / np.maximum.accumulate(eq) - 1).min()
    day = (risk * tr.set_index(tr["exit_t"].dt.floor("D"))["r"]).groupby(level=0).sum()
    day = day.reindex(pd.date_range(tr["time"].min().floor("D"), day.index.max(), freq="D"), fill_value=0)
    worst_day = day.min() / risk
    mean, se, *_ = cluster_stats(tr["r"].to_numpy(), cluster_key(tr))
    years = max((tr["exit_t"].max() - tr["time"].min()).days / 365.25, 0.1)
    return {"trades": len(tr), "win_rate": (tr["r"] > 0).mean(), "exp_r": mean, "se_cluster": se,
            "sum_r": tr["r"].sum(), "sum_r_per_year": tr["r"].sum() / years, "sharpe": day.mean() / day.std() * np.sqrt(365),
            "max_dd_pct": dd * 100, "worst_day_r": worst_day, "days_below_-3r": int((day / risk <= -3).sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="binance", choices=["binance", "pionex"])
    ap.add_argument("--fee", type=float, default=0.0005)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--out", default="output")
    args = ap.parse_args()
    coins = [c for c in TOP100 if c.isascii()]
    jobs = [(c, iv, args.source, args.fee) for iv in ("4H", "2H") for c in coins]
    parts = []
    with ProcessPoolExecutor(args.workers) as ex:
        for got in ex.map(_work, jobs):
            if got is not None:
                parts.append(got)
    allt = pd.concat(parts, ignore_index=True)
    for c in ("time", "exit_t"):
        allt[c] = pd.to_datetime(allt[c], utc=True)
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    allt.to_csv(out / f"exit_unify_trades_{args.source}.csv.gz", index=False)
    t0, t1 = allt["time"].min(), allt["time"].max() + pd.Timedelta(days=1)
    pers = {"全期": (t0, t1), "2020-23": (t0, SPLIT), "2024+": (SPLIT, t1)} if args.source == "binance" else \
        {"全期": (t0, t1), "2023-24": (t0, pd.Timestamp("2025-01-01", tz="UTC")), "2025+": (pd.Timestamp("2025-01-01", tz="UTC"), t1)}
    books = {}
    for iv in ("4H", "2H"):
        for style in STYLES:
            books[(iv, style)] = per_interval(allt[(allt["interval"] == iv) & (allt["style"] == style)])
    combos = {  # 名稱 → (4H 用哪種出場, 2H 用哪種出場)
        "現在（4H 用 4H式、2H 用 2H式）": ("4H式", "2H式"),
        "都用 4H式": ("4H式", "4H式"),
        "都用 2H式": ("2H式", "2H式"),
    }
    rows = []
    for iv in ("4H", "2H"):
        for style in STYLES:
            for pn, (a, b) in pers.items():
                rows.append({"group": f"只看 {iv}", "setup": style, "dedupe": "—", "period": pn,
                             **account(books[(iv, style)], a, b)})
    for name, (s4, s2) in combos.items():
        both = pd.concat([books[("4H", s4)], books[("2H", s2)]], ignore_index=True)
        for dd_name, tr in (("兩張都做", both), ("同幣同向只做一張", dedupe(both))):
            for pn, (a, b) in pers.items():
                rows.append({"group": "4H+2H 帳戶", "setup": name, "dedupe": dd_name, "period": pn, **account(tr, a, b)})
    pd.DataFrame(rows).to_csv(out / f"exit_unify_{args.source}.csv", index=False)
    print(pd.DataFrame(rows).round(3).to_string(), file=sys.stderr)


if __name__ == "__main__":
    main()
