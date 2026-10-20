"""實盤維加斯的「誠實版」回測：時點幣池（沒有倖存者偏差）＋ 滑價與資金費率（只做分析，不改實盤規則）。

python -m grid_trading.pit_universe        # 先建每月成交額表（data/binance/pit_monthly_volume.csv）
python -m research.pit_backtest            → output/pit_trades.csv.gz、output/pit_yearly.csv

舊的回測用的是「今天」的市值前 100 名（grid_trading/universe.py，2026-10-12 的快照），早年被美化：
當年在榜上、後來下市或跌出榜的幣沒算到。這裡改成每個月只交易「上個月成交額前 100 名」的幣安 USDT 永續合約
（含後來下市的 LUNA、FTT 等），比較兩種幣池，以及扣掉滑價和資金費率前後（grid_trading/costs.py）。

規則和實盤一樣（2026-10-20 起）：4H、2H 都用 1/3/5R 各平 1/4、TP1 後保本、TP2 後移到 TP1、最後 1/4 追蹤 3 ATR，
止損 1.5 ATR；同一週期同一個幣同時只一張，同幣同向 2H、4H 兩張都做。每筆 0.25% 風險，每邊手續費 0.05%。
只用合約 K 線（2020 年起）；2H 直接用幣安的 2h K 線（和 1h 從 epoch 起算重取樣一樣）。
訊號在幣池裡才進場；進場後掉出幣池的單照常出場。
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

import research.entry_research as er
from grid_trading.binance_history import binance_symbols
from grid_trading.costs import apply_costs
from grid_trading.pit_universe import coin_of, fetch_symbol, membership
from grid_trading.signals.strategy import LONG
from grid_trading.signals.vegas_trend import VegasTrend
from grid_trading.universe import TOP100
from research.exit_unify import per_interval
from research.stop_research import cluster_key, cluster_stats

RISK = 0.0025
PLAN = ((1.0, 3.0, 5.0), (0.25, 0.25, 0.25), 3.0)
START = pd.Timestamp("2020-01-01", tz="UTC")
KLINE = {"4H": "4h", "2H": "2h"}


def _work(args):
    sym, interval, fee = args
    df = fetch_symbol(sym, KLINE[interval])
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
    er.PLANS = {interval: PLAN}
    s = er.simulate(arr, idx, ds, interval, fee, 0.0)
    # 近 30 天平均每日成交額（USDT），給滑價分級用
    bars_per_day = 6 if interval == "4H" else 12
    adv = (p["volume"] * p["close"]).rolling(30 * bars_per_day, min_periods=bars_per_day).sum() / 30
    return pd.DataFrame({"symbol": sym, "coin": coin_of(sym), "interval": interval, "i": idx, "side": ds,
                         "time": t.iloc[idx + 1].to_numpy(), "exit_t": t.iloc[np.minimum(s["j"], len(t) - 1)].to_numpy(),
                         "r": s["r"], "hit": s["hit"], "kind": s["kind"], "j": s["j"], "risk_pct": s["risk_pct"],
                         "adv_usdt": adv.iloc[idx].to_numpy()})


def snapshot_symbols() -> set[str]:
    """舊回測的幣池：今天的前 100 名快照對應到的幣安合約代號。"""
    return {s for c in TOP100 if c.isascii() for s in binance_symbols(c)}


def run(fee: float, workers: int, n: int) -> pd.DataFrame:
    mem = membership(n=n)
    mem["month"] = mem["month"].astype(str)
    snap = snapshot_symbols()
    syms = sorted(set(mem.loc[mem["month"] >= "2020-01", "symbol"]) | snap)
    print(f"{len(syms)} 個合約（時點幣池 ∪ 快照）", file=sys.stderr)
    jobs = [(s, iv, fee) for iv in ("4H", "2H") for s in syms]
    parts = []
    with ProcessPoolExecutor(workers) as ex:
        for k, got in enumerate(ex.map(_work, jobs, chunksize=4)):
            if got is not None:
                parts.append(got)
            if k % 100 == 0:
                print(f"  {k}/{len(jobs)}", file=sys.stderr)
    tr = pd.concat(parts, ignore_index=True)
    for c in ("time", "exit_t"):
        tr[c] = pd.to_datetime(tr[c], utc=True)
    tr = tr[tr["time"] >= START]
    pit = set(zip(mem["symbol"], mem["month"]))
    tr["in_pit"] = [(s, m) in pit for s, m in zip(tr["symbol"], tr["time"].dt.strftime("%Y-%m"))]
    tr["in_snapshot"] = tr["symbol"].isin(snap)
    return apply_costs(tr)


def book(tr: pd.DataFrame, col: str) -> pd.DataFrame:
    """某個幣池的實盤帳本：先篩幣池，再每個週期同一個合約同時只一張，2H、4H 合併。"""
    sub = tr[tr[col]]
    parts = [per_interval(sub[sub["interval"] == iv].rename(columns={"coin": "_c", "symbol": "coin"}))
             .rename(columns={"coin": "symbol", "_c": "coin"}) for iv in ("4H", "2H")]
    return pd.concat(parts, ignore_index=True).sort_values("exit_t").reset_index(drop=True)


def summarize(tr: pd.DataFrame, rcol: str) -> dict:
    r = tr[rcol].to_numpy()
    if not len(r):
        return {"trades": 0}
    mean, se, *_ = cluster_stats(r, cluster_key(tr))
    eq = np.concatenate([[1.0], np.cumprod(1 + RISK * r)])
    return {"trades": len(r), "coins": tr["symbol"].nunique(), "win_rate": (r > 0).mean(), "exp_r": mean, "se_cluster": se,
            "t": mean / se if se else np.nan, "simple_pct": r.sum() * RISK * 100,
            "max_dd_pct": (eq / np.maximum.accumulate(eq) - 1).min() * 100,
            "long_r": r[tr["side"].to_numpy() > 0].mean(), "short_r": r[tr["side"].to_numpy() < 0].mean()}


def yearly(books: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for name, tr in books.items():
        for rcol, cost in (("r", "未扣"), ("r_net", "扣滑價+資金費率")):
            for y, g in tr.groupby(tr["exit_t"].dt.year):
                rows.append({"universe": name, "costs": cost, "year": y, **summarize(g, rcol)})
            years = (tr["exit_t"].max() - tr["time"].min()).days / 365.25
            s = summarize(tr, rcol)
            rows.append({"universe": name, "costs": cost, "year": "全期", **s,
                         "simple_per_year_pct": s["simple_pct"] / years})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fee", type=float, default=0.0005)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--n", type=int, default=100, help="每月取成交額前幾名")
    ap.add_argument("--out", default="output")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(exist_ok=True)
    tr = run(a.fee, a.workers, a.n)
    tr.to_csv(out / "pit_trades.csv.gz", index=False)
    books = {"快照（今天前 100 名）": book(tr, "in_snapshot"), "時點（當月成交額前 100 名）": book(tr, "in_pit")}
    y = yearly(books)
    y.to_csv(out / "pit_yearly.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.max_rows", 200):
        print(y.round(3).to_string())


if __name__ == "__main__":
    main()
