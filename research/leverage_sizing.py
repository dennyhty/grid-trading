"""開槓桿要開多少、要放多少 U、逐倉還是全倉（只做分析，不改實盤規則）。

python -m research.leverage_sizing      （幣安 2020 起）→ output/leverage_sizing_*.csv

實盤規則：4H、2H 都用 1/3/5R 出場，同一個幣同方向只做一張，止損 1.5 ATR = 1R。
每筆風險固定 X% 時，倉位面額 = 風險 ÷ 止損距離%，跟槓桿無關；槓桿只決定要押多少保證金。
1. 每筆止損距離% 的分布 → 逐倉每筆最多可以開幾倍，強平價才會在止損外面夠遠。
2. 每一刻同時持有的單數、總面額（帳戶的幾倍）、全部同時打到止損的總虧損 → 1 倍槓桿夠不夠用、要幾倍。
3. 每筆風險 0.25% / 0.5% / 0.75% / 1% 的回撤、最差一天、最差 30 天，加上同時持倉上限。
4. 實際成交比止損差的單（跳空、追蹤止損）：最差 R。
"""

from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

import research.entry_research as er
from research.exit_unify import STYLES, dedupe, per_interval
from research.stop_research import load_frame
from grid_trading.signals.strategy import LONG
from grid_trading.signals.vegas_trend import VegasTrend
from grid_trading.universe import TOP100

FEE = 0.0005


def _work(args):
    coin, interval = args
    df = load_frame(coin, interval, "binance")
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
    er.PLANS = {interval: STYLES["2H式"]}
    s = er.simulate(arr, idx, ds, interval, FEE, 0.0)
    return pd.DataFrame({"coin": coin, "interval": interval, "i": idx, "side": ds,
                         "time": t.iloc[idx + 1].to_numpy(), "exit_t": t.iloc[np.minimum(s["j"], len(t) - 1)].to_numpy(),
                         "r": s["r"], "hit": s["hit"], "kind": s["kind"], "j": s["j"], "stop_pct": s["risk_pct"]})


def open_book(tr: pd.DataFrame) -> pd.DataFrame:
    """每次有單進出場時，記下同時持有幾張、總面額（以每筆 1% 風險、帳戶 = 1 計）。"""
    ev = pd.concat([pd.DataFrame({"t": tr["time"], "n": 1, "notional": 0.01 / tr["stop_pct"]}),
                    pd.DataFrame({"t": tr["exit_t"], "n": -1, "notional": -0.01 / tr["stop_pct"]})])
    ev = ev.sort_values(["t", "n"]).reset_index(drop=True)  # 同一刻先出場再進場
    ev["open"] = ev["n"].cumsum()
    ev["notional_x"] = ev["notional"].cumsum()
    return ev


def cap_open(tr: pd.DataFrame, k: int) -> pd.DataFrame:
    tr = tr.sort_values("time").reset_index(drop=True)
    keep = np.zeros(len(tr), bool)
    exits: list = []
    for n, (t, e) in enumerate(zip(tr["time"], tr["exit_t"])):
        exits = [x for x in exits if x > t]
        if len(exits) < k:
            keep[n] = True
            exits.append(e)
    return tr[keep]


def risk_stats(tr: pd.DataFrame, risk: float) -> dict:
    order = tr.sort_values("exit_t")
    eq = np.cumprod(1 + risk * order["r"].to_numpy())
    peak = np.maximum.accumulate(eq)
    dd = (eq / peak - 1).min()
    day = (risk * tr.set_index(tr["exit_t"].dt.floor("D"))["r"]).groupby(level=0).sum()
    day = day.reindex(pd.date_range(day.index.min(), day.index.max(), freq="D"), fill_value=0)
    years = (tr["exit_t"].max() - tr["time"].min()).days / 365.25
    return {"risk_pct": risk * 100, "trades": len(tr), "exp_r": tr["r"].mean(),
            "simple_per_year_pct": risk * tr["r"].sum() / years * 100,
            "cagr_pct": (eq[-1] ** (1 / years) - 1) * 100, "max_dd_pct": dd * 100,
            "worst_day_pct": day.min() * 100, "worst_30d_pct": day.rolling(30).sum().min() * 100,
            "sharpe": day.mean() / day.std() * np.sqrt(365)}


def main():
    out = Path("output")
    out.mkdir(exist_ok=True)
    cache = out / "leverage_sizing_trades.csv.gz"
    if cache.exists():
        allt = pd.read_csv(cache)
    else:
        coins = [c for c in TOP100 if c.isascii()]
        jobs = [(c, iv) for iv in ("4H", "2H") for c in coins]
        with ProcessPoolExecutor(3) as ex:
            allt = pd.concat([g for g in ex.map(_work, jobs) if g is not None], ignore_index=True)
        allt.to_csv(cache, index=False)
    for c in ("time", "exit_t"):
        allt[c] = pd.to_datetime(allt[c], utc=True)
    book = pd.concat([per_interval(allt[allt["interval"] == iv]) for iv in ("4H", "2H")], ignore_index=True)
    live = dedupe(book).reset_index(drop=True)

    # 1. 止損距離
    q = [0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
    rows = []
    for name, g in [("全部", live), ("4H", live[live["interval"] == "4H"]), ("2H", live[live["interval"] == "2H"]),
                    ("2024+", live[live["time"] >= "2024-01-01"])]:
        sp = g["stop_pct"] * 100
        rows.append({"group": name, "n": len(g), **{f"p{int(x*100)}": sp.quantile(x) for x in q}, "max": sp.max()})
    stops = pd.DataFrame(rows)
    stops.to_csv(out / "leverage_sizing_stops.csv", index=False)

    # 2. 同時持倉、總面額
    ev = open_book(live)
    ev = ev[ev["n"] == 1]  # 只看進場那一刻（最高點一定出現在進場時）
    rows = []
    for name, g in [("全部", ev), ("2024+", ev[ev["t"] >= "2024-01-01"]), ("2026", ev[ev["t"] >= "2026-01-01"])]:
        rows.append({"period": name, "open_p50": g["open"].median(), "open_p95": g["open"].quantile(.95),
                     "open_p99": g["open"].quantile(.99), "open_max": g["open"].max(),
                     "notional_x_at_1pct_p50": g["notional_x"].median(), "notional_x_at_1pct_p95": g["notional_x"].quantile(.95),
                     "notional_x_at_1pct_p99": g["notional_x"].quantile(.99), "notional_x_at_1pct_max": g["notional_x"].max(),
                     "max_open_time": g.loc[g["open"].idxmax(), "t"]})
    conc = pd.DataFrame(rows)
    conc.to_csv(out / "leverage_sizing_concurrency.csv", index=False)

    # 3. 風險比例 × 同時持倉上限
    rows = []
    for cap in (None, 30, 20, 15, 10):
        tr = live if cap is None else cap_open(live, cap)
        for risk in (0.0025, 0.005, 0.0075, 0.01):
            for pn, g in [("全期", tr), ("2024+", tr[tr["time"] >= "2024-01-01"])]:
                rows.append({"cap": cap or "不限", "period": pn, **risk_stats(g, risk)})
    grid = pd.DataFrame(rows)
    grid.to_csv(out / "leverage_sizing_risk_grid.csv", index=False)

    # 4. 比止損差的成交
    lo = live["r"]
    tail = pd.DataFrame([{"worst_r": lo.min(), "p001_r": lo.quantile(.001), "n_below_-1.1R": int((lo < -1.1).sum()),
                          "n_below_-1.5R": int((lo < -1.5).sum()), "n": len(lo)}])
    tail.to_csv(out / "leverage_sizing_tail.csv", index=False)
    pd.set_option("display.width", 250)
    for df in (stops, conc, grid, tail):
        print(df.round(3).to_string(), file=sys.stderr)


if __name__ == "__main__":
    main()
