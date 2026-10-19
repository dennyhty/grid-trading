"""2026 年 3~5 月大回撤拆解，和幾種「風險面」改法的回測（只做分析，不改實盤規則）。

python -m research.exit_unify --source binance_ext     # 先產生交易明細
python -m research.drawdown_study                      → output/drawdown_rules.csv

改法都只看「進場那一刻以前已經出場的單」，沒有偷看未來：
- 同時持倉上限（總數 / 每個方向）
- 熔斷：當天（UTC）已經有 K 張單打到止損，當天剩下和隔天不再開新單（全部 / 只停同方向）
- 回撤減碼：帳戶從高點回撤超過 X R，之後的單用一半風險，回到新高才恢復
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from research.stop_research import cluster_key, cluster_stats
from research.yearly_returns import books

RISK = 0.0025
SPLIT = pd.Timestamp("2024-01-01", tz="UTC")
EP = (pd.Timestamp("2026-03-01", tz="UTC"), pd.Timestamp("2026-05-10", tz="UTC"))


def simulate(tr: pd.DataFrame, max_open=None, max_side=None, halt_k=None, halt_side=False, halt_days=1,
             dd_brake=None, brake_mult=0.5) -> pd.DataFrame:
    tr = tr.sort_values(["time", "interval"]).reset_index(drop=True)
    ns = lambda c: (tr[c] - pd.Timestamp(0, tz="UTC")) // pd.Timedelta(1)
    times, exits = ns("time").to_numpy(), ns("exit_t").to_numpy()
    sides, rs = tr["side"].to_numpy(), tr["r"].to_numpy()
    DAY = 86_400 * 10**9
    mult = np.zeros(len(tr))
    open_ = []  # (exit_t, side, idx)
    realized, peak = 0.0, 0.0
    stops_by_day: dict = {}
    halt_until = {1: 0, -1: 0}
    for k in range(len(tr)):
        t = times[k]
        still = []
        for e, s, i in sorted(open_, key=lambda x: x[0]):
            if e <= t:
                realized += rs[i] * mult[i]
                peak = max(peak, realized)
                if rs[i] <= -0.9:
                    day = e // DAY
                    key = (day, s if halt_side else 0)
                    stops_by_day[key] = stops_by_day.get(key, 0) + 1
                    if halt_k and stops_by_day[key] >= halt_k:
                        until = (day + halt_days + 1) * DAY
                        for side in ((s,) if halt_side else (1, -1)):
                            halt_until[side] = max(halt_until[side], until)
            else:
                still.append((e, s, i))
        open_ = still
        s = sides[k]
        if halt_k and t < halt_until[s]:
            continue
        if max_open and len(open_) >= max_open:
            continue
        if max_side and sum(1 for o in open_ if o[1] == s) >= max_side:
            continue
        mult[k] = brake_mult if (dd_brake and peak - realized >= dd_brake) else 1.0
        open_.append((exits[k], s, k))
    out = tr.assign(mult=mult)
    return out[out["mult"] > 0]


def metrics(tr: pd.DataFrame, a=None, b=None) -> dict:
    if a is not None:
        tr = tr[(tr["time"] >= a) & (tr["time"] < b)]
    o = tr.sort_values("exit_t")
    rr = (o["r"] * o["mult"]).to_numpy()
    eq = np.cumprod(1 + RISK * rr)
    dd = (eq / np.maximum.accumulate(np.concatenate([[1.0], eq]))[1:] - 1).min()
    day = (RISK * pd.Series(rr, index=o["exit_t"].dt.floor("D").to_numpy())).groupby(level=0).sum()
    day = day.reindex(pd.date_range(o["time"].min().floor("D"), day.index.max(), freq="D"), fill_value=0)
    years = max((o["exit_t"].max() - o["time"].min()).days / 365.25, 0.1)
    mean, se, *_ = cluster_stats(o["r"].to_numpy(), cluster_key(o))
    return {"trades": len(o), "exp_r": mean, "se": se, "simple_per_year_pct": rr.sum() * RISK * 100 / years,
            "sharpe": day.mean() / day.std() * np.sqrt(365), "max_dd_pct": dd * 100}


def episode_dd(tr: pd.DataFrame) -> float:
    o = tr.sort_values("exit_t")
    rr = (o["r"] * o["mult"]).to_numpy()
    eq = np.cumprod(1 + RISK * rr)
    m = (o["exit_t"] >= EP[0]) & (o["exit_t"] < EP[1])
    pk = np.maximum.accumulate(eq)
    return float(((eq / pk - 1)[m.to_numpy()]).min() * 100)


RULES = {
    "現在（不限制）": {},
    "同時最多 20 張": {"max_open": 20},
    "同時最多 15 張": {"max_open": 15},
    "同時最多 10 張": {"max_open": 10},
    "同方向最多 10 張": {"max_side": 10},
    "同方向最多 8 張": {"max_side": 8},
    "同方向最多 5 張": {"max_side": 5},
    "當天 5 張止損 → 停到隔天（全部）": {"halt_k": 5},
    "當天 8 張止損 → 停到隔天（全部）": {"halt_k": 8},
    "當天 5 張止損 → 停到隔天（同方向）": {"halt_k": 5, "halt_side": True},
    "當天 8 張止損 → 停到隔天（同方向）": {"halt_k": 8, "halt_side": True},
    "回撤 20R 後減半風險": {"dd_brake": 20},
    "回撤 40R 後減半風險": {"dd_brake": 40},
    "同方向最多 8 張 + 當天 5 張止損停同方向": {"max_side": 8, "halt_k": 5, "halt_side": True},
    "回撤 10R 後減半風險": {"dd_brake": 10},
    "回撤 30R 後減半風險": {"dd_brake": 30},
    "同時最多 15 張 + 回撤 20R 後減半": {"max_open": 15, "dd_brake": 20},
}


def breakdown(tr: pd.DataFrame) -> None:
    o = tr.sort_values("exit_t").assign(cum=lambda d: d["r"].cumsum())
    dd = o["cum"] - o["cum"].cummax()
    i = dd.idxmin()
    j = o.loc[:i, "cum"].idxmax()
    w = o[(o["exit_t"] > o.loc[j, "exit_t"]) & (o["exit_t"] <= o.loc[i, "exit_t"])]
    print(f"回撤：{o.loc[j, 'exit_t']} → {o.loc[i, 'exit_t']}，{len(w)} 筆，{w['r'].sum():.1f}R，勝率 {(w['r'] > 0).mean():.1%}")
    print(w.groupby("side")["r"].agg(["count", "sum", "mean"]).round(3))
    print(w.groupby("interval")["r"].agg(["count", "sum", "mean"]).round(3))
    s = w[w["r"] <= -0.9]
    print("全額止損", len(s), "筆；止損最多的日子：")
    print(s.groupby(s["exit_t"].dt.floor("D")).size().sort_values(ascending=False).head(6))
    y = o.assign(y=o["exit_t"].dt.year)
    print(y[y["y"] >= 2024].groupby(["y", "side"])["r"].agg(["count", "mean"]).round(3))


def main() -> None:
    live = books("binance_ext")["4H+2H（實盤）"]
    breakdown(live)
    t0, t1 = live["time"].min(), live["exit_t"].max() + pd.Timedelta(days=1)
    rows = []
    for name, kw in RULES.items():
        sim = simulate(live, **kw)
        row = {"rule": name, **metrics(sim), "dd_2026_mar_may_pct": episode_dd(sim)}
        for pn, (a, b) in {"2017-23": (t0, SPLIT), "2024+": (SPLIT, t1)}.items():
            m = metrics(sim, a, b)
            row[f"{pn}_per_year_pct"] = m["simple_per_year_pct"]
            row[f"{pn}_sharpe"] = m["sharpe"]
            row[f"{pn}_max_dd_pct"] = m["max_dd_pct"]
        rows.append(row)
    # 對照：全部單一律用 0.8 倍風險（單純下小一點）
    sim = simulate(live).assign(mult=0.8)
    rows.append({"rule": "對照：每筆風險 ×0.8", **metrics(sim), "dd_2026_mar_may_pct": episode_dd(sim)})
    out = pd.DataFrame(rows)
    out.to_csv("output/drawdown_rules.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(out.round(2).to_string())


if __name__ == "__main__":
    main()
