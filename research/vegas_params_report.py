"""research.vegas_params 的比較表（python -m research.vegas_params report）。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from research.experiments import log_experiment, required_t, trials
from research.validation_gate import GateConfig, cluster_t_values, evaluate
from research.vegas_params import OUT, VARIANTS, books

RISK = 0.003
SPLIT = pd.Timestamp("2024-01-01", tz="UTC")
POOLS = {"in_pit": "時點幣池", "liquid": "時點幣池＋日均成交額≥5千萬U"}


def daily(tr: pd.DataFrame, days: pd.DatetimeIndex) -> pd.Series:
    d = pd.Series(tr["r_net"].to_numpy() * RISK, index=tr["exit_t"].dt.floor("D")).groupby(level=0).sum()
    return d.reindex(days, fill_value=0.0)


def diff_t(a: pd.Series, b: pd.Series) -> float:
    """兩個帳本日報酬差的 t（以週為群，避免同一週的單互相關）。"""
    d = a - b
    wk = d.groupby(d.index.strftime("%G-%V")).sum()
    return d.mean() / (wk.std(ddof=1) * np.sqrt(len(wk)) / len(d)) if len(wk) > 2 else np.nan


def row(name, tr, d, d0, n_trials) -> dict:
    r = tr["r_net"].to_numpy()
    ts = cluster_t_values(tr, "r_net")
    eq = (1 + d).cumprod()
    yrs = (d.index[-1] - d.index[0]).days / 365.25
    out = {"variant": name, "trades": len(r), "win": (r > 0).mean(), "exp_r": r.mean(),
           "t_min": min(ts.values()), "t_req": required_t(n_trials),
           "is_r": r[(tr["time"] < SPLIT).to_numpy()].mean(), "oos_r": r[(tr["time"] >= SPLIT).to_numpy()].mean(),
           "yr_pct": d.sum() / yrs * 100, "sharpe": d.mean() / d.std() * np.sqrt(365),
           "dd_pct": (eq / eq.cummax() - 1).min() * 100, "corr_base": d.corr(d0),
           "diff_t_vs_base": diff_t(d, d0) if name != "base" else np.nan}
    for y, g in tr.groupby(tr["exit_t"].dt.year):
        out[f"r{y}"] = g["r_net"].mean()
    return out


def report() -> None:
    tr = pd.read_csv(OUT / "vp_trades.csv.gz", parse_dates=["time", "exit_t"])
    n_trials = trials(count_configs=True) + 12
    rows, by_iv = [], []
    for pool, pname in POOLS.items():
        bk = books(tr, pool)
        days = pd.date_range(min(b["exit_t"].min() for b in bk.values()).floor("D"),
                             max(b["exit_t"].max() for b in bk.values()).floor("D"), freq="D", tz="UTC")
        d0 = daily(bk["base"], days)
        for v in VARIANTS:
            rows.append({"pool": pname, **row(v, bk[v], daily(bk[v], days), d0, n_trials)})
            for iv in ("4H", "2H"):
                sub, sub0 = bk[v][bk[v]["interval"] == iv], bk["base"][bk["base"]["interval"] == iv]
                by_iv.append({"pool": pname, "interval": iv,
                              **row(v, sub, daily(sub, days), daily(sub0, days), n_trials)})
        if pool == "in_pit":
            cfg = GateConfig(n_trials=n_trials)
            for v in VARIANTS[1:]:
                g = evaluate(bk[v], baseline=bk["base"], cfg=cfg)
                print(v, g["verdict"])
    s, iv = pd.DataFrame(rows), pd.DataFrame(by_iv)
    s.to_csv(OUT / "vp_summary.csv", index=False)
    iv.to_csv(OUT / "vp_by_interval.csv", index=False)
    with pd.option_context("display.width", 300, "display.max_columns", 40):
        print(s.round(3).to_string())
        print(iv.round(3).to_string())


def log(summary: pd.DataFrame, verdicts: dict[str, str]) -> None:
    pit = summary[summary["pool"] == POOLS["in_pit"]].set_index("variant")
    for v in VARIANTS[1:]:
        r = pit.loc[v]
        log_experiment(family="vegas_params", hypothesis=f"維加斯換參數：{v}", params={"variant": v},
                       data_source="binance", period="2020-2026", trades=int(r["trades"]), exp_r=round(r["exp_r"], 3),
                       t_stat=round(r["t_min"], 2), sharpe=round(r["sharpe"], 2), verdict=verdicts[v],
                       report="signals/report_vegas_params_20261120.md", notes="configs=2; 2H+4H")
