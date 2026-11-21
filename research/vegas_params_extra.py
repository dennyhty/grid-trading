"""加訊號（不取代）：其他通道／觸發多出來、現行沒有的那些單，本身賺不賺、加進去帳戶會不會更好。2026-11-21。

python -m research.vegas_params_extra   （先跑 research.vegas_params run）
事先定好 4 組（2H、4H 一起算，記 8 組設定）：288/338 多出來的單、576/676 多出來的、兩者合起來、EMA12 穿越觸發多出來的。
「多出來」= 同一幣、同週期、同一根 K 棒、同方向，現行沒有出訊號。
兩種加法：(a) 獨立倉位（像多一個策略，同幣同週期可以和現行各一張）；(b) 共用倉位（同幣同週期還是只一張，先到先做）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from research.pit_backtest import book
from research.validation_gate import cluster_t_values
from research.vegas_params import OUT
from research.vegas_params_report import POOLS, RISK, SPLIT, daily, diff_t

ADDS = {"288/338 多出來": ["ch288"], "576/676 多出來": ["ch576"], "288 + 576 多出來": ["ch288", "ch576"],
        "EMA12 穿越多出來": ["e12_trig"]}
KEY = ["symbol", "interval", "i", "side"]


def stats(name, tr, d, d0) -> dict:
    r = tr["r_net"].to_numpy()
    eq = (1 + d).cumprod()
    yrs = (d.index[-1] - d.index[0]).days / 365.25
    out = {"set": name, "trades": len(r), "win": (r > 0).mean() if len(r) else np.nan, "exp_r": r.mean(),
           "t_min": min(cluster_t_values(tr, "r_net").values()) if len(r) > 50 else np.nan,
           "is_r": r[(tr["time"] < SPLIT).to_numpy()].mean(), "oos_r": r[(tr["time"] >= SPLIT).to_numpy()].mean(),
           "yr_pct": d.sum() / yrs * 100, "sharpe": d.mean() / d.std() * np.sqrt(365),
           "dd_pct": (eq / eq.cummax() - 1).min() * 100, "corr_base": d.corr(d0), "diff_t": diff_t(d, d0)}
    for y, g in tr.groupby(tr["exit_t"].dt.year):
        out[f"r{y}"] = g["r_net"].mean()
    return out


def main() -> None:
    tr = pd.read_csv(OUT / "vp_trades.csv.gz", parse_dates=["time", "exit_t"])
    base_raw = tr[tr["variant"] == "base"]
    base_keys = set(map(tuple, base_raw[KEY].to_numpy()))
    rows = []
    for pool, pname in POOLS.items():
        b0 = book(base_raw, pool)
        days = pd.date_range(b0["exit_t"].min().floor("D"), b0["exit_t"].max().floor("D"), freq="D", tz="UTC")
        d0 = daily(b0, days)
        rows.append({"pool": pname, "mode": "現行", **stats("現行", b0, d0, d0)})
        for name, vs in ADDS.items():
            ex = tr[tr["variant"].isin(vs)].drop_duplicates(KEY)
            ex = ex[[k not in base_keys for k in map(tuple, ex[KEY].to_numpy())]]
            eb = book(ex, pool)
            rows.append({"pool": pname, "mode": "只看多出來的單", **stats(name, eb, daily(eb, days), d0)})
            a = pd.concat([b0, eb])
            rows.append({"pool": pname, "mode": "(a) 獨立倉位相加", **stats(name, a, daily(a, days), d0)})
            u = book(pd.concat([base_raw, ex]), pool)
            rows.append({"pool": pname, "mode": "(b) 共用倉位", **stats(name, u, daily(u, days), d0)})
    s = pd.DataFrame(rows)
    s.to_csv(OUT / "vp_extra.csv", index=False)
    with pd.option_context("display.width", 300, "display.max_columns", 40):
        print(s.round(3).to_string())


if __name__ == "__main__":
    main()
