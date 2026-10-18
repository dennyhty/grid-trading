"""實盤維加斯（4H、2H、4H+2H 合併）每年報酬和歷史累積報酬曲線（只做分析，不改實盤規則）。

python -m research.exit_unify --source binance      # 先產生 output/exit_unify_trades_binance.csv.gz
python -m research.yearly_returns                   → output/yearly_returns.csv、output/equity_curve.csv

實盤規則（2026-10-15 起）：4H 和 2H 都用 1/3/5R 各平 1/4 的出場，同一個幣同方向只做一張（2H 同時出現時優先）。
每筆風險 0.25%，損益算在出場那天、那一年。
- 單利：每筆都用起始資金的 0.25% 下單，報酬 = 總 R × 0.25%（網站上的「簡單加總」，比較接近實際）。
- 複利：每筆用當下資金的 0.25% 下單。每年的複利報酬從年初資金重新算。
- 最大回撤：用複利資金曲線，年度回撤只看當年內的高點。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from grid_trading.binance_history import load
from research.exit_unify import dedupe, per_interval

RISK = 0.0025


def books() -> dict[str, pd.DataFrame]:
    allt = pd.read_csv("output/exit_unify_trades_binance.csv.gz")
    for c in ("time", "exit_t"):
        allt[c] = pd.to_datetime(allt[c], utc=True)
    live = {iv: per_interval(allt[(allt["interval"] == iv) & (allt["style"] == "2H式")]) for iv in ("4H", "2H")}
    return {"4H": live["4H"], "2H": live["2H"], "4H+2H（實盤）": dedupe(pd.concat(live.values(), ignore_index=True))}


def max_dd(r: np.ndarray) -> float:
    eq = np.concatenate([[1.0], np.cumprod(1 + RISK * r)])
    return (eq / np.maximum.accumulate(eq) - 1).min()


def yearly(tr: pd.DataFrame) -> pd.DataFrame:
    tr = tr.sort_values("exit_t")
    rows = []
    for y, g in tr.groupby(tr["exit_t"].dt.year):
        r = g["r"].to_numpy()
        rows.append({"year": y, "trades": len(g), "win_rate": (r > 0).mean(), "exp_r": r.mean(), "sum_r": r.sum(),
                     "simple_pct": r.sum() * RISK * 100, "compound_pct": (np.prod(1 + RISK * r) - 1) * 100,
                     "max_dd_pct": max_dd(r) * 100,
                     "first_exit": g["exit_t"].min().date(), "last_exit": g["exit_t"].max().date()})
    r = tr["r"].to_numpy()
    years = (tr["exit_t"].max() - tr["time"].min()).days / 365.25
    comp = np.prod(1 + RISK * r)
    rows.append({"year": "全期", "trades": len(tr), "win_rate": (r > 0).mean(), "exp_r": r.mean(), "sum_r": r.sum(),
                 "simple_pct": r.sum() * RISK * 100, "compound_pct": (comp - 1) * 100, "max_dd_pct": max_dd(r) * 100,
                 "simple_per_year_pct": r.sum() * RISK * 100 / years, "cagr_pct": (comp ** (1 / years) - 1) * 100,
                 "years": years, "first_exit": tr["time"].min().date(), "last_exit": tr["exit_t"].max().date()})
    return pd.DataFrame(rows)


def curve(tr: pd.DataFrame, days: pd.DatetimeIndex) -> pd.DataFrame:
    tr = tr.sort_values("exit_t")
    d = tr.set_index(tr["exit_t"].dt.floor("D"))["r"]
    simple = (d.groupby(level=0).sum() * RISK).reindex(days, fill_value=0).cumsum()
    comp = np.log1p(RISK * d).groupby(level=0).sum().reindex(days, fill_value=0).cumsum()
    return pd.DataFrame({"simple_pct": simple * 100, "compound_pct": np.expm1(comp) * 100}, index=days)


def btc_yearly(start, end) -> pd.Series:
    c = load("BTC", "4h").set_index("time")["close"]
    c = c[(c.index >= start) & (c.index < end + pd.Timedelta(days=1))]
    last = c.groupby(c.index.year).last()
    first = c.iloc[0]
    prev = last.shift(1).fillna(first)
    return (last / prev - 1) * 100


def main() -> None:
    bk = books()
    t0 = min(b["time"].min() for b in bk.values()).floor("D")
    t1 = max(b["exit_t"].max() for b in bk.values()).floor("D")
    days = pd.date_range(t0, t1, freq="D")
    ys, cs = [], []
    for name, tr in bk.items():
        y = yearly(tr)
        y.insert(0, "book", name)
        ys.append(y)
        c = curve(tr, days)
        c.columns = [f"{name}|{k}" for k in c.columns]
        cs.append(c)
    out = pd.concat(ys, ignore_index=True)
    btc = btc_yearly(t0, t1)
    out["btc_pct"] = out["year"].map(lambda y: btc.get(y, np.nan) if y != "全期" else np.nan)
    out.to_csv("output/yearly_returns.csv", index=False)
    cv = pd.concat(cs, axis=1)
    cv.index.name = "date"
    cv.to_csv("output/equity_curve.csv")
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(out.round(3).to_string())


if __name__ == "__main__":
    main()
