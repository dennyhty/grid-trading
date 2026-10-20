"""網站「績效」頁的維加斯實盤規則回測資料（web/perf/vegas_backtest.json）。

python -m research.exit_unify --source binance_ext     # 先產生 output/exit_unify_trades_binance_ext.csv.gz
python -m research.site_performance                    → web/perf/vegas_backtest.json

規則和實盤一樣：4H、2H 都用 1/3/5R 各平 1/4、TP1 後保本、TP2 後移到 TP1、最後 1/4 追蹤 3 ATR；
合併帳戶 2H、4H 各自做（2026-10-20 起同幣同向兩張都做，同一週期同一個幣同時只一張）。每筆 0.25% 風險，每邊手續費 0.05%，沒算滑價和資金費率。
損益記在出場那天。最大回撤只看已平倉（每筆出場後的資金），含未平倉浮動損益的回撤會更深。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.exit_unify import per_interval
from research.stop_research import cluster_key, cluster_stats

RISK = 0.0025
BOOKS = ("4H+2H", "4H", "2H")


def books(source: str) -> dict[str, pd.DataFrame]:
    allt = pd.read_csv(f"output/exit_unify_trades_{source}.csv.gz")
    for c in ("time", "exit_t"):
        allt[c] = pd.to_datetime(allt[c], utc=True)
    live = {iv: per_interval(allt[(allt["interval"] == iv) & (allt["style"] == "2H式")]) for iv in ("4H", "2H")}
    both = pd.concat(live.values(), ignore_index=True)  # 2026-10-20 起同幣同向 2H、4H 兩張都做
    return {k: v.sort_values(["exit_t", "time"]).reset_index(drop=True)
            for k, v in {"4H+2H": both, "4H": live["4H"], "2H": live["2H"]}.items()}


def btc_daily(days: pd.DatetimeIndex) -> pd.Series:
    try:
        from grid_trading.binance_history import load_extended as get
    except ImportError:  # 舊版沒有現貨補長
        from grid_trading.binance_history import load as get
    c = get("BTC", "4h")
    if c.empty:
        return pd.Series(np.nan, index=days)
    return c.set_index("time")["close"].resample("D").last().reindex(days).ffill()


def daily_returns(tr: pd.DataFrame, days: pd.DatetimeIndex) -> pd.Series:
    """每天（出場日）的單利報酬：sum R × 0.25%。"""
    return (tr.groupby(tr["exit_t"].dt.floor("D"))["r"].sum() * RISK).reindex(days, fill_value=0.0)


def streak(r: np.ndarray) -> int:
    best = cur = 0
    for x in r:
        cur = cur + 1 if x <= 0 else 0
        best = max(best, cur)
    return best


def metrics(tr: pd.DataFrame, days: pd.DatetimeIndex) -> dict:
    r = tr["r"].to_numpy()
    if not len(r):
        return {"trades": 0}
    span = days[(days >= tr["time"].min().floor("D")) & (days <= tr["exit_t"].max().floor("D"))]
    years = len(span) / 365.25
    eq = np.cumprod(1 + RISK * r)
    peak = np.maximum.accumulate(np.concatenate([[1.0], eq]))[1:]
    dd = eq / peak - 1
    # 回撤最久：從高點到回到高點（或到資料結束）的天數
    t = tr["exit_t"].to_numpy()
    longest, top = 0, t[0]
    for k in range(len(eq)):
        if eq[k] >= peak[k]:
            top = t[k]
        longest = max(longest, int((t[k] - top) / np.timedelta64(1, "D")))
    d = daily_returns(tr, span)
    sd = d.std()
    down = np.sqrt((np.minimum(d, 0) ** 2).mean())
    wins, losses = r[r > 0], r[r <= 0]
    mean, se, *_ = cluster_stats(r, cluster_key(tr))
    cagr = eq[-1] ** (1 / years) - 1 if years > 0 else np.nan
    mdd = dd.min()
    out = {
        "trades": int(len(r)), "coins": int(tr["coin"].nunique()), "years": round(years, 2),
        "start": str(tr["time"].min().date()), "end": str(tr["exit_t"].max().date()),
        "trades_per_year": len(r) / years, "win_rate": (r > 0).mean(), "avg_r": r.mean(), "median_r": float(np.median(r)),
        "total_r": r.sum(), "avg_win_r": wins.mean() if len(wins) else 0, "avg_loss_r": losses.mean() if len(losses) else 0,
        "profit_factor": wins.sum() / -losses.sum() if losses.sum() < 0 else None,
        "simple_pct": r.sum() * RISK * 100, "simple_per_year_pct": r.sum() * RISK * 100 / years,
        "compound_pct": (eq[-1] - 1) * 100, "cagr_pct": cagr * 100, "max_dd_pct": mdd * 100,
        "max_dd_r": float(np.min(np.cumsum(r) - np.maximum.accumulate(np.concatenate([[0.0], np.cumsum(r)]))[1:])),
        "longest_dd_days": longest, "max_losing_streak": streak(r),
        "sharpe": d.mean() / sd * np.sqrt(365) if sd > 0 else None,
        "sortino": d.mean() / down * np.sqrt(365) if down > 0 else None,
        "calmar": cagr / -mdd if mdd < 0 else None,
        "t_cluster": mean / se if se and se > 0 else None,
    }
    for side, name in ((1, "long"), (-1, "short")):
        x = r[tr["side"].to_numpy() == side]
        out[name] = {"trades": int(len(x)), "win_rate": (x > 0).mean() if len(x) else None,
                     "avg_r": x.mean() if len(x) else None, "total_r": x.sum()}
    return out


def yearly(tr: pd.DataFrame, btc: pd.Series) -> list[dict]:
    rows = []
    for y, g in tr.groupby(tr["exit_t"].dt.year):
        r = g["r"].to_numpy()
        eq = np.concatenate([[1.0], np.cumprod(1 + RISK * r)])
        b = btc[btc.index.year == y].dropna()
        prev = btc[btc.index.year == y - 1].dropna()
        base = prev.iloc[-1] if len(prev) else (b.iloc[0] if len(b) else np.nan)
        rows.append({"year": int(y), "trades": int(len(r)), "win_rate": (r > 0).mean(), "avg_r": r.mean(), "total_r": r.sum(),
                     "simple_pct": r.sum() * RISK * 100, "compound_pct": (eq[-1] - 1) * 100,
                     "max_dd_pct": (eq / np.maximum.accumulate(eq) - 1).min() * 100,
                     "btc_pct": (b.iloc[-1] / base - 1) * 100 if len(b) else None})
    return rows


def monthly(tr: pd.DataFrame) -> list[list]:
    m = tr.groupby(tr["exit_t"].dt.strftime("%Y-%m"))["r"].agg(["sum", "count"])
    return [[k, round(v["sum"] * RISK * 100, 2), int(v["count"])] for k, v in m.iterrows()]


def rolling(tr: pd.DataFrame, n: int = 500, every: int = 10) -> list[list]:
    """最近 n 筆的平均 R（照出場順序），每 every 筆取一點。"""
    roll = tr["r"].rolling(n).mean()
    idx = range(n - 1, len(tr), every)
    return [[str(tr["exit_t"].iloc[i].date()), round(float(roll.iloc[i]), 4)] for i in idx]


def histogram(r: np.ndarray) -> list[list]:
    edges = np.arange(-1.5, 5.01, 0.25)
    x = np.clip(r, edges[0], edges[-1] - 1e-9)
    cnt, _ = np.histogram(x, bins=edges)
    return [[round(float(a), 2), int(c)] for a, c in zip(edges[:-1], cnt)]


def recent(tr: pd.DataFrame, days: pd.DatetimeIndex) -> list[dict]:
    end = tr["exit_t"].max()
    spans = [("全期", tr["time"].min()), ("2024 年起", pd.Timestamp("2024-01-01", tz="UTC")),
             ("最近 1 年", end - pd.Timedelta(days=365)), ("最近 6 個月", end - pd.Timedelta(days=182)),
             (f"{end.year} 年", pd.Timestamp(f"{end.year}-01-01", tz="UTC"))]
    out = []
    for name, t0 in spans:
        m = metrics(tr[tr["exit_t"] >= t0], days)
        out.append({"name": name, **{k: m.get(k) for k in ("trades", "win_rate", "avg_r", "simple_per_year_pct",
                                                            "max_dd_pct", "sharpe", "t_cluster", "profit_factor")}})
    return out


def clean(x):
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else round(float(x), 4)
    if isinstance(x, np.integer):
        return int(x)
    return x


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="binance_ext", choices=["binance", "binance_ext"])
    ap.add_argument("--out", type=Path, default=Path("web/perf/vegas_backtest.json"))
    a = ap.parse_args()
    bk = books(a.source)
    t0 = min(b["time"].min() for b in bk.values()).floor("D")
    t1 = max(b["exit_t"].max() for b in bk.values()).floor("D")
    days = pd.date_range(t0, t1, freq="D")
    btc = btc_daily(days)
    out = {"source": a.source, "risk": RISK, "generated": str(pd.Timestamp.now(tz="UTC").date()),
           "dates": [str(d.date()) for d in days], "books": {}}
    btc_norm = (btc / btc.dropna().iloc[0] - 1) * 100 if btc.notna().any() else btc
    out["btc_pct"] = [None if pd.isna(v) else round(float(v), 1) for v in btc_norm]
    for name in BOOKS:
        tr = bk[name]
        d = daily_returns(tr, days)
        simple = d.cumsum() * 100
        comp = np.exp(np.log1p(d).cumsum())
        dd = (comp / comp.cummax() - 1) * 100
        out["books"][name] = {
            "metrics": metrics(tr, days), "yearly": yearly(tr, btc), "monthly": monthly(tr),
            "recent": recent(tr, days), "rolling_avg_r": rolling(tr), "histogram": histogram(tr["r"].to_numpy()),
            "curve": {"simple_pct": simple.round(2).tolist(), "compound_pct": ((comp - 1) * 100).round(2).tolist(),
                      "dd_pct": dd.round(2).tolist()},
        }
        if name == "4H+2H":
            out["books"][name]["by_interval"] = {iv: {k: metrics(g, days).get(k) for k in ("trades", "win_rate", "avg_r", "total_r")}
                                                 for iv, g in tr.groupby("interval")}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(clean(out), ensure_ascii=False, separators=(",", ":")))
    # 訊號頁上方那一行摘要用的小檔（不用載整份回測）
    keys = ("trades", "win_rate", "avg_r", "profit_factor", "max_dd_pct", "cagr_pct", "simple_per_year_pct", "start", "end")
    summary = {name: {k: out["books"][name]["metrics"].get(k) for k in keys} for name in BOOKS}
    a.out.with_name("summary.json").write_text(json.dumps(clean(summary), ensure_ascii=False))
    m = out["books"]["4H+2H"]["metrics"]
    print(f"{a.out}：{m['trades']} 筆，每筆 {m['avg_r']:+.3f}R，單利 {m['simple_per_year_pct']:+.1f}%/年，"
          f"CAGR {m['cagr_pct']:+.1f}%，回撤 {m['max_dd_pct']:.1f}%，Sharpe {m['sharpe']:.2f}")


if __name__ == "__main__":
    main()
