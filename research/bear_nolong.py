"""深入研究「BTC 跌破日線 EMA200 就不做多」（只做分析，不改實盤規則）。

python -m research.exit_unify --source binance     # 先產生 output/exit_unify_trades_binance.csv.gz
python -m research.bear_nolong                     → output/bear_nolong_*.csv

實盤規則（2026-10-15 起）：4H 和 2H 都用 1/3/5R 出場，同一個幣同方向只做一張；每筆 0.25% 風險。
1. 逐年、4H/2H 分開：濾網前後的帳戶績效，以及被擋掉的多單本身賺賠。
2. 參數敏感度：EMA 長度 100/150/200/250、要跌破多少才算（0/2/5%）、要連續幾天在下面（1/3/7）、
   改看 EMA200 是否下彎、改用每個幣自己的日線 EMA200。
3. 風險：最大回撤、最差一天、同時持有的多單數、30 天一塊重抽 Sharpe 差（和現在比）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from grid_trading import indicators as ind
from grid_trading.binance_history import load
from grid_trading.history import resample
from grid_trading.spot_history import load_btc_long
from research.exit_unify import SPLIT, account, dedupe, per_interval

RISK = 0.0025


def btc_daily() -> pd.DataFrame:
    d = load_btc_long().reset_index(drop=True)
    d["time"] = pd.to_datetime(d["time"], unit="ms", utc=True)
    return d.set_index("time")["close"]


def bear_flag(close: pd.Series, n: int = 200, buf: float = 0.0, days: int = 1, slope: bool = False) -> pd.DataFrame:
    """回傳 (可用時間, 熊市)。熊市 = 收盤低於 EMA(n) 超過 buf，而且連續 days 天；slope=True 改成 EMA 比 10 天前低。"""
    e = ind.ema(close, n)
    if slope:
        bear = e < e.shift(10)
    else:
        bear = close < e * (1 - buf)
        if days > 1:
            bear = bear.rolling(days).sum() >= days
    bear = bear.where(np.arange(len(close)) >= n, False)
    return pd.DataFrame({"avail": close.index + pd.Timedelta(days=1), "bear": bear.astype(float).to_numpy()})


def attach(tr: pd.DataFrame, flag: pd.DataFrame, col: str) -> pd.DataFrame:
    m = pd.merge_asof(tr[["time"]], flag.sort_values("avail"), left_on="time", right_on="avail", direction="backward")
    tr[col] = m["bear"].fillna(0).astype(bool).to_numpy()
    return tr


def coin_bear(tr: pd.DataFrame) -> np.ndarray:
    out = np.zeros(len(tr), bool)
    for coin, idx in tr.groupby("coin").groups.items():
        df = load(coin, "1h")
        if df.empty:
            continue
        c = resample(df, "1D", origin="epoch").set_index("time")["close"]
        f = bear_flag(c)
        sub = tr.loc[idx, ["time"]].sort_values("time")
        m = pd.merge_asof(sub, f, left_on="time", right_on="avail", direction="backward")
        out[tr.index.get_indexer(sub.index)] = m["bear"].fillna(0).astype(bool).to_numpy()
    return out


def daily_pnl(book: pd.DataFrame, idx: pd.DatetimeIndex) -> np.ndarray:
    s = (RISK * book.set_index(book["exit_t"].dt.floor("D"))["r"]).groupby(level=0).sum()
    return s.reindex(idx, fill_value=0).to_numpy()


def max_open_longs(book: pd.DataFrame) -> int:
    ev = pd.concat([pd.Series(1, index=book.loc[book.side > 0, "time"]),
                    pd.Series(-1, index=book.loc[book.side > 0, "exit_t"])]).sort_index()
    return int(ev.cumsum().max()) if len(ev) else 0


def main() -> None:
    allt = pd.read_csv("output/exit_unify_trades_binance.csv.gz")
    for c in ("time", "exit_t"):
        allt[c] = pd.to_datetime(allt[c], utc=True)
    sig = pd.concat([per_interval(allt[(allt["interval"] == iv) & (allt["style"] == "2H式")]) for iv in ("4H", "2H")],
                    ignore_index=True).sort_values("time").reset_index(drop=True)
    btc = btc_daily()
    variants = {"EMA200（原版）": bear_flag(btc)}
    for n in (100, 150, 250):
        variants[f"EMA{n}"] = bear_flag(btc, n)
    for buf in (0.02, 0.05):
        variants[f"EMA200 跌破 {int(buf * 100)}%"] = bear_flag(btc, buf=buf)
    for days in (3, 7):
        variants[f"EMA200 連續 {days} 天"] = bear_flag(btc, days=days)
    variants["EMA200 下彎"] = bear_flag(btc, slope=True)
    for name, f in variants.items():
        sig = attach(sig, f, name)
    sig["個幣自己的 EMA200"] = coin_bear(sig)
    names = list(variants) + ["個幣自己的 EMA200"]

    t0, t1 = sig["time"].min(), sig["time"].max() + pd.Timedelta(days=1)
    pers = {"全期": (t0, t1), "2020-23": (t0, SPLIT), "2024+": (SPLIT, t1)}
    base = dedupe(sig)
    books = {"現在": base}
    for nm in names:
        books[nm] = dedupe(sig[~(sig[nm] & (sig["side"] > 0))])

    # 1. 帳戶績效 + 被擋掉的多單
    rows = []
    for nm, book in books.items():
        for pn, (a, b) in pers.items():
            r = account(book, a, b)
            blocked = sig[(sig["time"] >= a) & (sig["time"] < b) & (sig["side"] > 0)]
            blocked = blocked[blocked[nm]] if nm != "現在" else blocked.iloc[:0]
            rows.append({"variant": nm, "period": pn, **r, "max_open_longs": max_open_longs(book[(book.time >= a) & (book.time < b)]),
                         "blocked": len(blocked), "blocked_r": blocked["r"].mean() if len(blocked) else np.nan,
                         "blocked_win": (blocked["r"] > 0).mean() if len(blocked) else np.nan})
    acc = pd.DataFrame(rows)
    acc.to_csv("output/bear_nolong_variants.csv", index=False)

    # 逐年、4H/2H（原版）
    rows = []
    nm = "EMA200（原版）"
    for y in range(2020, 2027):
        a, b = pd.Timestamp(f"{y}-01-01", tz="UTC"), pd.Timestamp(f"{y + 1}-01-01", tz="UTC")
        for iv in ("4H", "2H", "合計"):
            def pick(bk):
                return bk if iv == "合計" else bk[bk["interval"] == iv]
            r0, r1 = account(pick(books["現在"]), a, b), account(pick(books[nm]), a, b)
            bl = sig[(sig.time >= a) & (sig.time < b) & (sig.side > 0) & sig[nm]]
            bl = pick(bl)
            rows.append({"year": y, "interval": iv, "trades_now": r0.get("trades"), "sum_r_now": r0.get("sum_r"),
                         "trades_filter": r1.get("trades"), "sum_r_filter": r1.get("sum_r"),
                         "dd_now": r0.get("max_dd_pct"), "dd_filter": r1.get("max_dd_pct"),
                         "blocked": len(bl), "blocked_sum_r": bl["r"].sum(), "blocked_mean_r": bl["r"].mean() if len(bl) else np.nan})
    yearly = pd.DataFrame(rows)
    yearly.to_csv("output/bear_nolong_yearly.csv", index=False)

    # 3. 重抽：Sharpe 差（濾網 − 現在），30 天一塊
    idx = pd.date_range(t0.floor("D"), t1.floor("D"), freq="D")
    a0 = daily_pnl(books["現在"], idx)
    rng = np.random.default_rng(7)
    sh = lambda x: x.mean() / x.std() * np.sqrt(365)
    rows = []
    for nm in names:
        b0 = daily_pnl(books[nm], idx)
        for pn, m in (("全期", np.ones(len(idx), bool)), ("2020-23", idx < SPLIT), ("2024+", idx >= SPLIT)):
            x, y = a0[m], b0[m]
            T = len(x)
            ds = []
            for _ in range(2000):
                ii = (rng.integers(0, T - 30, size=T // 30 + 1)[:, None] + np.arange(30)).ravel()[:T]
                ds.append(sh(y[ii]) - sh(x[ii]))
            ds = np.array(ds)
            rows.append({"variant": nm, "period": pn, "sharpe_now": sh(x), "sharpe_filter": sh(y),
                         "diff": sh(y) - sh(x), "ci05": np.quantile(ds, 0.05), "ci95": np.quantile(ds, 0.95),
                         "p_better": (ds > 0).mean()})
    boot = pd.DataFrame(rows)
    boot.to_csv("output/bear_nolong_bootstrap.csv", index=False)

    cols = ["variant", "period", "trades", "win_rate", "exp_r", "sum_r_per_year", "sharpe", "max_dd_pct", "worst_day_r",
            "max_open_longs", "blocked", "blocked_r", "blocked_win"]
    with pd.option_context("display.width", 250, "display.max_rows", 300):
        print(acc[cols].round(3).to_string())
        print(yearly.round(2).to_string())
        print(boot.round(3).to_string())


if __name__ == "__main__":
    main()
