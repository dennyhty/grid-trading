"""Larry Williams 的「外包日」反轉法，在派網的加密貨幣永續合約上回測（2026-10-15）。

原始規則（做多，日線）：
1. 外包日：當天最高 > 前一天最高、最低 < 前一天最低，而且收盤 < 前一天最低。
2. 隔天開盤 < 外包日收盤（低開）才用開盤價買進。
3. 固定金額停損；沒停損就在「第一個開盤價高於買價」的那天開盤賣出。

加密貨幣 24 小時交易，日線開盤幾乎等於前一天收盤，所以低開條件拆成兩個版本：嚴格（開盤 < 收盤）和不看。
固定金額停損換成 k 倍 ATR(14)，另外測停在外包日最低點。最多抱 max_hold 根 K 線，到期以收盤價出場。
做空是鏡像：外包日收盤 > 前一天最高，隔天高開才賣出，第一個開盤價低於賣價時回補。
基準：每一根 K 線都進場、出場規則相同，看外包日這個條件到底有沒有比亂進場好。

成本：手續費 0.05% x 2 + 滑價 0.05% x 2（停損另加滑價 0.05%）。R = 損益 / 停損距離。
用法：python -m research.outside_day [--pionex]
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from grid_trading.binance_history import fetch_binance
from grid_trading.universe import TOP100

OUT = Path("output/outside_day")
FEE = 0.0005
SLIP = 0.0005
STOP_SLIP = 0.0005
STOPS = ["1atr", "2atr", "3atr", "bar"]
MAX_HOLD = {"1d": 20, "4h": 60}


def atr(df: pd.DataFrame, n: int = 14) -> np.ndarray:
    h, l, c = df["high"].to_numpy(), df["low"].to_numpy(), df["close"].to_numpy()
    pc = np.r_[np.nan, c[:-1]]
    tr = np.nanmax(np.c_[h - l, np.abs(h - pc), np.abs(l - pc)], axis=1)
    return pd.Series(tr).ewm(alpha=1 / n, adjust=False, min_periods=n).mean().to_numpy()


def simulate(df: pd.DataFrame, side: str, stop: str, gap: bool, max_hold: int, baseline: bool = False) -> list[dict]:
    o, h, l, c = (df[k].to_numpy() for k in ("open", "high", "low", "close"))
    t = df["time"].to_numpy()
    a = atr(df)
    n = len(df)
    out = []
    i = 15
    while i < n - 2:
        s = 1 if side == "long" else -1
        if baseline:
            sig = True
        elif s == 1:
            sig = h[i] > h[i - 1] and l[i] < l[i - 1] and c[i] < l[i - 1]
        else:
            sig = h[i] > h[i - 1] and l[i] < l[i - 1] and c[i] > h[i - 1]
        if not sig or not np.isfinite(a[i]):
            i += 1
            continue
        e = i + 1
        if gap and not (s * (c[i] - o[e]) > 0):
            i += 1
            continue
        entry = o[e]
        if stop == "bar":
            sp = l[i] if s == 1 else h[i]
        else:
            sp = entry - s * float(stop[0]) * a[i]
        risk = s * (entry - sp)
        if risk <= entry * 0.002:  # 停損太近（例如開盤已經在外包日最低點下方）就不做
            i += 1
            continue
        exit_px, j, why = None, e, ""
        while True:
            if j > e and s * (o[j] - entry) > 0:  # 第一個獲利開盤
                exit_px, why = o[j], "profit_open"
                break
            if j > e and s * (o[j] - sp) <= 0:  # 開盤就跳過停損
                exit_px, why = o[j] * (1 - s * STOP_SLIP), "stop"
                break
            if (s == 1 and l[j] <= sp) or (s == -1 and h[j] >= sp):
                exit_px, why = sp * (1 - s * STOP_SLIP), "stop"
                break
            if j - e + 1 >= max_hold or j == n - 1:
                exit_px, why = c[j], "time"
                break
            j += 1
        gross = s * (exit_px - entry) / entry
        net = gross - 2 * (FEE + SLIP)
        out.append({"t_in": t[e], "t_out": t[j], "r": net / (risk / entry), "ret": net,
                    "bars": j - e + 1, "why": why, "stop_pct": risk / entry})
        i = j if not baseline else i + 1  # 同一個幣持倉中不重複進場（基準例外，每根都算）
    return out


def cluster_t(r: np.ndarray, key: np.ndarray) -> tuple[float, float]:
    n = len(r)
    if n < 2:
        return np.nan, np.nan
    s = pd.Series(r).groupby(key)
    m = r.mean()
    se = np.sqrt(((s.sum() - s.count() * m) ** 2).sum()) / n
    return m, m / se if se > 0 else np.nan


def summarize(trades: pd.DataFrame, label: dict) -> list[dict]:
    rows = []
    if trades.empty:
        return rows
    trades = trades.copy()
    trades["day"] = pd.to_datetime(trades["t_in"]).dt.floor("D")
    for period, sel in [("all", slice(None)), ("2020-23", trades["day"] < "2024-01-01"),
                        ("2024+", trades["day"] >= "2024-01-01")]:
        g = trades.loc[sel]
        if len(g) < 2:
            continue
        r = g["r"].to_numpy()
        m, tt = cluster_t(r, g["day"].to_numpy())
        rows.append({**label, "period": period, "trades": len(g), "win": (r > 0).mean(), "mean_r": m,
                     "cluster_t": tt, "mean_ret_pct": 100 * g["ret"].mean(), "avg_bars": g["bars"].mean(),
                     "stop_share": (g["why"] == "stop").mean(), "time_share": (g["why"] == "time").mean(),
                     "worst_r": r.min(), "per_year": len(g) / max((g["day"].max() - g["day"].min()).days / 365.25, 0.1)})
    return rows


def load_binance(coins: list[str], interval: str) -> dict[str, pd.DataFrame]:
    with ThreadPoolExecutor(6) as ex:
        dfs = list(ex.map(lambda c: fetch_binance(c, interval), coins))
    return {c: d for c, d in zip(coins, dfs) if len(d) > 100}


def load_pionex(coins: list[str]) -> dict[str, pd.DataFrame]:
    from grid_trading.history import fetch_history
    out = {}
    for c in coins:
        try:
            d = fetch_history(f"{c}_USDT_PERP", "1D", days=1400)
        except Exception:
            continue
        if len(d) > 100:
            out[c] = d
    return out


def run(data: dict[str, pd.DataFrame], tf: str, source: str) -> pd.DataFrame:
    rows = []
    for side in ("long", "short"):
        for stop in STOPS:
            for gap in (True, False):
                for base in (False, True):
                    if base and (gap or stop == "bar"):
                        continue
                    tr = []
                    for coin, df in data.items():
                        for x in simulate(df, side, stop, gap, MAX_HOLD[tf], baseline=base):
                            tr.append({**x, "coin": coin})
                    rows += summarize(pd.DataFrame(tr), {"source": source, "tf": tf, "side": side, "stop": stop,
                                                         "gap": gap, "rule": "baseline" if base else "outside_day"})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pionex", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    coins = [c for c in TOP100 if c.isascii()]
    res = []
    for tf in ("1d", "4h"):
        data = load_binance(coins, tf)
        print(tf, "coins", len(data))
        res.append(run(data, tf, "binance"))
    if args.pionex:
        res.append(run(load_pionex(coins), "1d", "pionex"))
    df = pd.concat(res)
    df.to_csv(OUT / "outside_day_summary.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 500)
    print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
