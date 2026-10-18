"""牛熊濾網：用 BTC 日線 EMA200、200 週均線判斷牛熊，套在實盤維加斯上（只做分析，不改實盤規則）。

python -m research.exit_unify --source binance      # 先產生 output/exit_unify_trades_binance.csv.gz
python -m research.regime_filter                    → output/regime_filter.csv、output/regime_split.csv

實盤規則（2026-10-15 起）：4H 和 2H 都用 1/3/5R 各平 1/4 的出場，同一個幣同方向只做一張。
牛熊判斷只用前一根已收盤的 K 線：
- 日線 EMA200：BTC 日收盤在日線 EMA200 之上 = 牛市。
- 200 週均線：BTC 週收盤（週日 24:00 UTC）在 200 週簡單均線之上 = 牛市。BTC 日線 2017-08 前用 CoinMarketCap 補。
做法：牛市只做多、熊市只做空（順勢）；或只擋一邊（熊市不做多 / 牛市不做空）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from grid_trading import indicators as ind
from grid_trading.spot_history import load_btc_long
from research.exit_unify import SPLIT, account, dedupe, per_interval


def regimes() -> pd.DataFrame:
    d = load_btc_long().reset_index(drop=True)
    d["time"] = pd.to_datetime(d["time"], unit="ms", utc=True)
    d = d.set_index("time")
    daily = pd.DataFrame({"avail": d.index + pd.Timedelta(days=1),
                          "ema200d": (d["close"] > ind.ema(d["close"], 200)).astype(float).to_numpy()})
    w = d["close"].resample("W-SUN", label="right", closed="right").last()
    ma = w.rolling(200).mean()
    weekly = pd.DataFrame({"avail": w.index + pd.Timedelta(days=1),
                           "sma200w": np.where(ma.isna(), np.nan, (w > ma).astype(float)).astype(float)})
    return daily, weekly


def tag(tr: pd.DataFrame, daily: pd.DataFrame, weekly: pd.DataFrame) -> pd.DataFrame:
    tr = tr.sort_values("time").reset_index(drop=True)
    for reg in (daily, weekly):
        m = pd.merge_asof(tr[["time"]], reg.sort_values("avail"), left_on="time", right_on="avail", direction="backward")
        col = reg.columns[1]
        tr[col] = m[col].to_numpy()
    return tr


def main() -> None:
    allt = pd.read_csv("output/exit_unify_trades_binance.csv.gz")
    for c in ("time", "exit_t"):
        allt[c] = pd.to_datetime(allt[c], utc=True)
    live = pd.concat([per_interval(allt[(allt["interval"] == iv) & (allt["style"] == "2H式")]) for iv in ("4H", "2H")],
                     ignore_index=True)
    daily, weekly = regimes()
    allsig = tag(live, daily, weekly)
    t0, t1 = allsig["time"].min(), allsig["time"].max() + pd.Timedelta(days=1)
    pers = {"全期": (t0, t1), "2020-23": (t0, SPLIT), "2024+": (SPLIT, t1)}
    rules = {
        "不加濾網（現在）": lambda bull, s: np.ones(len(s), bool),
        "牛市只做多、熊市只做空": lambda bull, s: np.where(bull, s > 0, s < 0),
        "熊市不做多（牛市多空都做）": lambda bull, s: bull | (s < 0),
        "牛市不做空（熊市多空都做）": lambda bull, s: ~bull | (s > 0),
    }
    rows = []
    for reg in ("ema200d", "sma200w"):
        bull = allsig[reg].fillna(1).astype(bool).to_numpy()
        for name, f in rules.items():
            # 濾網擋掉的訊號不佔位置：先過濾再做「同幣同向只做一張」
            book = dedupe(allsig[f(bull, allsig["side"].to_numpy())])
            for pn, (a, b) in pers.items():
                rows.append({"regime": reg, "rule": name, "period": pn, **account(book, a, b)})
    out = pd.DataFrame(rows)
    out.to_csv("output/regime_filter.csv", index=False)
    # 不加濾網時，牛/熊 × 多/空 的每筆 R
    base = dedupe(allsig)
    split = []
    for reg in ("ema200d", "sma200w"):
        for pn, (a, b) in pers.items():
            x = base[(base["time"] >= a) & (base["time"] < b)]
            for bull in (True, False):
                for side in (1, -1):
                    y = x[(x[reg].fillna(1).astype(bool) == bull) & (x["side"] == side)]
                    split.append({"regime": reg, "period": pn, "market": "牛" if bull else "熊",
                                  "side": "多" if side > 0 else "空", "trades": len(y),
                                  "win": (y["r"] > 0).mean(), "exp_r": y["r"].mean()})
    sp = pd.DataFrame(split)
    sp.to_csv("output/regime_split.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_rows", 200):
        print(out[["regime", "rule", "period", "trades", "win_rate", "exp_r", "se_cluster", "sum_r_per_year",
                   "sharpe", "max_dd_pct", "worst_day_r"]].round(3).to_string())
        print(sp.round(3).to_string())
    # 各期間的牛市時間比例
    for reg, df in (("ema200d", daily), ("sma200w", weekly)):
        s = df.set_index("avail")[df.columns[1]]
        for pn, (a, b) in pers.items():
            v = s[(s.index >= a) & (s.index < b)].dropna().astype(float)
            print(reg, pn, "牛市比例", round(v.mean(), 3))


if __name__ == "__main__":
    main()
