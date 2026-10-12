"""橫斷面強弱輪動：python alt_momentum.py [--out output]

每隔 H 天（週一 00:00 UTC，也就是週日日線收盤）把流動性夠的幣依過去 L 天漲幅排名：
- mom：做多最強 k 個、做空最弱 k 個；rev：反過來（短期反轉）。
- 每筆單的 1R = stop 倍的日線 ATR(14)，到下次再平衡就收盤平倉，中間碰到止損先出場。
- 流動性門檻：過去 30 天平均日成交額 >= 10M USDT（幣安），至少 L+30 天的歷史。
- 隨機對照：同一批再平衡日、同樣的多空筆數，從同一天符合條件的幣裡隨機挑。
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from grid_trading import indicators as ind
from grid_trading.altsim import FEE, coins, load_bars, sim, to_frame

DAY = 86_400_000_000_000
MONDAY = pd.Timestamp("2023-01-02", tz="UTC").value // DAY  # 週一的日序號
MIN_DOLLAR_VOL = 10e6


def configs():
    for L, H, k, mode, stop in itertools.product((7, 14, 28, 56), (7, 14), (3, 5, 8), ("mom", "rev"), (1.5, 3.0)):
        yield f"{mode}|L{L}|H{H}|k{k}|sl{stop}", dict(L=L, H=H, k=k, mode=mode, stop=stop)


def panel():
    data = {c: load_bars(c, "1d") for c in coins()}
    data = {c: b for c, b in data.items() if b is not None and len(b.t) > 120}
    days = pd.Index(sorted({int(t // DAY) for b in data.values() for t in b.t}))
    close = pd.DataFrame({c: pd.Series(b.c, index=b.t // DAY) for c, b in data.items()}).reindex(days)
    dvol = pd.DataFrame({c: pd.Series(b.c * b.v, index=b.t // DAY) for c, b in data.items()}).reindex(days)
    atr = pd.DataFrame({c: pd.Series(ind.atr(b.df).to_numpy(), index=b.t // DAY) for c, b in data.items()}).reindex(days)
    hist = close.notna().cumsum()
    return data, days, close, dvol.rolling(30, min_periods=20).mean(), atr, hist


def day_index(b, day):
    k = np.searchsorted(b.t // DAY, day)
    return int(k) if k < len(b.t) and b.t[k] // DAY == day else None


def run(p: dict, data, days, close, dvol, atr, hist, null_rng=None):
    L, H, k, mode = p["L"], p["H"], p["k"], p["mode"]
    ret = close / close.shift(L) - 1
    rows = []
    for day in days:
        if (day - MONDAY) % H != H - 1:  # 週日收盤再平衡
            continue
        ok = (dvol.loc[day] >= MIN_DOLLAR_VOL) & (hist.loc[day] >= L + 30) & ret.loc[day].notna() & atr.loc[day].notna()
        elig = ret.loc[day][ok]
        if len(elig) < 2 * k + 4:
            continue
        if null_rng is None:
            order = (elig if mode == "mom" else -elig).sort_values()  # 分數由小到大：尾端最強做多、前端最弱做空
            longs, shorts = list(order.index[-k:]), list(order.index[:k])
        else:
            pick = null_rng.choice(elig.index.to_numpy(), size=2 * k, replace=False)
            longs, shorts = list(pick[:k]), list(pick[k:])
        for side, names in ((1, longs), (-1, shorts)):
            for c in names:
                b = data[c]
                i = day_index(b, day)
                if i is None:
                    continue
                risk = p["stop"] * atr.at[day, c]
                res = sim(b, i, side, risk, [], FEE, H)
                if res is None:
                    continue
                r, j, _ = res
                rows.append((c, side, i, b.t[i] + b.bar_ns, b.t[j], r, risk / b.cl[i], 0.0, j - i))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="output")
    ap.add_argument("--null-draws", type=int, default=0, help="每個設定的隨機對照次數（只輸出期望值）")
    a = ap.parse_args()
    data, days, close, dvol, atr, hist = panel()
    all_rows, nulls = [], []
    for cfg, p in configs():
        for c, side, i, t, xt, r, rp, tpk, bars in run(p, data, days, close, dvol, atr, hist):
            all_rows.append((cfg, c, side, i, t, xt, r, rp, tpk, bars))
    t = to_frame(all_rows)
    Path(a.out).mkdir(exist_ok=True)
    t.to_csv(Path(a.out) / "alt_momentum_1d_trades.csv.gz", index=False)
    print(len(t), "trades")
    return 0


if __name__ == "__main__":
    sys.exit(main())
