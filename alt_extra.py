"""強弱輪動與 VWAP 回踩的細部檢查：多空兩腿、隨機對照、只用老幣的穩健性測試。python alt_extra.py"""

from __future__ import annotations

import sys
import warnings

import numpy as np
import pandas as pd

import alt_momentum as am
import alt_strategies as st
from grid_trading.altsim import TRAIN, VALID, load_bars, random_null, stats, window, coins, FEE

warnings.filterwarnings("ignore")
ALL = (TRAIN[0], VALID[1])


def line(name, s):
    return (f"  {name:<8} n={s['n']:>5} 期望{s['exp_r']:+.3f}R(SE{s['se']:.3f}) 含滑價{s['exp_r_slip']:+.3f}R "
            f"夏普{s['sharpe']:.2f} 年化{s['cagr_pct']:+.0f}% 回撤{s['max_dd_pct']:.0f}%")


def momentum_checks():
    t = pd.read_csv("output/alt_momentum_1d_trades.csv.gz", parse_dates=["time", "exit_time"])
    panel = am.panel()
    cfgs = dict(am.configs())
    for label, cfg in (("正向選中", "mom|L7|H14|k8|sl1.5"), ("反向選中", "mom|L14|H14|k8|sl3.0"),
                       ("穩健性參考", "mom|L14|H7|k5|sl3.0")):
        g = t[t.cfg == cfg]
        print(f"\n[{label}] {cfg}")
        for nm, w in (("選參數段" if label != "反向選中" else "驗證段(2020-23)", TRAIN), ("2024-26", VALID), ("全期", ALL)):
            gw = window(g, w)
            print(line(nm, stats(gw)))
            for side, nm2 in ((1, "  做多腿"), (-1, "  做空腿")):
                s = stats(gw[gw.side == side])
                print(line(nm2, s))
        # 隨機挑幣：同一批再平衡日
        draws = [[] for _ in (TRAIN, VALID)]
        for seed in range(30):
            rows = am.run(cfgs[cfg], *panel, null_rng=np.random.default_rng(seed))
            tf = pd.DataFrame(rows, columns=["sym", "side", "i", "time", "exit_time", "r", "risk_pct", "tpk", "bars"])
            tf["time"] = pd.to_datetime(tf["time"], unit="ns", utc=True)
            for k, w in enumerate((TRAIN, VALID)):
                draws[k].append(window(tf, w)["r"].mean())
        for k, (nm, w) in enumerate((("2020-23", TRAIN), ("2024-26", VALID))):
            real = window(g, w)["r"].mean()
            d = np.array(draws[k])
            print(f"  隨機挑幣 {nm}: 平均 {d.mean():+.3f}R 標準差 {d.std():.3f}（真實 {real:+.3f}R；比隨機高的次數 {int((d >= real).sum())}/30）")
    # 只用 2020 年上半年就有資料的老幣
    data, days, close, dvol, atr, hist = panel
    old = [c for c, b in data.items() if b.t[0] <= pd.Timestamp("2020-07-01", tz="UTC").value]
    sub = ({c: data[c] for c in old}, days, close[old], dvol[old], atr[old], hist[old])
    print(f"\n[只用老幣 {len(old)} 檔（2020-07 前就有資料，避開「事後才進前 100」的偏誤）]")
    for cfg in ("mom|L7|H14|k8|sl1.5", "mom|L14|H14|k8|sl3.0", "mom|L14|H7|k5|sl3.0", "mom|L7|H14|k5|sl1.5"):
        p = dict(cfgs[cfg]); p["k"] = min(p["k"], 5)
        rows = am.run(p, *sub)
        tf = pd.DataFrame(rows, columns=["sym", "side", "i", "time", "exit_time", "r", "risk_pct", "tpk", "bars"])
        tf["time"] = pd.to_datetime(tf["time"], unit="ns", utc=True)
        tf["exit_time"] = pd.to_datetime(tf["exit_time"], unit="ns", utc=True)
        tf["r_slip"] = tf["r"] - 2 * 0.0005 / tf["risk_pct"]
        tf["cfg"] = cfg
        print(f" {cfg} (k={p['k']})")
        for nm, w in (("2020-23", TRAIN), ("2024-26", VALID)):
            print(line(nm, stats(window(tf, w))))


def trend_masks(syms, tf="4h"):
    out, data = {}, {}
    for s in syms:
        b = load_bars(s, tf)
        x = st.base_ind(b)
        top, bot = np.maximum(x["ema144"], x["ema169"]), np.minimum(x["ema144"], x["ema169"])
        up = (b.c > top) & (x["ema144"] > x["ema169"])
        dn = (b.c < bot) & (x["ema144"] < x["ema169"])
        out[s], data[s] = {1: up, -1: dn}, b
    return out, data


def null_checks():
    syms = coins()
    masks, data = trend_masks(syms)
    for fam, f, cfgs in (("VWAP 回踩", "output/alt_vwap_4h_trades.csv.gz", ["vwapPull|week|sl2.5", "vwapPull|month|sl2.5"]),
                         ("基準 4H 維加斯趨勢", "output/alt_vegas_4h_trades.csv.gz", ["vegas|strict", "vegas|current"])):
        t = pd.read_csv(f, parse_dates=["time", "exit_time"])
        for cfg in cfgs:
            g = t[t.cfg == cfg]
            print(f"\n[{fam}] {cfg}")
            for nm, w in (("2020-23", TRAIN), ("2024-26", VALID)):
                gw = window(g, w)
                real = gw["r"].mean()
                a = random_null(gw, data, draws=20, tps_mult=(1, 2, 3))
                b = random_null(gw, data, draws=20, tps_mult=(1, 2, 3), masks=masks)
                print(f"  {nm}: 真實 {real:+.3f}R (n={len(gw)}) | 隨機進場 {a['mean']:+.3f}±{a['sd']:.3f} | "
                      f"隨機進場但趨勢同向 {b['mean']:+.3f}±{b['sd']:.3f}")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "mom"):
        momentum_checks()
    if which in ("all", "null"):
        null_checks()
