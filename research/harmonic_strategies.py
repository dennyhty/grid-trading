"""常見諧波型態回測（只做分析，不改實盤規則）。

python -m research.harmonic_strategies    → output/harmonic_summary.csv、output/harmonic_trades.pkl

需要 numba（和 stop_research.py 共用模擬程式）。資料：幣安 USDT 永續 1H，合成 4H/日線。

找轉折：ATR 門檻的 zigzag（2、3、5 倍 ATR，見 threads_strategies.zigzag），只用已收盤的 K 棒。
最後一個轉折確認後，依型態算出 D（進場價）和停損，掛限價單；下一個轉折確認或 100 根內沒成交就取消，
確認時價格已經越過 D 就不追。成交那根同時碰到停損一律算停損。停損外再加 0.1 ATR。

型態（X、A、B、C 是最後四個轉折；比例都是 Scott Carney 的標準值，容許範圍見 PATTERNS）：
- 加特利 Gartley：AB = 0.618 XA，BC = 0.382~0.886 AB，D = 0.786 XA，停損在 X。
- 蝙蝠 Bat：AB = 0.382~0.5 XA，D = 0.886 XA，停損在 1.13 XA。
- 另類蝙蝠 Alt Bat：AB <= 0.382 XA，D = 1.13 XA，停損在 1.27 XA。
- 蝴蝶 Butterfly：AB = 0.786 XA，D = 1.27 XA，停損在 1.414 XA。
- 螃蟹 Crab：AB = 0.382~0.618 XA，D = 1.618 XA，停損在 1.902 XA。
- 深螃蟹 Deep Crab：AB = 0.886 XA，D = 1.618 XA，停損在 1.902 XA。
- Cypher：AB = 0.382~0.618 XA，C 超過 A 達 XA 的 1.13~1.414，D = XC 的 0.786，停損在 X。
- 鯊魚 Shark（O、X、A、B 四個轉折）：B 超過 X 達 XA 的 1.13~1.618，D = OX 的 0.886，停損在 1.13 OX。
- AB=CD（A、B、C 三個轉折）：BC = 0.382~0.886 AB，D = C 加上一段 AB，停損在 1.27 倍 AB。
另外要求 CD 是 BC 的合理延伸（PATTERNS 的 cd 欄；AB=CD、鯊魚不檢查）。

出場：harm = 諧波常用的目標，最後一段（進 D 的那段）回撤 0.382 平一半、保本、0.618 平剩下；
q135 = 實盤式 1/3/5R 各平 1/4 加 3 ATR 追蹤；tp1 = 1R 全出；tp2 = 2R 全出。手續費每邊 0.05%。
同一個幣同一個型態同時只持有一張。
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from grid_trading.universe import TOP100
from research.stop_research import SPLIT, busy_filter, cluster_stats, sim_many
from research.threads_strategies import FEE, MAX_BARS, fill_limit, frame, zigzag

OUT = Path("output")
# 名稱: (AB/XA 範圍, BC/AB 範圍, D 在 XA 的位置, 停損在 XA 的位置, CD/BC 範圍)
PATTERNS = {
    "gartley": ((0.58, 0.66), (0.382, 0.886), 0.786, 1.0, (1.13, 1.618)),
    "bat": ((0.382, 0.5), (0.382, 0.886), 0.886, 1.13, (1.618, 2.618)),
    "alt_bat": ((0.25, 0.382), (0.382, 0.886), 1.13, 1.27, (2.0, 3.618)),
    "butterfly": ((0.74, 0.82), (0.382, 0.886), 1.27, 1.414, (1.618, 2.618)),
    "crab": ((0.382, 0.618), (0.382, 0.886), 1.618, 1.902, (2.24, 3.618)),
    "deep_crab": ((0.85, 0.92), (0.382, 0.886), 1.618, 1.902, (2.0, 3.618)),
}
EXITS = {  # 名稱: (TP 的種類, 平倉比例, 追蹤幾倍 ATR)；"leg" 是最後一段的回撤比例，其他是 R 倍數
    "harm": (("leg", (0.382, 0.618)), (0.5, 0.5), 0.0),
    "q135": (("r", (1.0, 3.0, 5.0)), (0.25, 0.25, 0.25), 3.0),
    "tp1": (("r", (1.0,)), (1.0,), 0.0),
    "tp2": (("r", (2.0,)), (1.0,), 0.0),
}


def inr(x: float, lo_hi: tuple[float, float]) -> bool:
    return lo_hi[0] <= x <= lo_hi[1]


def setups(piv: list[tuple]) -> list[tuple]:
    """每個轉折確認時檢查所有型態。回傳 (型態, 確認索引, 方向, D, 停損（未加緩衝）, 最後一段起點, 下一轉折確認索引)。"""
    out = []
    for n in range(3, len(piv)):
        nxt = piv[n + 1][3] if n + 1 < len(piv) else 10**9
        x, a, b, c = (p[1] for p in piv[n - 3:n + 1])
        conf = piv[n][3]
        d = piv[n][2]                     # 最後一個轉折是高點 → 往下找 D 做多
        xa, ab, bc = abs(a - x), abs(b - a), abs(c - b)
        if xa == 0 or ab == 0 or bc == 0:
            continue
        for name, (r_ab, r_bc, r_d, r_s, r_cd) in PATTERNS.items():
            if not (inr(ab / xa, r_ab) and inr(bc / ab, r_bc)):
                continue
            dp = a + r_d * (x - a)
            if not inr(abs(dp - c) / bc, r_cd):
                continue
            out.append((name, conf, d, dp, a + r_s * (x - a), c, nxt))
        # Cypher：C 超過 A，D 在 XC 的 0.786
        if inr(ab / xa, (0.382, 0.618)) and inr(abs(c - x) / xa, (1.13, 1.414)) and d * (c - a) > 0:
            out.append(("cypher", conf, d, c + 0.786 * (x - c), x, c, nxt))
        # 鯊魚：O、X、A、B（這裡的 x、a、b、c），B 超過 X 達 XA 的 1.13~1.618，D 在 OX 的 0.886
        o_, x_, a_, b_ = x, a, b, c
        ox, xa_ = abs(x_ - o_), abs(a_ - x_)
        if ox > 0 and xa_ > 0 and inr(abs(b_ - a_) / xa_, (1.13, 1.618)):
            out.append(("shark", conf, d, x_ + 0.886 * (o_ - x_), x_ + 1.13 * (o_ - x_), b_, nxt))
        # AB=CD：最後三個轉折
        if inr(bc / ab, (0.382, 0.886)):
            out.append(("abcd", conf, d, c + (b - a), c + 1.27 * (b - a), c, nxt))
    return out


def _coin(args):
    coin, tf = args
    df = frame(coin, tf)
    if df is None:
        return []
    op, hi, lo, cl, atr = (df[k].to_numpy() for k in ("open", "high", "low", "close", "atr"))
    rows = []
    for k in (2.0, 3.0, 5.0):
        for name, conf, d, dp, sp, leg0, nxt in setups(zigzag(df, k)):
            if conf + 1 >= len(df) or np.isnan(atr[conf]):
                continue
            stop = sp - d * 0.1 * atr[conf]
            if d * (cl[conf] - dp) <= 0 or d * (dp - stop) <= 0:   # 已經越過 D、或停損在錯的一邊
                continue
            j = fill_limit(df, conf + 1, min(nxt, conf + 101), d, dp)
            if j < 0:
                continue
            same = (lo[j] <= stop) if d > 0 else (hi[j] >= stop)
            rows.append((name, k, j, d, dp, stop, leg0, same, conf))
    if not rows:
        return []
    t = pd.DataFrame(rows, columns=["pattern", "k", "fill", "side", "entry", "stop", "leg0", "same", "sig"])
    isc = np.ones(len(hi), np.bool_)
    starts = (t["fill"] + 1).to_numpy(np.int64)
    ds, entry, stop = t["side"].to_numpy(np.int64), t["entry"].to_numpy(), t["stop"].to_numpy()
    risk = np.abs(entry - stop)
    zero = np.zeros(len(t))
    same = t["same"].to_numpy(bool)
    for ex, ((kind_, levels), portions, trail_k) in EXITS.items():
        if kind_ == "r":
            tps = np.stack([entry + ds * lv * risk for lv in levels], axis=1)
        else:
            leg = t["leg0"].to_numpy() - entry
            tps = np.stack([entry + lv * leg for lv in levels], axis=1)
        trails = trail_k * atr[t["fill"].to_numpy()]
        r, hit, jx, kind, _ = sim_many(op, hi, lo, cl, isc, np.minimum(starts, len(df) - 1), ds, entry, risk, stop,
                                       tps, np.array(portions), 0, trails, FEE, 0.0, MAX_BARS[tf], 0, 0, zero, 0.0,
                                       0.0, zero, 0, cl, cl, False)
        t[f"r_{ex}"] = np.where(same, (-risk - FEE * entry - FEE * stop) / risk, r)
        t[f"kind_{ex}"] = np.where(same, 1, np.where(starts >= len(df), -1, kind))
        t[f"j_{ex}"] = np.where(same, t["fill"], jx)
    t["time"] = df["time"].to_numpy()[t["fill"].to_numpy()]
    t["risk_pct"] = risk / entry
    t["reward_leg"] = np.abs(t["leg0"] - entry) / risk          # 最後一段有幾 R（0.382 回撤大約是它的 0.382 倍）
    t["d_close"] = df["d_close"].to_numpy()[t["sig"].to_numpy()]
    t["d_ema200"] = df["d_ema200"].to_numpy()[t["sig"].to_numpy()]
    t["coin"], t["tf"] = coin, tf
    return [t]


def stats(g: pd.DataFrame, ex: str) -> dict:
    r = g[f"r_{ex}"].to_numpy()
    key = (pd.to_datetime(g["time"]).dt.floor("D").astype(str) + g["side"].astype(str)).to_numpy()
    mean, se, _, _, ncl = cluster_stats(r, key)
    return {"trades": len(r), "win": (r > 0).mean() if len(r) else np.nan, "exp_r": mean,
            "t_cl": mean / se if se else np.nan, "clusters": ncl}


def main() -> None:
    jobs = [(c, tf) for tf in ("1D", "4H", "1H") for c in TOP100]
    with ProcessPoolExecutor() as ex:
        parts = [p for res in ex.map(_coin, jobs, chunksize=4) for p in res]
    allr = pd.concat(parts, ignore_index=True)
    allr["time"] = pd.to_datetime(allr["time"], utc=True)
    OUT.mkdir(exist_ok=True)
    allr.to_pickle(OUT / "harmonic_trades.pkl")
    rows = []
    for (tf, pat, k), g in allr.groupby(["tf", "pattern", "k"]):
        g = g.sort_values(["coin", "fill"])
        for ex_name in EXITS:
            cid = pd.factorize(g["coin"])[0]
            take = busy_filter(cid, g["fill"].to_numpy(np.int64), g[f"j_{ex_name}"].to_numpy(np.int64),
                               g[f"kind_{ex_name}"].to_numpy(np.int64))
            gg = g[take]
            trend = (gg["side"] * (gg["d_close"] - gg["d_ema200"]) > 0).to_numpy()
            for fn, fm in (("all", np.ones(len(gg), bool)), ("ema200", trend)):
                for per, pm in (("全期", np.ones(len(gg), bool)), ("2020-23", (gg["time"] < SPLIT).to_numpy()),
                                ("2024+", (gg["time"] >= SPLIT).to_numpy())):
                    rows.append({"tf": tf, "pattern": pat, "k": k, "exit": ex_name, "filter": fn, "period": per,
                                 **stats(gg[fm & pm], ex_name)})
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "harmonic_summary.csv", index=False)
    best = out[(out["filter"] == "all")].pivot_table(index=["tf", "pattern", "k", "exit"], columns="period",
                                                      values=["exp_r", "t_cl", "win", "trades"])
    with pd.option_context("display.width", 250, "display.max_rows", 1000):
        print(best.round(3).to_string())


if __name__ == "__main__":
    main()
