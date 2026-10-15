"""vegas_trend 止損單共同特徵（只做分析，不改實盤規則）。

python -m research.stoploss_profile --binance-4h output/entry_signals_4H.csv.gz --binance-2h output/entry_signals_2H.csv.gz \
    --pionex-4h output/entry_signals_4H_pionex.csv.gz --pionex-2h output/entry_signals_2H_pionex.csv.gz

輸入是 entry_research.py 第一步的輸出（每個訊號在收盤時的 40 個特徵和實盤出場的 R）。
「止損單」= 還沒碰到 TP1 就被 1.5 ATR 止損打掉（hit == 0 且 kind == 1，約 −1R）。
實盤訊號 = live（2H 再加 ADX < 24），同一個幣同時只持有一張。

輸出（output/）：
- stoploss_auc_{週期}.csv：每個特徵分辨止損單的 AUC（0.5 = 完全分不出來），近 3 個月 vs 2020-23 vs 2024~2026/6。
- stoploss_filters_{週期}.csv：近 3 個月止損單最明顯的特徵，拿去當過濾（用較早期間的五分位門檻）時，被去掉那些單在各期間的平均 R。
- stoploss_reentry.csv：同一個幣、同方向、止損後幾天內又出訊號就跳過，帳戶 Sharpe 變化（30 天區塊重抽）。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from research.entry_research import FEATURES
from research.stop_research import busy_filter, cluster_key, cluster_stats

RECENT = pd.Timestamp("2026-07-01", tz="UTC")
SPLIT = pd.Timestamp("2024-01-01", tz="UTC")
CANDIDATES = {  # 近 3 個月止損單最明顯的特徵（看完 AUC 表挑出來的）：(特徵, 去掉高的還是低的)
    "4H": [("funding3d_o", "low"), ("macd_o", "low"), ("ret7_o", "low"), ("atr_ratio", "low"), ("vol_ratio", "high"),
           ("long_chan", "high"), ("range_atr", "high"), ("ext_atr", "high"), ("rsi_o", "high")],
    "2H": [("range_atr", "high"), ("vol_ratio", "high"), ("body_o", "high"), ("rank7_o", "low"), ("adx", "low"),
           ("weekday", "low"), ("ext_atr", "high"), ("rsi_o", "high")],
}


def load_live(path: str, interval: str) -> pd.DataFrame:
    f = pd.read_csv(path)
    for c in ("time", "signal_close", "exit_t"):
        f[c] = pd.to_datetime(f[c], utc=True)
    lv = f[f["live"]]
    if interval == "2H":
        lv = lv[lv["adx"] < 24]  # 2026-10-13 起實盤規則
    lv = lv[lv["kind"] != -1].sort_values(["coin", "i"]).reset_index(drop=True)
    take = busy_filter(pd.factorize(lv["coin"])[0], lv["i"].to_numpy(np.int64), lv["j"].to_numpy(np.int64),
                       lv["kind"].to_numpy(np.int64))
    lv = lv[take].reset_index(drop=True)
    lv["loss"] = (lv["hit"] == 0) & (lv["kind"] == 1)
    return lv


def periods(lv: pd.DataFrame) -> dict[str, np.ndarray]:
    t = lv["time"]
    first = "2020-23" if t.min() < SPLIT else "2023"
    return {first: (t < SPLIT).to_numpy(), "2024~2026/6": ((t >= SPLIT) & (t < RECENT)).to_numpy(),
            "近3個月": (t >= RECENT).to_numpy()}


def auc(x: pd.Series, y: pd.Series) -> float:
    m = x.notna()
    x, y = x[m], y[m]
    n1, n0 = int(y.sum()), int((~y).sum())
    if not n1 or not n0:
        return np.nan
    return (x.rank()[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def auc_table(lv: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feat, name in FEATURES.items():
        if feat not in lv:
            continue
        row = {"feature": feat, "name": name}
        for p, m in periods(lv).items():
            g = lv[m]
            row[f"auc_{p}"] = auc(g[feat], g["loss"])
            row[f"止損單中位數_{p}"] = g.loc[g["loss"], feat].median()
            row[f"其他中位數_{p}"] = g.loc[~g["loss"], feat].median()
        rows.append(row)
    return pd.DataFrame(rows)


def filter_table(lv: pd.DataFrame, interval: str, source: str) -> pd.DataFrame:
    per = periods(lv)
    key = cluster_key(lv)
    base = next(iter(per.values())) if source == "binance" else ~per["近3個月"]  # 門檻用最早那段（派網 2H 歷史短，用近 3 個月以前）
    rows = []
    for feat, side in CANDIDATES[interval]:
        x = lv[feat]
        if x[base].notna().sum() < 50:
            continue
        thr = np.nanquantile(x[base], 0.8 if side == "high" else 0.2)
        drop = ((x > thr) if side == "high" else (x < thr)).to_numpy()
        for p, m in per.items():
            d = m & drop
            mean, se, *_ = cluster_stats(lv["r"].to_numpy()[d], key[d])
            rows.append({"source": source, "feature": feat, "name": FEATURES[feat],
                         "drop": f"{'>' if side == 'high' else '<'}{thr:.3g}", "period": p, "dropped_n": int(d.sum()),
                         "dropped_stop_rate": lv["loss"][d].mean(), "kept_stop_rate": lv["loss"][m & ~drop].mean(),
                         "dropped_r": mean, "dropped_se_cluster": se, "all_r": lv["r"][m].mean()})
    return pd.DataFrame(rows)


def sharpe(x: np.ndarray) -> float:
    return x.mean() / x.std() * np.sqrt(365)


def reentry_table(lv: pd.DataFrame, interval: str, source: str, days_list=(1, 3, 7)) -> pd.DataFrame:
    lv = lv.sort_values(["coin", "time"]).reset_index(drop=True)
    g = lv.groupby("coin")
    idx = pd.date_range(lv["time"].min().floor("D"), lv["exit_t"].max().floor("D"), freq="D")

    def daily(mask):
        s = pd.Series(0.0025 * lv["r"].to_numpy()[mask], index=lv["exit_t"][mask].dt.floor("D")).groupby(level=0).sum()
        return s.reindex(idx, fill_value=0).to_numpy()

    key = cluster_key(lv)
    a = daily(np.ones(len(lv), bool))
    rows = []
    rng = np.random.default_rng(0)
    for days in days_list:
        flag = (g["loss"].shift(1).fillna(False).astype(bool) & (g["side"].shift(1) == lv["side"])
                & ((lv["time"] - g["exit_t"].shift(1)) < pd.Timedelta(days=days))).to_numpy()
        b = daily(~flag)
        cuts = [("2020-23", None, SPLIT), ("2024+", SPLIT, None)] if source == "binance" else \
            [("2023-24", None, pd.Timestamp("2025-01-01", tz="UTC")), ("2025+", pd.Timestamp("2025-01-01", tz="UTC"), None)]
        for lab, s, e in cuts:
            dm = np.ones(len(idx), bool)
            tm = np.ones(len(lv), bool)
            if s is not None:
                dm &= idx >= s
                tm &= (lv["time"] >= s).to_numpy()
            if e is not None:
                dm &= idx < e
                tm &= (lv["time"] < e).to_numpy()
            if (tm & flag).sum() < 20:
                continue
            A, B = a[dm], b[dm]
            n, nb = len(A), len(A) // 30
            boots = []
            for _ in range(2000):
                st = rng.integers(0, n - 30, nb)
                ii = (st[:, None] + np.arange(30)).ravel()
                boots.append(sharpe(B[ii]) - sharpe(A[ii]))
            boots = np.array(boots)
            mean, se, *_ = cluster_stats(lv["r"].to_numpy()[tm & flag], key[tm & flag])
            rows.append({"interval": interval, "source": source, "days": days, "period": lab,
                         "skipped": int((tm & flag).sum()), "skipped_r": mean, "skipped_se_cluster": se,
                         "skipped_stop_rate": lv["loss"][tm & flag].mean(), "others_r": lv["r"][tm & ~flag].mean(),
                         "sharpe_live": sharpe(A), "sharpe_skip": sharpe(B), "diff": sharpe(B) - sharpe(A),
                         "ci05": np.quantile(boots, 0.05), "ci95": np.quantile(boots, 0.95), "p_gt0": (boots > 0).mean()})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    for s in ("binance", "pionex"):
        for iv in ("4h", "2h"):
            ap.add_argument(f"--{s}-{iv}")
    ap.add_argument("--out", default="output")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    re_rows = []
    for iv in ("4H", "2H"):
        aucs, filt = [], []
        for src in ("binance", "pionex"):
            path = getattr(args, f"{src}_{iv.lower()}")
            if not path:
                continue
            lv = load_live(path, iv)
            aucs.append(auc_table(lv).assign(source=src))
            filt.append(filter_table(lv, iv, src))
            re_rows.append(reentry_table(lv, iv, src))
        if aucs:
            pd.concat(aucs).to_csv(out / f"stoploss_auc_{iv}.csv", index=False)
            pd.concat(filt).to_csv(out / f"stoploss_filters_{iv}.csv", index=False)
    if re_rows:
        pd.concat(re_rows).to_csv(out / "stoploss_reentry.csv", index=False)


if __name__ == "__main__":
    main()
