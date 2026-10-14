"""讀 alt_*_trades.csv.gz，做選參數 / 驗證 / 反向切分 / 隨機對照：python alt_report.py FILE [FILE...] [--out output]

選法：在選參數那段挑 t 值（期望值 / 標準誤）最高、且筆數 >= MIN_N 的設定，再看另一段的成績。
正向：2020-23 選、2024-26 驗證；反向：2024-26 選、2020-23 驗證。
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from grid_trading.altsim import TRAIN, VALID, load_bars, random_null, stats, window

warnings.filterwarnings("ignore")
MIN_N = 150
KEYS = ["n", "win", "exp_r", "se", "exp_r_slip", "cagr_pct", "max_dd_pct", "sharpe", "per_day"]


def per_cfg(t: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cfg, g in t.groupby("cfg"):
        row = {"cfg": cfg}
        for name, w in (("train", TRAIN), ("valid", VALID)):
            s = stats(window(g, w))
            row.update({f"{name}_{k}": s.get(k, np.nan) for k in KEYS + ["t"]})
        s = stats(g)
        row.update({f"all_{k}": s.get(k, np.nan) for k in KEYS})
        rows.append(row)
    return pd.DataFrame(rows)


def pick(df: pd.DataFrame, sel: str, other: str):
    c = df[df[f"{sel}_n"] >= MIN_N].sort_values(f"{sel}_t", ascending=False)
    return c.iloc[0] if len(c) else None


def fmt(row, name):
    g = lambda k: row[f"{name}_{k}"]
    return (f"n={int(g('n'))} 勝率{g('win') * 100:.0f}% 期望{g('exp_r'):+.3f}R(SE{g('se'):.3f}) 含滑價{g('exp_r_slip'):+.3f}R "
            f"年化{g('cagr_pct'):+.0f}% 回撤{g('max_dd_pct'):.0f}% 夏普{g('sharpe'):.2f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--out", default="output")
    ap.add_argument("--null", action="store_true", help="對選中的設定做隨機進場對照（最耗時）")
    a = ap.parse_args()
    for f in a.files:
        t = pd.read_csv(f, parse_dates=["time", "exit_time"])
        tag = Path(f).name.replace("_trades.csv.gz", "")
        df = per_cfg(t)
        df.to_csv(Path(a.out) / f"{tag}_configs.csv", index=False)
        print(f"\n===== {tag}: {len(df)} 個設定, {len(t)} 筆 =====")
        ok = df[(df.train_n >= MIN_N) & (df.valid_n >= MIN_N)]
        print(f"兩段筆數都 >= {MIN_N} 的設定 {len(ok)} 個；期望值 > 0 的：選參數段 {int((ok.train_exp_r > 0).sum())}，"
              f"驗證段 {int((ok.valid_exp_r > 0).sum())}，兩段都 > 0 {int(((ok.train_exp_r > 0) & (ok.valid_exp_r > 0)).sum())}；"
              f"含滑價兩段都 > 0 {int(((ok.train_exp_r_slip > 0) & (ok.valid_exp_r_slip > 0)).sum())}")
        print(f"全部設定平均：選參數段 {ok.train_exp_r.mean():+.3f}R，驗證段 {ok.valid_exp_r.mean():+.3f}R")
        for sel, other, label in (("train", "valid", "正向（2020-23 選, 2024-26 驗證）"),
                                  ("valid", "train", "反向（2024-26 選, 2020-23 驗證）")):
            r = pick(df, sel, other)
            if r is None:
                print(label, "沒有筆數足夠的設定")
                continue
            print(f"\n{label} 選中：{r['cfg']}")
            print("  選參數段:", fmt(r, sel))
            print("  驗證段  :", fmt(r, other))
            if a.null:
                g = t[t.cfg == r["cfg"]]
                print("  （隨機對照見 null 區塊）")
        # 選參數段排名前 5，看是不是集中在某個參數
        top = df[df.train_n >= MIN_N].sort_values("train_t", ascending=False).head(5)
        print("\n選參數段 t 值前 5 名的驗證成績：")
        for _, r in top.iterrows():
            print(f"  {r['cfg']}: 選 {r.train_exp_r:+.3f}R(n={int(r.train_n)}) → 驗 {r.valid_exp_r:+.3f}R(n={int(r.valid_n)}) 夏普 {r.valid_sharpe:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
