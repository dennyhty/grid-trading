"""階段 2 候選策略：跑預先登記的網格、只用樣本內挑主設定、樣本外看一次、過驗證關卡、和實盤維加斯比較。

python -m research.candidates_run            # 只算，不寫實驗帳本
python -m research.candidates_run --log      # 同時把每組設定（和主設定的樣本外、關卡結果）寫進 research/experiments.csv

候選（程式見各模組的說明）：
- A research/candidates_daily_trend.py：日線唐奇安突破（12 組）
- B research/candidates_xsec.py：每週橫斷面動能／低波動多空組合，beta 中性（10 組）
- C research/candidates_squeeze.py：日線波動壓縮後突破（8 組）

預先登記的流程（跑之前就固定）：
1. 每組設定都跑完整期間，但挑主設定只看樣本內（進場 2020-01 ~ 2023-12）：
   A、C 用樣本內扣成本 R 的群集 t（進場日+方向）最高、筆數 >= 100；B 用樣本內扣成本的日報酬夏普最高。
2. 主設定的 2024 年起是樣本外，只看這一次；其他設定的樣本外也列出來，但不拿來挑。
3. 主設定過 research.validation_gate.evaluate：基準 = 實盤維加斯時點帳本（相關性）、鄰近參數（高原）、
   同出場的隨機進場（A、C：同幣同年同方向隨機一天；B：同樣的合格名單隨機分多空）。
   試驗數 = 帳本既有的設定數 + 這次 30 組（不管有沒有 --log 都用同一個數字）。
4. B 是組合策略：關卡吃的是逐 (幣, 日) 的損益列（r x 0.25% 加總 = 組合日報酬），即「日報酬版」關卡；
   另外列出逐 (幣, 週) 的「等效交易」統計（1R = 一週一個標準差）。

輸出：output/candidates/*.pkl（明細）、reports/signals/candidates_*_20261020.csv、output/candidates/gate_*.md。
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import research.candidates_daily_trend as fam_a
import research.candidates_squeeze as fam_c
import research.candidates_xsec as fam_b
from research.candidates_common import (IS_END, RISK, compare_with, daily_series, finalize, load_daily, pick_headline,
                                        pool_table, random_entries, split, stats, vegas_book)
from research.experiments import log_experiment, read_ledger
from research.validation_gate import GateConfig, evaluate, report_md

OUT = Path("output/candidates")
REPORT_DIR = Path("reports/signals")
TAG = "20261020"
REPORT = f"signals/report_candidates_{TAG}.md"
N_NEW = len(fam_a.configs()) + len(fam_b.configs()) + len(fam_c.configs())
DATA_SOURCE = "binance um 4h->1d, PIT top100 by prior-month volume"


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def base_trials() -> int:
    """帳本裡「這次以外」的設定數（排除 cand_ 開頭的家族，重跑時才不會重複算）。"""
    import re
    total = 0
    for r in read_ledger():
        if r["family"].startswith("cand_"):
            continue
        m = re.search(r"configs=(\d+)", r.get("notes") or "")
        total += int(m.group(1)) if m else 1
    return total


def _r(x, k=4):
    return round(float(x), k) if x is not None and np.isfinite(x) else ""


# ---------------------------------------------------------------- A、C：逐筆交易的家族

def trade_family(mod, data, mem) -> dict:
    trades = {}
    for name, p in mod.configs().items():
        t0 = time.time()
        trades[name] = finalize(mod.raw_trades(data, p), mem)
        _log(f"[{mod.FAMILY}] {name}: {len(trades[name])} 筆，{time.time() - t0:.0f}s")
    rows = []
    for name, tr in trades.items():
        ins, oos = split(tr)
        rows.append({"family": mod.FAMILY, "config": name, **{f"is_{k}": v for k, v in stats(ins).items()},
                     **{f"oos_{k}": v for k, v in stats(oos).items()}, **{f"all_{k}": v for k, v in stats(tr).items()}})
    table = pd.DataFrame(rows)
    head = pick_headline(table, 100, "is_t")
    return {"trades": trades, "table": table, "headline": head}


# ---------------------------------------------------------------- B：組合

def _weekly_t(day: pd.Series) -> float:
    w = day.resample("W").sum()
    return float(w.mean() / w.std() * math.sqrt(len(w))) if w.std() > 0 else float("nan")


def xsec_family(data, mem) -> dict:
    panel = fam_b.build_panel(data)
    pool = set(zip(mem["symbol"], mem["month"]))
    rows_by, table = {}, []
    for name, p in fam_b.configs().items():
        rows = fam_b.run(panel, pool, p)
        rows_by[name] = rows
        rec = {"family": fam_b.FAMILY, "config": name}
        for lab, m in (("is", rows["time"] < IS_END), ("oos", rows["time"] >= IS_END), ("all", slice(None))):
            sub = rows[m]
            day = fam_b.daily(sub)
            pw = fam_b.position_weeks(sub)
            rec.update({f"{lab}_sharpe": fam_b.sharpe(day), f"{lab}_ann_ret_pct": day.mean() * 365 * 100,
                        f"{lab}_ann_vol_pct": day.std() * math.sqrt(365) * 100, f"{lab}_t_weekly": _weekly_t(day),
                        f"{lab}_sharpe_gross": fam_b.sharpe(fam_b.daily(sub, "r")),
                        f"{lab}_trades": len(pw), f"{lab}_win_rate": float((pw["r_net"] > 0).mean()),
                        f"{lab}_exp_r": float(pw["r_net"].mean()),
                        f"{lab}_long_r": float(pw.loc[pw["side"] > 0, "r_net"].mean()),
                        f"{lab}_short_r": float(pw.loc[pw["side"] < 0, "r_net"].mean()),
                        f"{lab}_cost_r_per_week": float(sub["cost_r"].sum() / max(len(pw), 1)),
                        f"{lab}_funding_r_per_week": float(sub["funding_r"].sum() / max(len(pw), 1))})
        table.append(rec)
        _log(f"[{fam_b.FAMILY}] {name}: IS 夏普 {rec['is_sharpe']:.2f}，樣本外 {rec['oos_sharpe']:.2f}")
    table = pd.DataFrame(table)
    head = str(table.sort_values("is_sharpe", ascending=False)["config"].iloc[0])
    return {"trades": rows_by, "table": table, "headline": head, "panel": panel, "pool": pool}


# ---------------------------------------------------------------- 關卡與比較

def btc_daily(data) -> pd.Series:
    d = data["BTCUSDT"].set_index("time")["close"]
    return d.pct_change().dropna()


def yearly(tr: pd.DataFrame, family: str, rcol: str = "r_net") -> pd.DataFrame:
    rows = []
    for y, g in tr.groupby(tr["time"].dt.year):
        s = stats(g, rcol)
        rows.append({"family": family, "year": int(y), **s})
    return pd.DataFrame(rows)


def gate_one(family: str, tr: pd.DataFrame, base: pd.DataFrame, rnd: pd.DataFrame, nbrs: list, n_trials: int) -> dict:
    res = evaluate(tr, baseline=base, random_baseline=rnd, cfg=GateConfig(n_trials=n_trials), rcol="r_net",
                   neighbors=nbrs)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"gate_{family}.md").write_text(report_md(res, family), encoding="utf-8")
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", action="store_true", help="寫進 research/experiments.csv")
    ap.add_argument("--draws", type=int, default=3, help="隨機進場基準重複幾次")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    data = load_daily()
    mem = pool_table()
    n_trials = base_trials() + N_NEW
    _log(f"試驗數 = {n_trials}（帳本既有 {n_trials - N_NEW} + 這次 {N_NEW}）")

    base = vegas_book()
    base_day = daily_series(base)
    btc = btc_daily(data)

    fams = {"A": (fam_a, trade_family(fam_a, data, mem)), "C": (fam_c, trade_family(fam_c, data, mem)),
            "B": (fam_b, xsec_family(data, mem))}

    # ---- 1. 先寫每組設定的樣本內結果（挑主設定之前就固定）
    if a.log:
        for key, (mod, res) in fams.items():
            for _, row in res["table"].iterrows():
                p = mod.configs()[row["config"]]
                if key == "B":
                    log_experiment(family=mod.FAMILY, hypothesis=mod.__doc__.splitlines()[0], params=p,
                                   data_source=DATA_SOURCE, period="2020-2023 IS", trades=int(row["is_trades"]),
                                   exp_r=_r(row["is_exp_r"]), t_stat=_r(row["is_t_weekly"], 3),
                                   sharpe=_r(row["is_sharpe"], 3), verdict="info", report=REPORT,
                                   notes=f"IS screen of pre-registered grid ({len(mod.configs())}); trades=position-weeks, "
                                         f"exp_r per position-week (1R=1 weekly sd), t=weekly returns; configs=1")
                else:
                    log_experiment(family=mod.FAMILY, hypothesis=mod.__doc__.splitlines()[0], params=p,
                                   data_source=DATA_SOURCE, period="2020-2023 IS", trades=int(row["is_trades"]),
                                   exp_r=_r(row["is_exp_r"]), se_cluster=_r(row["is_se_cluster"]),
                                   t_stat=_r(row["is_t"], 3), sharpe=_r(row["is_sharpe"], 3), verdict="info",
                                   report=REPORT, notes=f"IS screen of pre-registered grid ({len(mod.configs())}), "
                                                        f"r_net, win {row['is_win_rate']:.2f}; configs=1")

    # ---- 2. 主設定：隨機基準、鄰近、關卡、和維加斯比較
    gate_rows, corr_rows, year_rows, summary = [], [], [], {}
    for key, (mod, res) in fams.items():
        head = res["headline"]
        _log(f"[{mod.FAMILY}] 主設定（樣本內挑）：{head}")
        tr = res["trades"][head]
        nbrs = [res["trades"][n] for n in mod.neighbors(head)]
        if key == "B":
            rnd = pd.concat([fam_b.run(res["panel"], res["pool"], mod.configs()[head], rng=np.random.default_rng(100 + k))
                             for k in range(a.draws)], ignore_index=True)
            cand_day = fam_b.daily(tr)
            year_rows.append(fam_b.position_weeks(tr).assign(family=mod.FAMILY).pipe(
                lambda pw: pw.groupby(pw["time"].dt.year).agg(trades=("r_net", "size"), exp_r=("r_net", "mean"),
                                                              win_rate=("r_net", lambda x: (x > 0).mean()))
                .reset_index().rename(columns={"time": "year"}).assign(family=mod.FAMILY)))
            yd = cand_day.groupby(cand_day.index.year)
            year_rows[-1] = year_rows[-1].merge(
                pd.DataFrame({"year": yd.mean().index, "sharpe": (yd.mean() / yd.std() * math.sqrt(365)).values,
                              "ret_pct": (yd.sum() * 100).values}), on="year", how="left")
        else:
            rnd = random_entries(tr, data, mem, mod.plan_of_factory(data, mod.configs()[head]), draws=a.draws)
            cand_day = daily_series(tr)
            year_rows.append(yearly(tr, mod.FAMILY))
        if key == "B":
            # 組合是一個整體：多腿、空腿和 BTC 對沖同一天互相抵銷，若照方向分群會把負相關的兩群當成獨立，
            # 標準誤被灌大。日報酬版關卡把每一列的 side 設成 +1，讓分群以「日／週／月」為單位（多空拆分另外報告）。
            port = lambda x: x.assign(side=1)  # noqa: E731
            g = gate_one(mod.FAMILY, port(tr), base, port(rnd), [port(n) for n in nbrs], n_trials)
        else:
            g = gate_one(mod.FAMILY, tr, base, rnd, nbrs, n_trials)
        for k, c in g["checks"].items():
            gate_rows.append({"family": mod.FAMILY, "headline": head, "check": k, "label": c["label"],
                              "status": c["status"], "value": json.dumps(c["value"], default=float, ensure_ascii=False),
                              "threshold": c["threshold"], "detail": c["detail"]})
        cmp_ = compare_with(base_day, cand_day)
        bt = btc.reindex(cand_day.index).fillna(0.0)
        cmp_["corr_btc_daily"] = float(cand_day.corr(bt))
        # 只看樣本外
        oos_base, oos_cand = base_day[base_day.index >= IS_END], cand_day[cand_day.index >= IS_END]
        cmp_oos = compare_with(oos_base, oos_cand)
        corr_rows.append({"family": mod.FAMILY, "headline": head, **cmp_,
                          **{f"oos_{k}": v for k, v in cmp_oos.items()}})
        summary[key] = {"family": mod.FAMILY, "headline": head, "verdict": g["verdict"], "passed": g["passed"],
                        "info": g["info"], "random": g["checks"]["beats_random"]["detail"],
                        "plateau": g["checks"]["plateau"]["detail"], "neighbors": mod.neighbors(head)}
        with (OUT / f"{key}_{mod.FAMILY}.pkl").open("wb") as f:
            pickle.dump({"table": res["table"], "headline": head, "trades": tr, "random": rnd,
                         "neighbors": {n: res["trades"][n] for n in mod.neighbors(head)}, "gate": g, "compare": cmp_},
                        f)
        if a.log:
            row = res["table"].set_index("config").loc[head]
            if key == "B":
                log_experiment(family=mod.FAMILY, hypothesis="headline OOS (selected on IS only)", params=mod.configs()[head],
                               data_source=DATA_SOURCE, period="2024-2026 OOS", trades=int(row["oos_trades"]),
                               exp_r=_r(row["oos_exp_r"]), t_stat=_r(row["oos_t_weekly"], 3),
                               sharpe=_r(row["oos_sharpe"], 3), verdict="info", report=REPORT,
                               notes="same config as IS row, not a new trial; configs=0")
            else:
                log_experiment(family=mod.FAMILY, hypothesis="headline OOS (selected on IS only)", params=mod.configs()[head],
                               data_source=DATA_SOURCE, period="2024-2026 OOS", trades=int(row["oos_trades"]),
                               exp_r=_r(row["oos_exp_r"]), se_cluster=_r(row["oos_se_cluster"]),
                               t_stat=_r(row["oos_t"], 3), sharpe=_r(row["oos_sharpe"], 3), verdict="info",
                               report=REPORT, notes=f"same config as IS row, win {row['oos_win_rate']:.2f}; configs=0")
            t_min = g["checks"]["cluster_t"]["value"]
            log_experiment(family=mod.FAMILY, hypothesis="validation_gate on headline (baseline=vegas PIT book)",
                           params={**mod.configs()[head], "n_trials": n_trials}, data_source=DATA_SOURCE,
                           period="2020-2026", trades=g["info"]["trades"], exp_r=_r(g["info"]["mean_r"]),
                           t_stat=_r(t_min, 3), sharpe=_r(g["info"].get("sharpe_annual"), 3),
                           verdict="fail" if not g["passed"] else ("info" if g["incomplete"] else "pass"),
                           report=REPORT, notes=f"{g['verdict']}; corr_daily={cmp_['corr_daily']:.3f}; configs=0")
        _log(f"[{mod.FAMILY}] {g['verdict']}；日相關 {cmp_['corr_daily']:+.3f}")

    grid = pd.concat([fams[k][1]["table"].assign(headline=lambda t, k=k: t["config"] == fams[k][1]["headline"])
                      for k in ("A", "B", "C")], ignore_index=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    grid.to_csv(REPORT_DIR / f"candidates_grid_{TAG}.csv", index=False)
    pd.DataFrame(gate_rows).to_csv(REPORT_DIR / f"candidates_gate_{TAG}.csv", index=False)
    pd.DataFrame(corr_rows).to_csv(REPORT_DIR / f"candidates_corr_{TAG}.csv", index=False)
    pd.concat(year_rows, ignore_index=True).to_csv(REPORT_DIR / f"candidates_yearly_{TAG}.csv", index=False)
    with (OUT / "summary.json").open("w", encoding="utf-8") as f:
        json.dump({"n_trials": n_trials, **summary}, f, ensure_ascii=False, indent=1, default=str)
    print(json.dumps({"n_trials": n_trials, **summary}, ensure_ascii=False, indent=1, default=str))


if __name__ == "__main__":
    main()
