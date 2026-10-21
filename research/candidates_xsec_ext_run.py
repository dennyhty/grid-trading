"""候選 B 延伸版的跑法：現貨補長到 2017-09 + 三種對沖，預先登記 8 組，只用新的樣本內挑主設定，過驗證關卡。

python -m research.candidates_xsec_ext_run --is-only   # 開發／除錯用：只印樣本內（2017-09 ~ 2023-12），不看樣本外
python -m research.candidates_xsec_ext_run --log       # 正式跑一次：先把 8 組的樣本內寫進帳本，再挑主設定、看樣本外、過關卡

流程（跑之前就固定，見 research/candidates_xsec_ext.py 的說明）：
1. 8 組設定都跑 2017-09 ~ 2026-09。樣本內 = 2017-09 ~ 2023-12（新）；樣本外 = 2024-01 起（不變）；
   另列舊樣本內 2020-23（只算合約期的列，btc 對沖 combo q0.3 應該等於 PR #53 的 +0.18）。
2. 主設定 = 樣本內扣成本日報酬夏普最高的一組；另外每種對沖各取樣本內最好的一組，一起過關卡（都已算在 8 組裡，
   不是新的試驗）。
3. 關卡：research.validation_gate.evaluate，和 candidates_run 的 B 一樣用「日報酬版」：逐 (商品, 日) 的損益列，
   side 全設成 +1，讓分群以日／週／月為單位；代號去掉現貨前綴（XRP 現貨期與合約期算同一個幣）。
   基準 = 實盤維加斯時點帳本（2020 起）；隨機基準 = 同一批合格幣隨機分多空、同樣權重與對沖（3 次）；
   試驗數 = 帳本既有設定數（不含 xsec_ext）+ 8。
4. 和維加斯比較只看 2020 起（維加斯帳本只有 2020 起）。

輸出：reports/signals/xsec_ext_*_20261020.csv、output/candidates_ext/（明細 pkl、關卡 md）。
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import research.candidates_xsec as fb
import research.candidates_xsec_ext as fx
from grid_trading.spot_pit_universe import load_spot_daily, monthly_table, spot_membership
from research.candidates_common import compare_with, daily_series, load_daily, pool_table, vegas_book
from research.experiments import log_experiment, read_ledger
from research.validation_gate import GateConfig, evaluate, report_md

OUT = Path("output/candidates_ext")
REPORT_DIR = Path("reports/signals")
TAG = "20261020"
REPORT = f"signals/report_xsec_ext_{TAG}.md"
DATA_SOURCE = "binance spot 1d PIT top100 2017-09..2019-12 + um 4h->1d PIT top100 2020-01+"


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _r(x, k=4):
    return round(float(x), k) if x is not None and np.isfinite(x) else ""


def base_trials() -> int:
    """帳本裡 xsec_ext 以外的設定數（重跑時不重複算這個家族）。"""
    total = 0
    for r in read_ledger():
        if r["family"] == fx.FAMILY:
            continue
        m = re.search(r"configs=(\d+)", r.get("notes") or "")
        total += int(m.group(1)) if m else 1
    return total


# ---------------------------------------------------------------- 資料

def build():
    data = load_daily()
    fut = fb.build_panel(data)
    mem = pool_table()
    spot = load_spot_daily()
    table = monthly_table(spot)
    smem = spot_membership(table)                       # 交易用：2017-09 ~ 2019-12
    smem_idx = spot_membership(table, last="2020-01")   # 估指數 beta 用：補到 2020-01
    fut_pool = set(zip(mem["symbol"], mem["month"]))
    trade_pool = {(fx.SPOT_PREFIX + s, m) for s, m in zip(smem["symbol"], smem["month"])} \
        | {(s, m) for s, m in fut_pool if m >= fx.SPLICE_MONTH}
    index_pool = {(fx.SPOT_PREFIX + s, m) for s, m in zip(smem_idx["symbol"], smem_idx["month"])} \
        | {(s, m) for s, m in fut_pool if m > "2020-01"}
    panel = fx.build_ext_panel(fut, spot, trade_pool, index_pool, cache=True)
    return {"data": data, "fut": fut, "fut_pool": fut_pool, "panel": panel, "pool": trade_pool, "smem": smem}


def reproduce_check(env) -> float:
    """btc 對沖 combo q0.3 的合約期列要和 PR #53（candidates_xsec.run）相同。回傳 r_net 最大差。"""
    cfg = {"signal": "combo", "q": 0.3}
    orig = fb.run(env["fut"], env["fut_pool"], cfg)
    ext = fx.run_ext(env["panel"], env["pool"], {**cfg, "hedge": "btc"})
    ext = ext[(ext["era"] == "fut") & (ext["time"] >= fx.OLD_IS_START)]
    m = orig.merge(ext, on=["symbol", "time"], how="outer", suffixes=("_o", "_e"), indicator=True)
    if (m["_merge"] != "both").any():
        return float("inf")
    return float((m["r_net_o"] - m["r_net_e"]).abs().max())


# ---------------------------------------------------------------- 統計

def windows(rows: pd.DataFrame, is_only: bool) -> dict[str, pd.DataFrame]:
    t = rows["time"]
    w = {"is": rows[t < fx.IS_END],
         "old_is": rows[(rows["era"] == "fut") & (t < fx.IS_END)],
         "spot_era": rows[rows["era"] == "spot"]}
    if not is_only:
        w.update({"oos": rows[t >= fx.IS_END], "fut_era": rows[rows["era"] == "fut"], "all": rows})
    return w


def legs(sub: pd.DataFrame) -> dict:
    """多腿、空腿、對沖各自的扣成本 R 合計，成本與資金費率合計，等效交易（幣 x 週）數。"""
    nh = sub[~sub["hedge"]]
    pw = nh.groupby(["symbol", "period"]).size()
    return {"long_r": float(nh.loc[nh["side"] > 0, "r_net"].sum()), "short_r": float(nh.loc[nh["side"] < 0, "r_net"].sum()),
            "hedge_r": float(sub.loc[sub["hedge"], "r_net"].sum()), "cost_r": float(sub["cost_r"].sum()),
            "funding_r": float(sub["funding_r"].sum()), "position_weeks": int(len(pw))}


def row_stats(name: str, cfg: dict, rows: pd.DataFrame, is_only: bool) -> dict:
    rec = {"config": name, **cfg}
    for lab, sub in windows(rows, is_only).items():
        if sub.empty:
            continue
        day = fx.daily_ret(sub)
        rec[f"{lab}_start"] = str(day.index.min().date())
        for k, v in fx.perf(day).items():
            rec[f"{lab}_{k}"] = v
        rec[f"{lab}_sharpe_gross"] = fx.perf(fx.daily_ret(sub, "r"))["sharpe"]
        for k, v in legs(sub).items():
            rec[f"{lab}_{k}"] = v
    return rec


def drop_top_coins(rows: pd.DataFrame, k: int = 5) -> tuple[list, pd.DataFrame]:
    """拿掉扣成本總 R 最高的 k 個幣（非對沖、現貨與合約合併）的所有列（對沖列保留）。"""
    nh = rows[~rows["hedge"]]
    top = nh.groupby(nh["symbol"].map(fx.coin_key))["r_net"].sum().nlargest(k).index.tolist()
    keep = rows["hedge"] | ~rows["symbol"].map(fx.coin_key).isin(top)
    return top, rows[keep]


def yearly(rows: pd.DataFrame, name: str) -> pd.DataFrame:
    day = fx.daily_ret(rows)
    g = day.groupby(day.index.year)
    out = []
    for y, d in g:
        sub = rows[rows["time"].dt.year == y]
        eq = np.cumprod(1 + d.to_numpy())
        out.append({"config": name, "year": int(y), "days": len(d), "ret_pct": float(d.sum() * 100),
                    "sharpe": float(d.mean() / d.std() * math.sqrt(365)) if d.std() > 0 else float("nan"),
                    "max_dd_pct": float((eq / np.maximum.accumulate(eq) - 1).min() * 100), **legs(sub)})
    return pd.DataFrame(out)


def gate_rows(df: pd.DataFrame) -> pd.DataFrame:
    """日報酬版關卡的輸入：side 全設 +1（組合當一個單位），代號去掉現貨前綴。"""
    return df.assign(side=1, symbol=df["symbol"].map(fx.coin_key))


# ---------------------------------------------------------------- 主程式

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", action="store_true", help="寫進 research/experiments.csv")
    ap.add_argument("--is-only", action="store_true", help="只算樣本內，不看樣本外、不過關卡")
    ap.add_argument("--draws", type=int, default=3)
    a = ap.parse_args()
    if a.log and a.is_only:
        raise SystemExit("--log 要正式跑（不能和 --is-only 一起用）")
    OUT.mkdir(parents=True, exist_ok=True)
    env = build()
    p, pool = env["panel"], env["pool"]
    diff = reproduce_check(env)
    _log(f"PR #53 重現檢查（btc combo q0.3 合約期 r_net 最大差）：{diff:.2e}")
    if not diff < 1e-9:
        raise SystemExit("合約期和 PR #53 不一致，先修好再跑")

    cfgs = fx.configs()
    rows_by, table = {}, []
    for name, cfg in cfgs.items():
        rows = fx.run_ext(p, pool, cfg)
        rows_by[name] = rows
        rec = row_stats(name, cfg, rows, a.is_only)
        table.append(rec)
        _log(f"[{fx.FAMILY}] {name}: 樣本內(2017-23) 夏普 {rec['is_sharpe']:+.2f}，舊樣本內(2020-23) "
             f"{rec.get('old_is_sharpe', float('nan')):+.2f}，現貨期 {rec.get('spot_era_sharpe', float('nan')):+.2f}")
    table = pd.DataFrame(table)
    head = str(table.sort_values("is_sharpe", ascending=False)["config"].iloc[0])
    per_hedge = {h: str(table[table["hedge"] == h].sort_values("is_sharpe", ascending=False)["config"].iloc[0])
                 for h in fx.HEDGES}
    _log(f"主設定（只看新樣本內）：{head}；各對沖最好：{per_hedge}")
    if a.is_only:
        cols = [c for c in table.columns if c.startswith(("config", "is_", "old_is_", "spot_era_"))]
        print(table[cols].to_string())
        return

    n_new = len(cfgs)
    n_trials = base_trials() + n_new
    _log(f"試驗數 = {n_trials}（帳本既有 {n_trials - n_new} + 這次 {n_new}）")
    # ---- 1. 先把 8 組的樣本內寫進帳本（挑主設定前就固定）
    if a.log:
        for _, row in table.iterrows():
            log_experiment(family=fx.FAMILY, hypothesis=fx.__doc__.splitlines()[0], params=cfgs[row["config"]],
                           data_source=DATA_SOURCE, period="2017-09..2023-12 IS", trades=int(row["is_position_weeks"]),
                           t_stat=_r(row["is_t_weekly"], 3), sharpe=_r(row["is_sharpe"], 3), verdict="info",
                           report=REPORT,
                           notes=f"IS screen of pre-registered grid ({n_new}); daily-return Sharpe; t=weekly; "
                                 f"old IS 2020-23 sharpe {row['old_is_sharpe']:.3f}; configs=1")

    # ---- 2. 關卡、和維加斯比較、集中度、逐年
    base = vegas_book()
    base_day = daily_series(base)
    btc = pd.Series(p.close_ff[:, p.syms.index(fx.SPOT_HEDGE)], index=p.days).pct_change()
    btc_f = pd.Series(p.close_ff[:, p.syms.index(fb.HEDGE)], index=p.days).pct_change()
    btc = btc.where(p.days < fx.OLD_IS_START, btc_f).fillna(0.0)
    gated = list(dict.fromkeys([head, *per_hedge.values()]))
    gate_out, corr_out, year_out, coin_out, extra = [], [], [], [], {}
    for name in gated:
        cfg = cfgs[name]
        tr = rows_by[name]
        rnd = pd.concat([fx.run_ext(p, pool, cfg, rng=np.random.default_rng(100 + k)) for k in range(a.draws)],
                        ignore_index=True)
        nbrs = [rows_by[n] for n in fx.neighbors(name)]
        g = evaluate(gate_rows(tr), baseline=base, random_baseline=gate_rows(rnd),
                     cfg=GateConfig(n_trials=n_trials), rcol="r_net", neighbors=[gate_rows(n) for n in nbrs])
        (OUT / f"gate_{name.replace('|', '_')}.md").write_text(report_md(g, f"{fx.FAMILY} {name}"), encoding="utf-8")
        for k, c in g["checks"].items():
            gate_out.append({"config": name, "headline": name == head, "check": k, "label": c["label"],
                             "status": c["status"], "value": json.dumps(c["value"], default=float, ensure_ascii=False),
                             "threshold": c["threshold"], "detail": c["detail"]})
        cand_day = fx.daily_ret(tr)
        cmp_all = compare_with(base_day, cand_day)            # 重疊期間 = 2020 起
        oos_b, oos_c = base_day[base_day.index >= fx.IS_END], cand_day[cand_day.index >= fx.IS_END]
        cmp_oos = compare_with(oos_b, oos_c)
        top, rest = drop_top_coins(tr)
        rest_day = fx.daily_ret(rest)
        rec = {"config": name, "headline": name == head, **{f"vs_vegas_2020_{k}": v for k, v in cmp_all.items()},
               **{f"vs_vegas_oos_{k}": v for k, v in cmp_oos.items()},
               "corr_btc_daily_all": float(cand_day.corr(btc.reindex(cand_day.index).fillna(0.0))),
               "top5_coins": ",".join(top), "drop_top5_sharpe_all": fx.perf(rest_day)["sharpe"],
               "drop_top5_sharpe_is": fx.perf(rest_day[rest_day.index < fx.IS_END])["sharpe"],
               "drop_top5_sharpe_oos": fx.perf(rest_day[rest_day.index >= fx.IS_END])["sharpe"],
               "gate_verdict": g["verdict"], "gate_passed": g["passed"],
               "strict_t": g["checks"]["cluster_t"]["value"], "required_t": g["checks"]["cluster_t"]["threshold"],
               "dsr": g["checks"]["deflated_sharpe"]["value"], "beats_random_z": g["checks"]["beats_random"]["value"],
               "n_trials": n_trials}
        # 現貨期手續費敏感度（當時現貨 0.1%）：只影響現貨期
        hi_fee = fx.run_ext(p, pool, cfg, spot_fee=0.001)
        rec["spot_era_sharpe_fee10bp"] = fx.perf(fx.daily_ret(hi_fee[hi_fee["era"] == "spot"]))["sharpe"]
        rec["is_sharpe_fee10bp_spot"] = fx.perf(fx.daily_ret(hi_fee[hi_fee["time"] < fx.IS_END]))["sharpe"]
        corr_out.append(rec)
        year_out.append(yearly(tr, name))
        nh = tr[~tr["hedge"]]
        c = nh.groupby([nh["symbol"].map(fx.coin_key), "era"])["r_net"].sum().unstack(fill_value=0.0)
        c["total"] = c.sum(axis=1)
        c = c.sort_values("total", ascending=False)
        coin_out.append(pd.concat([c.head(10), c.tail(5)]).reset_index().rename(columns={"symbol": "coin"})
                        .assign(config=name))
        extra[name] = {"verdict": g["verdict"], "info": g["info"], "compare": cmp_all, "compare_oos": cmp_oos}
        with (OUT / f"{name.replace('|', '_')}.pkl").open("wb") as f:
            pickle.dump({"rows": tr, "random": rnd, "gate": g}, f)
        if a.log:
            row = table.set_index("config").loc[name]
            if name == head:
                log_experiment(family=fx.FAMILY, hypothesis="headline OOS (selected on 2017-23 IS only)", params=cfg,
                               data_source=DATA_SOURCE, period="2024-2026 OOS", trades=int(row["oos_position_weeks"]),
                               t_stat=_r(row["oos_t_weekly"], 3), sharpe=_r(row["oos_sharpe"], 3), verdict="info",
                               report=REPORT, notes="same config as IS row, not a new trial; configs=0")
            log_experiment(family=fx.FAMILY, hypothesis="validation_gate (daily-return rows, baseline=vegas PIT book)",
                           params={**cfg, "n_trials": n_trials, "headline": name == head}, data_source=DATA_SOURCE,
                           period="2017-09..2026-09", trades=g["info"]["trades"], exp_r=_r(g["info"]["mean_r"]),
                           t_stat=_r(g["checks"]["cluster_t"]["value"], 3), sharpe=_r(g["info"].get("sharpe_annual"), 3),
                           verdict="fail" if not g["passed"] else ("info" if g["incomplete"] else "pass"),
                           report=REPORT, notes=f"{g['verdict']}; corr_daily_2020+={cmp_all['corr_daily']:.3f}; configs=0")
        _log(f"[{fx.FAMILY}] {name}: {g['verdict']}")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    table.assign(headline=table["config"] == head).to_csv(REPORT_DIR / f"xsec_ext_grid_{TAG}.csv", index=False)
    pd.DataFrame(gate_out).to_csv(REPORT_DIR / f"xsec_ext_gate_{TAG}.csv", index=False)
    pd.DataFrame(corr_out).to_csv(REPORT_DIR / f"xsec_ext_corr_{TAG}.csv", index=False)
    pd.concat(year_out, ignore_index=True).to_csv(REPORT_DIR / f"xsec_ext_yearly_{TAG}.csv", index=False)
    pd.concat(coin_out, ignore_index=True).to_csv(REPORT_DIR / f"xsec_ext_coins_{TAG}.csv", index=False)
    pool_n = env["smem"].groupby("month").size().rename("spot_pool_n").reset_index()
    pool_n.to_csv(REPORT_DIR / f"xsec_ext_spot_pool_{TAG}.csv", index=False)
    summary = {"n_trials": n_trials, "headline": head, "per_hedge_best": per_hedge, "reproduce_diff": diff, **extra}
    with (OUT / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1, default=str)
    print(json.dumps(summary, ensure_ascii=False, indent=1, default=str))


if __name__ == "__main__":
    main()
