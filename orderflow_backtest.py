"""訂單流策略回測：python orderflow_backtest.py [--tf 4h] [--out output/orderflow]

資料：幣安 USDT 永續 1H 月檔（含主動買入量）＋每日 metrics（持倉量、多空比），先用
grid_trading.binance_history.fetch_binance(coin, "1h", flow=True) 和 grid_trading.binance_metrics.fetch_metrics 下載。
名單是 universe.TOP100 裡幣安有合約的幣。出場和 vegas_trend 相同（1.5 ATR 停損、TP 1/2/3R、40 天時間出場）。

輸出：
- orderflow_{tf}_grid.csv：每組參數的筆數、期望值，分 2020-2023（選參數）和 2024 之後（驗證）
- orderflow_{tf}_trades.csv.gz：每一筆交易（含手續費 0.05% 和多加 0.05% 滑價兩種 R）
- orderflow_{tf}_vegas_filters.csv：用訂單流過濾 vegas_trend（收緊 RSI）訊號，留下 / 剔除的成績
- orderflow_{tf}_random.csv：入選參數和隨機進場（同幣、同季、同方向、同出場、同筆數）的比較
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from grid_trading import signals
from grid_trading.orderflow import add_features, flow_bars
from grid_trading.signals import orderflow  # noqa: F401  登記訂單流策略
from grid_trading.signals.evaluate import trades_for
from grid_trading.signals.strategy import LONG, SHORT, Signal
from grid_trading.signals.tracker import track
from grid_trading.universe import TOP100

SPLIT = pd.Timestamp("2024-01-01", tz="UTC")
FEE = 0.0005
SLIP_FEE = 0.001  # 手續費 0.05% + 滑價 0.05%（每邊）
MAX_BARS = {"4h": 240, "2h": 480}  # 都是 40 天
RISK = 0.0025


def grid() -> dict[str, tuple[str, dict]]:
    out = {}
    for lb in (10, 20, 40):
        for mode in ("reversal", "trend"):
            out[f"cvd_div L{lb} {mode}"] = ("of_cvd_div", {"lookback": lb, "mode": mode})
    for n in (1, 3, 6):
        for k in (1.5, 2.0, 2.5):
            for mode in ("follow", "fade", "absorb"):
                for tf_ in (False, True):
                    out[f"taker n{n} k{k} {mode}{' +trend' if tf_ else ''}"] = (
                        "of_taker", {"n": n, "k": k, "mode": mode, "trend_filter": tf_})
    for n in (1, 3, 6):
        for k in (1.0, 1.5, 2.0):
            for mode in ("build", "build_fade", "unwind_fade", "unwind_follow"):
                out[f"oi n{n} k{k} {mode}"] = ("of_oi", {"n": n, "k": k, "mode": mode})
    for field in ("account_ratio", "top_position_ratio"):
        for k in (1.5, 2.0):
            for mode in ("fade", "follow"):
                for tf_ in (False, True):
                    out[f"crowd {field} k{k} {mode}{' +trend' if tf_ else ''}"] = (
                        "of_crowd", {"field": field, "k": k, "mode": mode, "trend_filter": tf_})
    return out


def family(label: str) -> str:
    return label.split(" ")[0]


def coins() -> list[str]:
    return [c for c in TOP100 if Path(f"data/binance/{c}_1h_flow.csv.gz").exists()]


def _bars(coin: str, tf: str) -> pd.DataFrame:
    df = flow_bars(coin, tf)
    return add_features(df) if len(df) > 250 else pd.DataFrame()


def run_coin(job) -> list[dict]:
    coin, tf, labels, fee_list = job
    df = _bars(coin, tf)
    if df.empty:
        return []
    configs = grid()
    sym = f"{coin}_USDT_PERP"
    rows = []
    for label in labels:
        name, params = configs[label]
        strat = signals.get(name, **params)
        base = trades_for(strat, df, sym, tf, FEE, MAX_BARS[tf])
        extra = {}
        if SLIP_FEE in fee_list:
            slip = trades_for(strat, df, sym, tf, SLIP_FEE, MAX_BARS[tf])
            extra = {(t["time"], t["side"]): t["r"] for t in slip}
        for t in base:
            t.pop("strict", None)
            rows.append(dict(t, coin=coin, variant=label, r_slip=extra.get((t["time"], t["side"]), np.nan)))
    return rows


def vegas_coin(job) -> list[dict]:
    """vegas_trend（收緊 RSI）的每一筆交易，附上訊號那根的訂單流欄位。"""
    coin, tf = job
    df = _bars(coin, tf)
    if df.empty:
        return []
    strat = signals.get("vegas_trend", rsi_long=(54, 70), rsi_short=(30, 46))
    sym = f"{coin}_USDT_PERP"
    base = trades_for(strat, df[["time", "open", "high", "low", "close", "volume"]], sym, tf, FEE, MAX_BARS[tf])
    slip = {(t["time"], t["side"]): t["r"] for t in
            trades_for(strat, df[["time", "open", "high", "low", "close", "volume"]], sym, tf, SLIP_FEE, MAX_BARS[tf])}
    bar = pd.Timedelta(tf)
    feat = df.set_index("time")
    cols = [c for c in df.columns if c.endswith("_z") or c.startswith("imb") or c in ("trend",)]
    rows = []
    for t in base:
        t.pop("strict", None)
        f = feat.loc[t["time"] - bar, cols].to_dict()
        rows.append(dict(t, coin=coin, variant="vegas_trend 收緊RSI", r_slip=slip.get((t["time"], t["side"]), np.nan), **f))
    return rows


def random_coin(job) -> list[dict]:
    """把每筆交易換成同幣、同季、同方向的隨機一根 K 線進場，出場規則不變。"""
    coin, tf, trades, reps, seed = job
    df = _bars(coin, tf)
    if df.empty or not trades:
        return []
    rng = np.random.default_rng(seed)
    bar = pd.Timedelta(tf)
    df = df[df.index >= 200].reset_index(drop=True)
    q = df["time"].dt.to_period("Q").astype(str).to_numpy()
    by_q = {k: np.flatnonzero(q == k) for k in np.unique(q)}
    out = []
    for rep in range(reps):
        for t in trades:
            idx = by_q.get(str(pd.Timestamp(t["time"] - bar).tz_localize(None).to_period("Q")))
            if idx is None or not len(idx):
                continue
            i = int(rng.choice(idx))
            r = df.iloc[i]
            risk = 1.5 * r.atr
            if not risk > 0:
                continue
            d = 1 if t["side"] == LONG else -1
            sig = Signal(coin, tf, t["side"], r.time + bar, float(r.close), float(r.close - d * risk),
                         [float(r.close + d * k * risk) for k in (1, 2, 3)], "random")
            res = track(sig, df, FEE, MAX_BARS[tf])
            if res.status == "CLOSED":
                out.append({"variant": t["variant"], "rep": rep, "coin": coin, "side": t["side"],
                            "time": sig.time, "exit_time": res.events[-1].time, "r": res.realized_r})
    return out


def stats(t: pd.DataFrame, col: str = "r") -> dict:
    if t.empty:
        return {"n": 0}
    r = t[col]
    return {"n": len(r), "exp": r.mean(), "se": r.std() / math.sqrt(len(r)) if len(r) > 1 else np.nan,
            "win": (r > 0).mean()}


def equity_stats(t: pd.DataFrame, col: str = "r") -> dict:
    """每筆風險 RISK，出場時記帳的資金曲線：年化、最大回撤、Sharpe（日報酬）。"""
    if t.empty:
        return {}
    t = t.sort_values("exit_time")
    eq = (1 + RISK * t[col]).cumprod()
    dd = (eq / eq.cummax() - 1).min()
    days = (t["exit_time"].max() - t["time"].min()).days or 1
    cagr = eq.iloc[-1] ** (365 / days) - 1
    daily = (RISK * t.set_index("exit_time")[col]).resample("1D").sum()
    daily = daily.reindex(pd.date_range(t["time"].min().floor("D"), t["exit_time"].max().floor("D"), freq="D",
                                        tz="UTC"), fill_value=0)
    sharpe = daily.mean() / daily.std() * math.sqrt(365) if daily.std() > 0 else np.nan
    return {"cagr": cagr, "max_dd": dd, "sharpe": sharpe, "per_day": len(t) / days}


def split_row(t: pd.DataFrame) -> dict:
    row = {}
    for name, part in (("all", t), ("is", t[t["time"] < SPLIT]), ("oos", t[t["time"] >= SPLIT])):
        for k, v in stats(part).items():
            row[f"{name}_{k}"] = v
        row[f"{name}_exp_slip"] = part["r_slip"].mean() if len(part) else np.nan
    row["long_exp"] = t.loc[t["side"] == LONG, "r"].mean()
    row["short_exp"] = t.loc[t["side"] == SHORT, "r"].mean()
    row["start"] = t["time"].min()
    for year, r in t.groupby(t["time"].dt.year)["r"]:
        row[f"y{year}"] = r.mean()
    return row


def main() -> int:
    p = argparse.ArgumentParser(description="訂單流策略回測")
    p.add_argument("--tf", default="4h", choices=["4h", "2h"])
    p.add_argument("--out", default="output/orderflow")
    p.add_argument("--reps", type=int, default=20, help="隨機進場重複次數")
    p.add_argument("--min-trades", type=int, default=200, help="選參數時，選參數那一段至少要有的筆數")
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cs = coins()
    labels = list(grid())
    print(f"{len(cs)} coins, {len(labels)} configs, {args.tf}", flush=True)
    workers = os.cpu_count() or 2

    with ProcessPoolExecutor(workers) as ex:
        rows = [r for part in ex.map(run_coin, [(c, args.tf, labels, (FEE, SLIP_FEE)) for c in cs]) for r in part]
        vrows = [r for part in ex.map(vegas_coin, [(c, args.tf) for c in cs]) for r in part]
    trades = pd.DataFrame(rows)
    vegas = pd.DataFrame(vrows)
    trades.to_csv(out / f"orderflow_{args.tf}_trades.csv.gz", index=False)
    vegas.to_csv(out / f"orderflow_{args.tf}_vegas_trades.csv.gz", index=False)

    table = []
    for label, t in trades.groupby("variant"):
        table.append({"variant": label, "family": family(label), **split_row(t)})
    table.append({"variant": "vegas_trend 收緊RSI", "family": "benchmark", **split_row(vegas)})
    summary = pd.DataFrame(table)
    summary.to_csv(out / f"orderflow_{args.tf}_grid.csv", index=False)

    # 每一類選一組：用 2020-2023 選（正向），也用 2024 之後選（反向），看另一段的表現
    picks = []
    for fam, g in summary[summary["family"] != "benchmark"].groupby("family"):
        for direction, sel, chk in (("正向", "is", "oos"), ("反向", "oos", "is")):
            cand = g[g[f"{sel}_n"] >= args.min_trades]
            if cand.empty:
                continue
            best = cand.sort_values(f"{sel}_exp", ascending=False).iloc[0]
            picks.append({"family": fam, "direction": direction, "variant": best["variant"],
                          "pick_n": best[f"{sel}_n"], "pick_exp": best[f"{sel}_exp"],
                          "check_n": best[f"{chk}_n"], "check_exp": best[f"{chk}_exp"], "check_se": best[f"{chk}_se"],
                          "check_exp_slip": best[f"{chk}_exp_slip"]})
    picks = pd.DataFrame(picks)
    picks.to_csv(out / f"orderflow_{args.tf}_picks.csv", index=False)

    # 入選的參數：和隨機進場比
    chosen = sorted(set(picks["variant"])) if len(picks) else []
    sub = trades[trades["variant"].isin(chosen)]
    jobs = [(c, args.tf, sub[sub["coin"] == c][["time", "side", "variant"]].to_dict("records"), args.reps, i)
            for i, c in enumerate(cs)]
    vj = [(c, args.tf, vegas[vegas["coin"] == c][["time", "side", "variant"]].to_dict("records"), args.reps, 1000 + i)
          for i, c in enumerate(cs)]
    with ProcessPoolExecutor(workers) as ex:
        rnd = pd.DataFrame([r for part in ex.map(random_coin, jobs + vj) for r in part])
    rnd.to_csv(out / f"orderflow_{args.tf}_random_trades.csv.gz", index=False)
    comp = []
    allt = pd.concat([sub, vegas])
    for label in chosen + ["vegas_trend 收緊RSI"]:
        t = allt[allt["variant"] == label]
        rr = rnd[rnd["variant"] == label]
        for period, tm, rm in (("全期", t["time"] >= t["time"].min(), rr["time"] >= rr["time"].min()),
                               ("2020-2023", t["time"] < SPLIT, rr["time"] < SPLIT),
                               ("2024後", t["time"] >= SPLIT, rr["time"] >= SPLIT)):
            tt, rp = t[tm], rr[rm]
            per_rep = rp.groupby("rep")["r"].mean() if len(rp) else pd.Series(dtype=float)
            comp.append({"variant": label, "period": period, "n": len(tt), "exp": tt["r"].mean(),
                         "exp_slip": tt["r_slip"].mean(), "random_exp": per_rep.mean(),
                         "random_beat_share": (per_rep >= tt["r"].mean()).mean() if len(per_rep) else np.nan,
                         **equity_stats(tt), **{f"slip_{k}": v for k, v in equity_stats(tt, "r_slip").items()}})
    comp = pd.DataFrame(comp)
    comp.to_csv(out / f"orderflow_{args.tf}_random.csv", index=False)

    # 用訂單流過濾 vegas_trend：訊號方向和訂單流一致的留下
    d = np.where(vegas["side"] == LONG, 1, -1)
    filters = {}
    for n in (1, 3, 6):
        filters[f"主動買賣差 {n}根 同向"] = d * vegas[f"imb{n}"] > 0
        filters[f"主動買賣差 {n}根 z>0.5 同向"] = d * vegas[f"imb{n}_z"] > 0.5
        filters[f"持倉量 {n}根 增加"] = vegas[f"doi{n}_z"] > 0
        filters[f"持倉量 {n}根 減少"] = vegas[f"doi{n}_z"] < 0
    filters["全體多空比 反向(散戶站另一邊)"] = d * vegas["account_ratio_z"] < 0
    filters["大戶持倉多空比 同向"] = d * vegas["top_position_ratio_z"] > 0
    frows = []
    has_oi = vegas["doi1_z"].notna()
    for name, mask in filters.items():
        base = vegas if not any(k in name for k in ("持倉", "多空比")) else vegas[has_oi]
        m = mask.loc[base.index]
        row = {"filter": name, "base_n": len(base)}
        for part, sel in (("is", base["time"] < SPLIT), ("oos", base["time"] >= SPLIT)):
            for grp, gm in (("keep", m), ("drop", ~m)):
                s = stats(base[sel & gm])
                row.update({f"{part}_{grp}_{k}": v for k, v in s.items() if k in ("n", "exp", "se")})
            row[f"{part}_all_exp"] = base.loc[sel, "r"].mean()
        frows.append(row)
    fdf = pd.DataFrame(frows)
    fdf.to_csv(out / f"orderflow_{args.tf}_vegas_filters.csv", index=False)

    with pd.option_context("display.width", 250, "display.max_columns", 40, "display.max_rows", 300):
        print(summary.sort_values("is_exp", ascending=False)[
            ["variant", "is_n", "is_exp", "oos_n", "oos_exp", "oos_exp_slip"]].round(3).to_string(index=False))
        print(picks.round(3).to_string(index=False))
        print(comp.round(3).to_string(index=False))
        print(fdf.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
