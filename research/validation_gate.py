"""策略驗證關卡：新策略要上線（或繼續當實盤）之前，一律先過這一關。

    python -m research.validation_gate --trades output/pit_trades.csv.gz --name vegas_live --pool in_pit

輸入是逐筆交易（DataFrame，欄位 symbol、side(+1/-1)、time(進場 UTC)、exit_t、r、r_net…），輸出一份
檢查清單：每項有數值、門檻、通過與否。重點是不要被運氣騙：

1. 筆數夠不夠；
2. 樣本內、樣本外（oos_start 之後）扣成本後的平均 R 都要為正；
3. 嚴格的群集 t 值：取「進場日+方向、出場日+方向、ISO 週、月」四種分群裡最小的 t，
   要大於「試了 n_trials 組設定」之後的門檻（research.experiments.required_t）；
4. Deflated Sharpe（日報酬，扣掉試驗次數的運氣）>= 0.95；
5. 贏過隨機進場（同樣出場規則）；
6. 參數高原：鄰近參數也要是正的；
7. 和實盤維加斯帳本的日報酬相關性要低（才有分散效果）；
8. 穩健性：拿掉前幾大幣、最好的一年、最好的幾個出場日後仍為正。

整體 passed = 所有「沒被略過」的檢查都通過；隨機基準／高原／相關性有任何一項沒給資料，
結論就是 incomplete（不完整），不能當成通過。
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

from research.experiments import deflated_sharpe, log_experiment, required_t, trials
from research.stop_research import cluster_key, cluster_stats

_N = NormalDist()
REQUIRED_COLUMNS = ("symbol", "side", "time", "exit_t")
# 沒給資料就會被略過、讓結論變成 incomplete 的檢查
OPTIONAL_CHECKS = ("beats_random", "plateau", "baseline_corr")

PASS, FAIL, SKIPPED, INFO = "pass", "fail", "skipped", "info"


def _default_trials() -> int:
    """實驗帳本裡「試過幾組設定」（含 notes 裡的 configs=N）。"""
    return trials(count_configs=True)


@dataclass
class GateConfig:
    """驗證關卡的門檻。n_trials 預設讀實驗帳本，之後每多試一組就會更嚴。"""

    oos_start: str = "2024-01-01"   # 樣本外起點（以進場時間切）
    min_trades: int = 300           # 最少筆數；樣本外至少 min_trades/3
    base_t: float = 2.0             # 未校正的 t 門檻，只當參考資訊
    n_trials: int = field(default_factory=_default_trials)
    max_corr: float = 0.5           # 與基準（實盤維加斯）日報酬相關係數上限
    drop_top_coins: int = 5         # 穩健性：拿掉總 R 最高的幾個幣
    risk: float = 0.0025            # 每筆風險（帳戶比例）
    # 以下是規格之外的細部門檻，一樣不要為了讓某個策略過關而調整
    drop_top_days: int = 5          # 穩健性：拿掉最好的幾個出場日
    dsr_min: float = 0.95           # Deflated Sharpe 下限
    random_z: float = 2.0           # 贏過隨機進場要超過幾個標準誤
    plateau_frac: float = 0.5       # 鄰近參數平均 R 至少是主設定的幾成
    periods_per_year: int = 365     # 日報酬年化（加密貨幣 365 天）


# ---------------------------------------------------------------- 小工具

def _check(status: str, value, threshold, detail: str, label: str) -> dict:
    """統一的檢查結果格式。passed 只有 status == pass 才是 True（略過不算通過）。"""
    return {"label": label, "value": value, "threshold": threshold, "passed": status == PASS,
            "status": status, "detail": detail}


def _prepare(trades: pd.DataFrame, rcol: str, what: str = "trades") -> pd.DataFrame:
    """檢查必要欄位、時間轉 UTC、丟掉 R 是 NaN 的列。"""
    missing = [c for c in (*REQUIRED_COLUMNS, rcol) if c not in trades.columns]
    if missing:
        raise ValueError(f"{what} 缺少欄位：{missing}")
    df = trades.copy()
    for c in ("time", "exit_t"):
        df[c] = pd.to_datetime(df[c], utc=True)
    return df[df[rcol].notna()].reset_index(drop=True)


def _entry_key(df: pd.DataFrame) -> np.ndarray:
    return cluster_key(df)


def _exit_key(df: pd.DataFrame) -> np.ndarray:
    return (df["exit_t"].dt.floor("D").astype(str) + df["side"].astype(str)).to_numpy()


def _week_key(df: pd.DataFrame) -> np.ndarray:
    iso = df["time"].dt.isocalendar()
    return (iso["year"].astype(str) + "-" + iso["week"].astype(str)).to_numpy()


def _month_key(df: pd.DataFrame) -> np.ndarray:
    return df["time"].dt.strftime("%Y-%m").to_numpy()


CLUSTERINGS = {"進場日+方向": _entry_key, "出場日+方向": _exit_key, "ISO週(進場)": _week_key, "月(進場)": _month_key}


def cluster_t_values(df: pd.DataFrame, rcol: str) -> dict[str, float]:
    """四種分群下的 t = 平均 / 群集穩健標準誤。"""
    x = df[rcol].to_numpy(float)
    out = {}
    for name, fn in CLUSTERINGS.items():
        mean, se, *_ = cluster_stats(x, fn(df))
        out[name] = float(mean / se) if se and se > 0 else float("nan")
    return out


def daily_returns(df: pd.DataFrame, rcol: str, risk: float, start=None, end=None) -> pd.Series:
    """單一帳本的日報酬：每筆 R x risk，依出場日入帳加總；沒交易的日子補 0。"""
    s = pd.Series(df[rcol].to_numpy(float) * risk, index=df["exit_t"].dt.floor("D")).groupby(level=0).sum()
    lo = pd.Timestamp(start if start is not None else df["time"].min()).floor("D")
    hi = pd.Timestamp(end if end is not None else df["exit_t"].max()).floor("D")
    return s.reindex(pd.date_range(lo, hi, freq="D"), fill_value=0.0)


def _mean(x) -> float:
    return float(np.mean(x)) if len(x) else float("nan")


def _pos(x: float) -> bool:
    return bool(np.isfinite(x) and x > 0)


# ---------------------------------------------------------------- 各項檢查

def _check_sample_split(df: pd.DataFrame, rcol: str, cfg: GateConfig) -> dict[str, dict]:
    oos = pd.Timestamp(cfg.oos_start, tz="UTC")
    ins_df, oos_df = df[df["time"] < oos], df[df["time"] >= oos]
    m_in, m_out = _mean(ins_df[rcol]), _mean(oos_df[rcol])
    min_oos = cfg.min_trades / 3
    return {
        "mean_in_sample": _check(
            PASS if _pos(m_in) else FAIL, m_in, "> 0",
            f"樣本內（< {cfg.oos_start}）{len(ins_df)} 筆，平均 {m_in:+.4f}R", "樣本內平均 R > 0"),
        "mean_oos": _check(
            PASS if _pos(m_out) and len(oos_df) >= min_oos else FAIL, m_out, f"> 0 且筆數 >= {min_oos:.0f}",
            f"樣本外（>= {cfg.oos_start}）{len(oos_df)} 筆，平均 {m_out:+.4f}R"
            + ("" if len(oos_df) >= min_oos else "；筆數不足"), "樣本外平均 R > 0"),
    }


def _check_cluster_t(df: pd.DataFrame, rcol: str, cfg: GateConfig) -> dict:
    ts = cluster_t_values(df, rcol)
    finite = [v for v in ts.values() if np.isfinite(v)]
    t_min = min(finite) if finite else float("nan")
    need = required_t(cfg.n_trials)
    ok = np.isfinite(t_min) and t_min >= need
    detail = ("、".join(f"{k} {v:.2f}" for k, v in ts.items())
              + f"；試了 {cfg.n_trials} 組設定，門檻 {need:.2f}；未校正門檻 {cfg.base_t:.1f}："
              + ("最小 t 達標" if np.isfinite(t_min) and t_min >= cfg.base_t else "最小 t 也未達標"))
    chk = _check(PASS if ok else FAIL, t_min, need, detail, "嚴格群集 t（四種分群取最小）")
    chk["per_clustering"] = ts
    chk["base_t"] = cfg.base_t
    chk["passes_base_t"] = bool(np.isfinite(t_min) and t_min >= cfg.base_t)
    return chk


def _check_dsr(df: pd.DataFrame, rcol: str, cfg: GateConfig) -> tuple[dict, dict]:
    day = daily_returns(df, rcol, cfg.risk)
    sd = day.std()
    years = len(day) / cfg.periods_per_year
    if len(day) < 3 or not sd or sd <= 0:
        return (_check(FAIL, float("nan"), f">= {cfg.dsr_min}", "日報酬標準差為 0 或天數太少，無法計算", "Deflated Sharpe"),
                {"days": len(day)})
    sharpe = float(day.mean() / sd * math.sqrt(cfg.periods_per_year))
    skew, kurt = float(day.skew()), float(day.kurt()) + 3.0   # pandas 的 kurt 是超額峰度，換成一般峰度
    dsr = deflated_sharpe(sharpe, cfg.n_trials, years, skew=skew, kurt=kurt, periods_per_year=cfg.periods_per_year)
    ok = bool(np.isfinite(dsr) and dsr >= cfg.dsr_min)
    detail = (f"年化夏普 {sharpe:.2f}，{len(day)} 天，偏態 {skew:.2f}，峰度 {kurt:.1f}，試驗數 {cfg.n_trials}")
    info = {"sharpe_annual": sharpe, "days": len(day), "skew": skew, "kurt": kurt}
    return _check(PASS if ok else FAIL, float(dsr), f">= {cfg.dsr_min}", detail, "Deflated Sharpe"), info


def _year_stats(df: pd.DataFrame, rcol: str) -> dict[int, tuple[int, float, float]]:
    """進場年 -> (筆數, 平均, 群集標準誤)。"""
    out = {}
    for y, g in df.groupby(df["time"].dt.year):
        mean, se, *_ = cluster_stats(g[rcol].to_numpy(float), cluster_key(g))
        out[int(y)] = (len(g), float(mean), float(se))
    return out


def _check_random(df: pd.DataFrame, rnd: pd.DataFrame | None, rcol: str, cfg: GateConfig) -> dict:
    label = "贏過隨機進場"
    if rnd is None:
        return _check(SKIPPED, None, f"差 > {cfg.random_z:g} 標準誤", "沒提供隨機進場基準，略過（不算通過）", label)
    rnd = _prepare(rnd, rcol, "random_baseline")
    s, r = _year_stats(df, rcol), _year_stats(rnd, rcol)
    years = sorted(set(s) & set(r))
    if not years:
        return _check(FAIL, float("nan"), f"差 > {cfg.random_z:g} 標準誤", "策略與隨機基準沒有共同的年份", label)
    # 以進場年分層配對：每年的策略平均減隨機平均，依策略筆數加權；兩組獨立，變異數相加
    n_tot = sum(s[y][0] for y in years)
    w = {y: s[y][0] / n_tot for y in years}
    diffs = {y: s[y][1] - r[y][1] for y in years}
    diff = sum(w[y] * diffs[y] for y in years)
    se = math.sqrt(sum(w[y] ** 2 * (s[y][2] ** 2 + r[y][2] ** 2) for y in years))
    z = diff / se if se > 0 else float("nan")
    ok = bool(np.isfinite(z) and z > cfg.random_z)
    n_pos = sum(d > 0 for d in diffs.values())
    detail = (f"策略 - 隨機 = {diff:+.4f}R，標準誤 {se:.4f}（{len(years)} 個共同年份，{n_pos} 年為正；"
              f"隨機基準 {len(rnd)} 筆）")
    chk = _check(PASS if ok else FAIL, float(z), f"> {cfg.random_z:g}", detail, label)
    chk["diff"], chk["se"], chk["by_year"] = float(diff), float(se), {y: float(d) for y, d in diffs.items()}
    return chk


def _check_plateau(main_mean: float, neighbors: list[pd.DataFrame] | None, rcol: str, cfg: GateConfig) -> dict:
    label = "參數高原"
    thr = f"所有鄰近 > 0 且 >= 主設定的 {cfg.plateau_frac:.0%}"
    if not neighbors:
        return _check(SKIPPED, None, thr, "沒提供鄰近參數的交易，略過（不算通過）", label)
    means = [_mean(_prepare(n, rcol, f"neighbors[{k}]")[rcol]) for k, n in enumerate(neighbors)]
    floor = cfg.plateau_frac * main_mean
    ok = all(_pos(m) and m >= floor for m in means)
    detail = f"主設定 {main_mean:+.4f}R；{len(means)} 組鄰近：" + "、".join(f"{m:+.4f}" for m in means)
    chk = _check(PASS if ok else FAIL, means, thr, detail, label)
    chk["min_neighbor"] = float(np.nanmin(means))
    return chk


def _check_corr(df: pd.DataFrame, base: pd.DataFrame | None, rcol: str, cfg: GateConfig) -> dict:
    label = "與實盤基準的相關性"
    if base is None:
        return _check(SKIPPED, None, f"< {cfg.max_corr}", "沒提供基準帳本，略過（不算通過）", label)
    base = _prepare(base, rcol, "baseline")
    lo = min(df["time"].min(), base["time"].min())
    hi = max(df["exit_t"].max(), base["exit_t"].max())
    a = daily_returns(df, rcol, cfg.risk, lo, hi)
    b = daily_returns(base, rcol, cfg.risk, lo, hi)
    corr = float(a.corr(b)) if a.std() > 0 and b.std() > 0 else float("nan")
    ok = bool(np.isfinite(corr) and corr < cfg.max_corr)
    return _check(PASS if ok else FAIL, corr, f"< {cfg.max_corr}",
                  f"出場日加總的日報酬相關係數（{len(a)} 天，基準 {len(base)} 筆）", label)


def _check_robustness(df: pd.DataFrame, rcol: str, cfg: GateConfig) -> dict[str, dict]:
    out = {}
    # 拿掉總 R 最高的幾個幣
    top = df.groupby("symbol")[rcol].sum().nlargest(cfg.drop_top_coins).index
    m = _mean(df.loc[~df["symbol"].isin(top), rcol])
    out["robust_drop_coins"] = _check(
        PASS if _pos(m) else FAIL, m, "> 0",
        f"拿掉總 R 最高的 {cfg.drop_top_coins} 個幣（{', '.join(map(str, top))}）後平均 {m:+.4f}R",
        f"拿掉前 {cfg.drop_top_coins} 大幣後 > 0")
    # 拿掉最好的一個曆年（依出場年）
    year = df["exit_t"].dt.year
    ysum = df.groupby(year)[rcol].sum()
    best = int(ysum.idxmax())
    m = _mean(df.loc[year != best, rcol])
    out["robust_drop_year"] = _check(
        PASS if _pos(m) else FAIL, m, "> 0", f"拿掉最好的一年（{best}，合計 {ysum.max():+.1f}R）後平均 {m:+.4f}R",
        "拿掉最好的一年後 > 0")
    # 拿掉最好的幾個出場日
    day = df["exit_t"].dt.floor("D")
    dsum = df.groupby(day)[rcol].sum()
    best_days = dsum.nlargest(cfg.drop_top_days).index
    m = _mean(df.loc[~day.isin(best_days), rcol])
    out["robust_drop_days"] = _check(
        PASS if _pos(m) else FAIL, m, "> 0",
        f"拿掉最好的 {cfg.drop_top_days} 個出場日（{', '.join(d.strftime('%Y-%m-%d') for d in best_days)}；"
        f"合計 {dsum.nlargest(cfg.drop_top_days).sum():+.1f}R）後平均 {m:+.4f}R",
        f"拿掉最好的 {cfg.drop_top_days} 個出場日後 > 0")
    return out


def _check_sides(df: pd.DataFrame, rcol: str) -> dict:
    sides = {}
    for name, sgn in (("long", 1), ("short", -1)):
        g = df[df["side"] == sgn]
        sides[name] = {"trades": len(g), "mean": _mean(g[rcol])}
    return _check(INFO, {k: v["mean"] for k, v in sides.items()}, None,
                  f"多單 {sides['long']['trades']} 筆 {sides['long']['mean']:+.4f}R；"
                  f"空單 {sides['short']['trades']} 筆 {sides['short']['mean']:+.4f}R", "多空拆分（參考）")


# ---------------------------------------------------------------- 主函式

def evaluate(trades: pd.DataFrame, baseline: pd.DataFrame | None = None, random_baseline: pd.DataFrame | None = None,
             cfg: GateConfig | None = None, rcol: str = "r_net", neighbors: list[pd.DataFrame] | None = None) -> dict:
    """跑完整個驗證關卡。

    trades：已經是單一帳本的逐筆交易（同週期同幣同時一張）；baseline：實盤維加斯的帳本，算相關性用；
    random_baseline：同樣出場規則、隨機進場的交易；neighbors：鄰近參數的交易（參數高原）。

    回傳 dict：
      checks     名稱 -> {label, value, threshold, passed, status(pass/fail/skipped/info), detail}
      passed     所有非略過的檢查都通過
      verdict    文字結論；隨機基準／高原／相關性有任何一項被略過則標示 incomplete
      incomplete 是否有被略過的必要檢查；skipped / failed 列出名稱
      info       一些摘要數字；config 為使用的門檻
    """
    cfg = cfg or GateConfig()
    df = _prepare(trades, rcol)
    checks: dict[str, dict] = {}

    n = len(df)
    checks["n_trades"] = _check(PASS if n >= cfg.min_trades else FAIL, n, f">= {cfg.min_trades}",
                                f"共 {n} 筆", "筆數")
    if n == 0:
        raise ValueError("沒有可用的交易")
    checks.update(_check_sample_split(df, rcol, cfg))
    checks["cluster_t"] = _check_cluster_t(df, rcol, cfg)
    dsr_chk, dsr_info = _check_dsr(df, rcol, cfg)
    checks["deflated_sharpe"] = dsr_chk
    checks["beats_random"] = _check_random(df, random_baseline, rcol, cfg)
    main_mean = _mean(df[rcol])
    checks["plateau"] = _check_plateau(main_mean, neighbors, rcol, cfg)
    checks["baseline_corr"] = _check_corr(df, baseline, rcol, cfg)
    checks.update(_check_robustness(df, rcol, cfg))
    checks["sides"] = _check_sides(df, rcol)

    failed = [k for k, c in checks.items() if c["status"] == FAIL]
    skipped = [k for k, c in checks.items() if c["status"] == SKIPPED]
    incomplete = any(k in skipped for k in OPTIONAL_CHECKS)
    passed = not failed
    if failed:
        verdict = "FAIL 未通過：" + "、".join(checks[k]["label"] for k in failed)
        if incomplete:
            verdict += "（另有未執行的檢查，結論也不完整 incomplete）"
    elif incomplete:
        verdict = ("INCOMPLETE 不完整：已執行的檢查全部通過，但尚未執行 "
                   + "、".join(checks[k]["label"] for k in skipped) + "，不能當成通過")
    else:
        verdict = "PASS 通過：所有檢查都通過"
    info = {"trades": n, "mean_r": main_mean, "rcol": rcol, "first_entry": str(df["time"].min()),
            "last_exit": str(df["exit_t"].max()), "symbols": int(df["symbol"].nunique()), **dsr_info}
    return {"checks": checks, "passed": passed, "verdict": verdict, "incomplete": incomplete, "failed": failed,
            "skipped": skipped, "info": info, "config": asdict(cfg)}


# ---------------------------------------------------------------- 報告

_MARK = {PASS: "✓", FAIL: "✗", SKIPPED: "–", INFO: "·"}


def _fmt(v) -> str:
    if v is None:
        return "—"
    if isinstance(v, dict):
        return "、".join(f"{k}:{_fmt(x)}" for k, x in v.items())
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_fmt(x) for x in v) + "]"
    if isinstance(v, (float, np.floating)):
        return f"{v:+.4f}" if abs(v) < 1 else f"{v:.2f}"
    return str(v)


def report_md(result: dict, name: str) -> str:
    """繁體中文 markdown 報告：✓ 通過、✗ 未通過、– 略過、· 僅參考。"""
    lines = [f"## 策略驗證關卡：{name}", "", f"**結論：{result['verdict']}**", "",
             "| 結果 | 檢查 | 數值 | 門檻 | 說明 |", "|:-:|---|---|---|---|"]
    for c in result["checks"].values():
        lines.append(f"| {_MARK[c['status']]} | {c['label']} | {_fmt(c['value'])} | {_fmt(c['threshold'])} | "
                     f"{c['detail'].replace('|', '/')} |")
    cfg, info = result["config"], result["info"]
    lines += ["", f"- 圖例：✓ 通過、✗ 未通過、– 略過（沒給資料，不算通過）、· 僅參考。報酬欄位 `{info['rcol']}`，"
                  f"{info['trades']} 筆、{info['symbols']} 個幣，{info['first_entry'][:10]} 至 {info['last_exit'][:10]}。",
              f"- 門檻：試驗數 {cfg['n_trials']}（嚴格 t 門檻 {required_t(cfg['n_trials']):.2f}），樣本外起點 {cfg['oos_start']}，"
              f"每筆風險 {cfg['risk']:.2%}。"]
    if result["skipped"]:
        lines.append("- 略過的檢查：" + "、".join(result["checks"][k]["label"] for k in result["skipped"]) + "。")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- 命令列

def _load(path: str) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["time", "exit_t"])


def run_cli(argv: list[str] | None = None) -> dict:
    """python -m research.validation_gate --trades output/pit_trades.csv.gz --name vegas_live --pool in_pit"""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--trades", required=True, help="逐筆交易 csv(.gz)")
    ap.add_argument("--name", required=True, help="策略名稱（記進帳本時當 family）")
    ap.add_argument("--pool", default="in_pit",
                    help="幣池欄位（True 的才算）；會用 pit_backtest.book 轉成實盤帳本。'none' 表示檔案已經是帳本")
    ap.add_argument("--rcol", default="r_net")
    ap.add_argument("--baseline", help="基準帳本（算相關性）csv(.gz)；有幣池欄位時會用 --baseline-pool 轉帳本")
    ap.add_argument("--baseline-pool", default="in_pit")
    ap.add_argument("--random", dest="random_baseline", help="隨機進場基準 csv(.gz)")
    ap.add_argument("--neighbor", action="append", default=[], help="鄰近參數的交易 csv(.gz)，可重複")
    ap.add_argument("--oos-start", default=GateConfig.oos_start)
    ap.add_argument("--min-trades", type=int, default=GateConfig.min_trades)
    ap.add_argument("--n-trials", type=int, help="試驗數，預設讀實驗帳本的 configs 總和")
    ap.add_argument("--log", action="store_true", help="把結果寫進實驗帳本（預設不寫）")
    ap.add_argument("--ledger", help="帳本路徑（預設 research/experiments.csv）")
    ap.add_argument("--report", default="", help="記進帳本的報告檔名")
    a = ap.parse_args(argv)

    def as_book(df: pd.DataFrame, pool: str) -> pd.DataFrame:
        if pool.lower() == "none" or pool not in df.columns:
            return df
        from research.pit_backtest import book   # 延後匯入：book 會拉進整個回測相依套件
        return book(df, pool)

    tr = as_book(_load(a.trades), a.pool)
    kw = {"oos_start": a.oos_start, "min_trades": a.min_trades}
    if a.n_trials is not None:
        kw["n_trials"] = a.n_trials
    cfg = GateConfig(**kw)
    res = evaluate(tr, baseline=as_book(_load(a.baseline), a.baseline_pool) if a.baseline else None,
                   random_baseline=_load(a.random_baseline) if a.random_baseline else None,
                   cfg=cfg, rcol=a.rcol, neighbors=[_load(p) for p in a.neighbor] or None)
    print(report_md(res, a.name))
    if a.log:
        ck = res["checks"]
        t_min = ck["cluster_t"]["value"]
        exp_id = log_experiment(
            path=a.ledger, family=a.name, hypothesis=f"validation_gate on {a.trades} pool={a.pool}",
            params={"oos_start": cfg.oos_start, "min_trades": cfg.min_trades, "n_trials": cfg.n_trials,
                    "rcol": a.rcol, "pool": a.pool},
            data_source=Path(a.trades).name, period=f"{res['info']['first_entry'][:4]}-{res['info']['last_exit'][:4]}",
            trades=res["info"]["trades"], exp_r=round(res["info"]["mean_r"], 4),
            t_stat=round(t_min, 3) if np.isfinite(t_min) else "",
            sharpe=round(res["info"].get("sharpe_annual", float("nan")), 3),
            verdict="fail" if not res["passed"] else ("info" if res["incomplete"] else "pass"),
            report=a.report, notes=f"validation_gate; {res['verdict']}")
        print(f"已寫入實驗帳本：{exp_id}", file=sys.stderr)
    return res


if __name__ == "__main__":
    run_cli()
