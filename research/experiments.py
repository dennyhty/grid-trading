"""實驗帳本：記錄每一個試過的策略／參數家族，之後驗證時才能校正「試了幾次」（多重檢定）。

- 帳本檔 research/experiments.csv（進 git）。每次新研究結束時呼叫 log_experiment() 補一列。
- trials() 回傳試驗數；deflated_sharpe() 和 required_t() 用這個數字把「最好的那個」的運氣扣掉。
- 一列可以代表一整個家族的掃描（例如 96 組參數），把組數寫在 notes 的 configs=N，
  trials(count_configs=True) 會把它算進去。沒寫 configs 的列算 1 次。

用法：
    from research.experiments import log_experiment, trials, deflated_sharpe, required_t
    log_experiment(family="vegas_stoploss", hypothesis="...", params={"atr": [1.0, 1.5]},
                   period="2020-2026", trades=9130, exp_r=0.147, t_stat=5.2, verdict="info",
                   report="signals/report_x.md", notes="configs=96")
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from statistics import NormalDist

LEDGER_PATH = Path(__file__).with_name("experiments.csv")
COLUMNS = ["ts", "experiment_id", "family", "hypothesis", "params", "data_source", "period", "trades", "exp_r",
           "se_cluster", "t_stat", "sharpe", "verdict", "report", "notes"]
VERDICTS = ("pass", "fail", "info")
_N = NormalDist()
EULER_GAMMA = 0.5772156649015329


def _path(path) -> Path:
    return Path(path) if path is not None else LEDGER_PATH


def make_id(family: str, params, period: str = "") -> str:
    """experiment_id = 家族名 + 參數與期間的 8 碼雜湊（同樣的設定同樣的 id）。"""
    p = params if isinstance(params, str) else json.dumps(params, sort_keys=True, ensure_ascii=False, default=str)
    return f"{family}-{hashlib.sha1(f'{p}|{period}'.encode()).hexdigest()[:8]}"


def log_experiment(path=None, **fields) -> str:
    """在帳本最後加一列，回傳 experiment_id。檔案不存在就建立（含標題列）。

    必填 family；其他欄位見 COLUMNS，沒給的留空。params 可以是 dict（會存成 JSON）。
    ts 預設為現在（UTC）。verdict 必須是 pass / fail / info。
    """
    unknown = set(fields) - set(COLUMNS)
    if unknown:
        raise ValueError(f"unknown fields: {sorted(unknown)}")
    if not fields.get("family"):
        raise ValueError("family is required")
    if fields.get("verdict") not in (None, "") and fields["verdict"] not in VERDICTS:
        raise ValueError(f"verdict must be one of {VERDICTS}")
    params = fields.get("params", "")
    if not isinstance(params, str):
        params = json.dumps(params, sort_keys=True, ensure_ascii=False, default=str)
    row = {c: "" for c in COLUMNS}
    row.update(fields)
    row["params"] = params
    row["ts"] = fields.get("ts") or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    row["experiment_id"] = fields.get("experiment_id") or make_id(row["family"], params, str(row["period"]))
    p = _path(path)
    new = not p.exists() or p.stat().st_size == 0
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        if new:
            w.writeheader()
        w.writerow(row)
    return row["experiment_id"]


def read_ledger(path=None) -> list[dict]:
    p = _path(path)
    if not p.exists():
        return []
    with p.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def trials(family: str | None = None, path=None, count_configs: bool = False) -> int:
    """帳本的列數（可只算某個家族）。count_configs=True 時，把 notes 裡的 configs=N 加總（沒寫的列算 1）。"""
    rows = [r for r in read_ledger(path) if family is None or r["family"] == family]
    if not count_configs:
        return len(rows)
    total = 0
    for r in rows:
        m = re.search(r"configs=(\d+)", r.get("notes") or "")
        total += int(m.group(1)) if m else 1
    return total


# ---------------------------------------------------------------- 多重檢定校正

def expected_max_sharpe(n_trials: int, sr_var_trials: float) -> float:
    """n 次獨立試驗、真實夏普都是 0 時，最好那個的期望夏普（Bailey & López de Prado 2014，單位同 sr_var_trials）。"""
    if n_trials <= 1:
        return 0.0
    return math.sqrt(sr_var_trials) * ((1 - EULER_GAMMA) * _N.inv_cdf(1 - 1 / n_trials)
                                       + EULER_GAMMA * _N.inv_cdf(1 - 1 / (n_trials * math.e)))


def deflated_sharpe(sharpe_annual: float, n_trials: int, n_obs_years: float, skew: float = 0.0, kurt: float = 3.0,
                    sr_var_trials: float | None = None, periods_per_year: int = 365) -> float:
    """Deflated Sharpe Ratio：真實夏普大於「n_trials 次隨機策略中最好那個的期望夏普」的機率。

    公式（Bailey & López de Prado 2014）：
        DSR = Phi( (SR - SR0) * sqrt(T - 1) / sqrt(1 - skew*SR + (kurt - 1)/4 * SR^2) )
        SR0 = sqrt(V) * ((1-g) * Phi^-1(1 - 1/N) + g * Phi^-1(1 - 1/(N e)))，g 為 Euler 常數，N = n_trials。
    內部把年化夏普換成每期（預設每日，加密貨幣 365 天）的夏普，T = n_obs_years * periods_per_year；
    skew、kurt（一般峰度，常態 = 3）是每期報酬的偏態與峰度。
    sr_var_trials：各試驗年化夏普的變異數；預設 1 / n_obs_years（夏普估計本身的抽樣變異，近似值，
    若手上有各設定的實際夏普，用它們的變異數更準）。
    """
    if n_obs_years <= 0:
        raise ValueError("n_obs_years must be > 0")
    if sr_var_trials is None:
        sr_var_trials = 1.0 / n_obs_years
    sr0_annual = expected_max_sharpe(int(n_trials), sr_var_trials)
    scale = math.sqrt(periods_per_year)
    sr, sr0 = sharpe_annual / scale, sr0_annual / scale
    t_obs = n_obs_years * periods_per_year
    denom = 1 - skew * sr + (kurt - 1) / 4 * sr * sr
    if denom <= 0 or t_obs <= 1:
        return float("nan")
    return _N.cdf((sr - sr0) * math.sqrt(t_obs - 1) / math.sqrt(denom))


def required_t(n_trials: int, alpha: float = 0.05) -> float:
    """試了 n_trials 次後，單邊檢定要求的 t 值門檻（Bonferroni：alpha / n），至少 2.0。"""
    n = max(int(n_trials), 1)
    return max(2.0, _N.inv_cdf(1 - alpha / n))


# ---------------------------------------------------------------- 既有研究的種子資料

def _seed(report, ts, family, hypothesis, verdict, period="2020-2026", data_source="binance", **kw):
    return dict(ts=ts, family=family, hypothesis=hypothesis, verdict=verdict, period=period, data_source=data_source,
                report=f"signals/{report}", **kw)


SEED_ROWS = [
    _seed("report_vegas_trend_exit_unify_20261015.md", "2026-10-15T00:00:00Z", "vegas_trend_live",
          "4H+2H 維加斯順勢，RSI 54-70/30-46，1/3/5R 出場（實盤基準）", "pass", period="2017-2026", trades=9130,
          exp_r=0.147, se_cluster=0.028, sharpe=1.59, notes="baseline; 2H alone +0.16R t 3.1/3.5 (basic_indicators report)"),
    _seed("report_basic_indicators_opus_max_20261013.md", "2026-10-13T00:00:00Z", "basic_indicators_reversal",
          "反轉派指標（RSI30/70、KD、MFI、威廉、CCI、布林回歸等）9 規則 x 4H/2H x 6 出場", "fail",
          notes="configs=108; 216 period-split combos only 2 slightly >0; account Sharpe none positive"),
    _seed("report_basic_indicators_opus_max_20261013.md", "2026-10-13T00:00:00Z", "basic_indicators_trend",
          "順勢派指標（MA、MACD、唐奇安、超級趨勢、一目、維加斯等）x 4H/2H x 實盤/1R 出場", "info",
          notes="configs=204 (312 total incl. reversal rule set); 26 significant after block-bootstrap multiple testing, all trend/vegas; "
                "daily-first-signal: 19-27 of 90 positive both periods ~ chance (22)"),
    _seed("report_vegas_trend_entry_20261013.md", "2026-10-13T00:00:00Z", "vegas_entry_filters",
          "41 種進場特徵 x 5 切法過濾維加斯訊號；2H ADX>=24 跳過", "info",
          notes="configs=410; only 14/205 (4H) and 18/205 (2H) beat live in both periods ~ chance; 2H ADX<24 adopted"),
    _seed("report_vegas_trend_stoploss_20261013.md", "2026-10-13T00:00:00Z", "vegas_stoploss",
          "96 種止損設定（ATR 倍數、結構、收盤確認、保本時機）", "fail",
          notes="configs=96; none robustly beats 1.5 ATR; trail 3.5 ATR +0.013R not significant"),
    _seed("report_tp_ladder_20261018.md", "2026-10-18T00:00:00Z", "tp_ladder",
          "止盈階梯 1/2/3R、1/3/5R、2/4/6R、1/2/4R", "info", period="2017-2026", trades=9130, exp_r=0.147,
          se_cluster=0.028, sharpe=1.59, notes="configs=4; 2/4/6R +0.228R paired +0.086+-0.018 but win rate 39%, DD -42%"),
    _seed("report_be_timing_20261018.md", "2026-10-18T00:00:00Z", "be_timing",
          "TP1 後保本 vs TP2 後才保本等 5 種移動止損時機", "info", period="2017-2026", trades=7882, exp_r=0.214,
          sharpe=1.58, notes="configs=5; B paired +0.069R ~4SE but win rate 54%->31%, same Sharpe"),
    _seed("report_regime_filter_20261018.md", "2026-10-18T00:00:00Z", "regime_filter",
          "BTC 日線 EMA200 / 200 週均線牛熊濾網（只做順向、熊市不做多、牛市不做空）", "fail", period="2020-2026",
          trades=4926, exp_r=0.087, sharpe=0.95, notes="configs=6; every filter <= baseline +0.150R/Sharpe 1.70"),
    _seed("report_alt_strategies_20261012.md", "2026-10-12T00:00:00Z", "alt_xsection_momentum",
          "橫斷面強弱輪動：做多最強、做空最弱（L/H/k/stop 共 96 組）", "pass", period="2024-2026", trades=1142,
          exp_r=0.104, se_cluster=0.062, t_stat=1.7, sharpe=0.80,
          notes="configs=96; selected on 2020-23, validated 2024-26; random-pick control 0/30; t approx from exp_r/SE"),
    _seed("report_alt_strategies_20261012.md", "2026-10-12T00:00:00Z", "alt_meanrev",
          "均值回歸（布林+RSI+盤整過濾）2H/4H", "fail", period="2020-2023", exp_r=-0.09, notes="configs=160; 0 of 80 positive in selection period"),
    _seed("report_alt_strategies_20261012.md", "2026-10-12T00:00:00Z", "alt_vwap",
          "VWAP 偏離回歸（24 組）與 VWAP 順勢回踩（4 組）", "fail", exp_r=-0.105,
          notes="configs=28; reversion all negative (-0.05 to -0.12R); trend pullback +0.04-0.07R but ~random in 2024-26"),
    _seed("report_smc_ict_20261012.md", "2026-10-12T00:00:00Z", "smc_ict",
          "SMC/ICT：4H 掃流動性+結構轉變（日線同向）、OB/FVG 回踩，1H/2H/4H", "info", period="2024-2026", trades=1965,
          exp_r=0.077, notes="configs=252; 32 positive both periods (24 at 4H); cluster-level 2024+ only ~1.5 SE; OB retest -0.09R"),
    _seed("report_orderflow_20261012.md", "2026-10-12T00:00:00Z", "orderflow",
          "訂單流：CVD 背離、主動買賣失衡、OI x 價格、多空比（4H/2H）", "info", period="2024-2026", trades=1837,
          exp_r=0.092, notes="configs not stated; OI-up-with-price long/short +0.092R but cluster-level only +0.027R; CVD divergence -0.08 to -0.15R"),
    _seed("report_harmonic_20261018.md", "2026-10-18T00:00:00Z", "harmonic",
          "9 種諧波型態 x 3 週期 x 3 靈敏度 x 4 出場（+EMA200 順勢濾網）", "fail", exp_r=-0.11,
          notes="configs=324; 180 with n>=100 median -0.11R, none t>2; +136 with EMA200 filter"),
    _seed("report_outside_day_20261015.md", "2026-10-15T00:00:00Z", "outside_day",
          "Larry Williams 外包日反轉（日線/4H、多空、止損 1-3 ATR 或外包日低點）", "fail",
          notes="configs=48; none positive in both periods, ~ baseline of random entry"),
]


def seed_ledger(path=None, force: bool = False) -> int:
    """把既有研究摘要寫進帳本。帳本已存在時不動（force=True 才附加）。回傳寫入列數。"""
    p = _path(path)
    if p.exists() and p.stat().st_size and not force:
        return 0
    for row in SEED_ROWS:
        log_experiment(path=p, **row)
    return len(SEED_ROWS)


if __name__ == "__main__":
    print(f"seeded {seed_ledger()} rows" if not LEDGER_PATH.exists() else f"{trials()} rows, {trials(count_configs=True)} configs")
