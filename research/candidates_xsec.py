"""候選 B：每週橫斷面動能／低波動多空組合（時點幣池、波動縮放、對 BTC 做 beta 中性、扣成本）。

經濟理由（誰把錢輸給你）：
- 動能：資訊在幣與幣之間擴散得慢、散戶追價後到，過去 2～8 週的強勢幣接下來幾週相對續強（Liu & Tsyvinski 等）。
- 低波動：散戶偏好「樂透型」高波動小幣，把它們的價格墊高，長期相對報酬偏低（betting-against-beta）。
  輸家是買高波動小幣的散戶和為了槓桿上限只能買高 beta 的人。
- 和維加斯的關係：這是「相對」報酬，整個組合對 BTC 做 beta 中性，理論上不靠大盤崩盤日賺錢；
  是組合層級的策略，所以直接建日報酬序列，不是逐筆 R。

規則：
- 每週日日線收盤（= 週一 00:00 UTC）再平衡，持有到下週日收盤（固定數量，不每天調回權重）。
- 合格：週一所在月份在時點幣池（上個月成交額前 100）、有至少 61 天日線、30 天波動與 60 天 beta 算得出來；合格少於 10 個就空手。
- 分數：mom{L} = 過去 L 天報酬；lowvol = 負的 30 天日報酬標準差；combo = mom28 和 lowvol 的橫斷面百分位平均。
- 做多分數最高的 q 比例、做空最低的 q 比例（每邊至少 3 個）。
- 每個部位名目 = 0.25% / (30 天日波動 x sqrt(7))，也就是「一週一個標準差 = 帳戶 0.25%」，
  所以每個部位一週的報酬換成 R（1R = 一週一個標準差）就和其他候選的 0.25% 風險可比。
- beta 中性：各部位對 BTC 的 60 天 beta 乘名目加總，用 BTCUSDT 反向對沖（對沖單一樣付成本）。
- 成本：換手名目 x（手續費每邊 0.05% + 依 30 天日均成交額的滑價分級，grid_trading.costs.slippage_bps），
  資金費率用幣安歷史費率（多單在正費率時付錢，空單收錢），以每天開始時的名目計。

網格（預先登記，共 10 組）：分數 ∈ {mom14, mom28, mom56, lowvol, combo} × q ∈ {0.2, 0.3}。
"""

from __future__ import annotations

import itertools
import math
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from grid_trading.costs import fetch_funding, slippage_bps
from research.candidates_common import FEE, RISK

FAMILY = "cand_xsec_portfolio"
PANEL_CACHE = Path("data/candidates/xsec_panel.pkl")
MIN_ELIG, MIN_SIDE, MIN_HIST = 10, 3, 61
HEDGE = "BTCUSDT"


def configs() -> dict[str, dict]:
    out = {}
    for sig, q in itertools.product(("mom14", "mom28", "mom56", "lowvol", "combo"), (0.2, 0.3)):
        out[f"{sig}|q{q:g}"] = {"signal": sig, "q": q}
    return out


def neighbors(name: str) -> list[str]:
    """鄰近設定：q 換另一格；動能另外加回看長度相鄰的一格。"""
    p = configs()[name]
    q2 = 0.3 if p["q"] == 0.2 else 0.2
    out = [f"{p['signal']}|q{q2:g}"]
    moms = ["mom14", "mom28", "mom56"]
    if p["signal"] in moms:
        i = moms.index(p["signal"])
        out += [f"{moms[j]}|q{p['q']:g}" for j in (i - 1, i + 1) if 0 <= j < 3]
    return out


# ---------------------------------------------------------------- 面板資料

@dataclass
class Panel:
    days: pd.DatetimeIndex     # 日線開盤日（UTC 00:00）
    syms: list
    close: np.ndarray          # 原始收盤（沒資料為 NaN）
    close_ff: np.ndarray       # 向前填補（下架後部位停在最後價格）
    adv: np.ndarray            # 30 天平均日成交額
    vol: np.ndarray            # 30 天日報酬標準差
    beta: np.ndarray           # 對 BTC 的 60 天 beta
    hist: np.ndarray           # 到當天為止有幾天資料
    fund: np.ndarray           # 這根日 K（k 00:00, k+1 00:00] 內的資金費率加總


def funding_daily(df: pd.DataFrame, days: pd.DatetimeIndex) -> np.ndarray:
    """把每次結算的費率加到所屬的日 K：結算時間 τ 落在 (k 00:00, k+1 00:00] 就算第 k 天。"""
    if df is None or len(df) == 0:
        return np.zeros(len(days))
    t = pd.to_datetime(df["time"], utc=True).dt.floor("s")
    bucket = (t - pd.Timedelta(nanoseconds=1)).dt.floor("D")
    s = pd.Series(df["rate"].to_numpy(float), index=bucket).groupby(level=0).sum()
    return s.reindex(days, fill_value=0.0).to_numpy()


def rolling_beta(r: pd.DataFrame, rb: pd.Series, win: int = 60, minp: int = 40) -> pd.DataFrame:
    cov = r.rolling(win, min_periods=minp).cov(rb)
    var = rb.rolling(win, min_periods=minp).var()
    return cov.div(var, axis=0)


def build_panel(data: dict[str, pd.DataFrame], funding: dict | None = None, cache: bool = True) -> Panel:
    """日線面板。funding 不給就用 fetch_funding（有快取）抓；cache=True 時面板存到 data/candidates/。"""
    syms = sorted(data)
    if cache and funding is None and PANEL_CACHE.exists():
        with PANEL_CACHE.open("rb") as f:
            p = pickle.load(f)
        if p.syms == syms:
            return p
    close = pd.DataFrame({s: data[s].set_index("time")["close"] for s in syms}).sort_index()
    qv = pd.DataFrame({s: data[s].set_index("time")["qv"] for s in syms}).reindex(close.index)
    days = close.index
    ret = np.log(close / close.shift(1))
    vol = ret.rolling(30, min_periods=20).std()
    beta = rolling_beta(ret, ret[HEDGE])
    adv = qv.rolling(30, min_periods=5).mean()
    hist = close.notna().cumsum()
    fetched = funding is None
    if fetched:
        funding = {}
        for s in syms:
            try:
                funding[s] = fetch_funding(s)
            except Exception:   # noqa: BLE001 - 抓不到就當 0，和 apply_costs 一樣
                funding[s] = None
    fund = np.column_stack([funding_daily(funding.get(s), days) for s in syms])
    p = Panel(days, syms, close.to_numpy(), close.ffill().to_numpy(), adv.to_numpy(), vol.to_numpy(),
              beta.to_numpy(), hist.to_numpy(), fund)
    if cache and fetched:
        PANEL_CACHE.parent.mkdir(parents=True, exist_ok=True)
        with PANEL_CACHE.open("wb") as f:
            pickle.dump(p, f)
    return p


# ---------------------------------------------------------------- 訊號與權重

def pct_rank(x: np.ndarray) -> np.ndarray:
    """橫斷面百分位（0～1，NaN 保持 NaN）。"""
    s = pd.Series(x)
    return s.rank(pct=True).to_numpy()


def scores(p: Panel, a: int, signal: str) -> np.ndarray:
    """第 a 天收盤時每個幣的分數（越大越想做多）。只用到第 a 天（含）以前的資料。"""
    def mom(L):
        if a - L < 0:
            return np.full(len(p.syms), np.nan)
        with np.errstate(invalid="ignore", divide="ignore"):
            return p.close[a] / p.close[a - L] - 1
    if signal.startswith("mom"):
        return mom(int(signal[3:]))
    if signal == "lowvol":
        return -p.vol[a]
    if signal == "combo":
        return (pct_rank(mom(28)) + pct_rank(-p.vol[a])) / 2
    raise ValueError(signal)


def select(score: np.ndarray, elig: np.ndarray, q: float, rng: np.random.Generator | None = None):
    """回傳 (多單索引, 空單索引)。合格不足 MIN_ELIG 回傳空。rng 給的話隨機挑（同樣的多空數量）。"""
    cand = np.flatnonzero(elig & np.isfinite(score))
    if len(cand) < MIN_ELIG:
        return np.array([], int), np.array([], int)
    k = max(MIN_SIDE, int(q * len(cand)))
    if rng is not None:
        pick = rng.permutation(cand)
        return np.sort(pick[:k]), np.sort(pick[k:2 * k])
    order = cand[np.argsort(score[cand], kind="stable")]   # 由小到大
    return np.sort(order[-k:]), np.sort(order[:k])


def target_notional(longs, shorts, vol: np.ndarray, beta: np.ndarray, hedge_idx: int, n: int,
                    risk: float = RISK) -> np.ndarray:
    """每個部位名目 = risk / (日波動 x sqrt(7))，多正空負；再用 BTC 反向對沖掉 beta x 名目 的總和。"""
    w = np.zeros(n)
    w[longs] = risk / (vol[longs] * math.sqrt(7))
    w[shorts] = -risk / (vol[shorts] * math.sqrt(7))
    b = np.nan_to_num(beta, nan=1.0)
    net_beta = float((w * b).sum())
    w[hedge_idx] -= net_beta / (b[hedge_idx] if b[hedge_idx] else 1.0)
    return w


def rebalance_days(days: pd.DatetimeIndex) -> np.ndarray:
    """週日的日 K（收盤 = 週一 00:00）。"""
    return np.flatnonzero(days.dayofweek == 6)


# ---------------------------------------------------------------- 組合模擬

def run(p: Panel, pool: set, cfg: dict, rng: np.random.Generator | None = None, fee: float = FEE,
        start: str = "2020-01-01") -> pd.DataFrame:
    """回傳逐 (幣, 日) 的列：symbol、side、time（當天 00:00）、exit_t（當天 23:59:59）、period（再平衡序號）、
    r（價格損益 / 0.25%）、funding_r、cost_r、r_net、hedge（是否為 BTC 對沖部位）。
    損益都是帳戶比例除以 RISK，所以 validation_gate 的 r x 0.25% 會還原成組合日報酬。"""
    n = len(p.syms)
    hedge_idx = p.syms.index(HEDGE)
    reb = rebalance_days(p.days)
    reb = reb[p.days[reb] >= pd.Timestamp(start, tz="UTC") - pd.Timedelta(days=1)]
    months = (p.days + pd.Timedelta(days=1)).strftime("%Y-%m")
    in_pool = np.array([[(s, m) in pool for s in p.syms] for m in months[reb]])
    qty = np.zeros(n)            # 持有數量（名目 / 進場價）
    out = []
    slip = np.vectorize(slippage_bps)
    for k, a in enumerate(reb):
        px = p.close_ff[a]
        old = np.nan_to_num(qty * px)
        elig = in_pool[k] & np.isfinite(p.close[a]) & (p.hist[a] >= MIN_HIST) & np.isfinite(p.vol[a]) \
            & (p.vol[a] > 0) & np.isfinite(p.beta[a])
        longs, shorts = select(scores(p, a, cfg["signal"]), elig, cfg["q"], rng)
        new = target_notional(longs, shorts, p.vol[a], p.beta[a], hedge_idx, n) if len(longs) else np.zeros(n)
        trade = np.abs(new - old)
        cost = trade * (fee + slip(p.adv[a]) / 1e4)
        with np.errstate(invalid="ignore", divide="ignore"):
            qty = np.where(new != 0, new / px, 0.0)
        b = reb[k + 1] if k + 1 < len(reb) else len(p.days) - 1
        held = np.flatnonzero(qty != 0)
        # 換手成本：新部位記在持有的第一天，只出場的記在舊部位最後一天（第 a 天）
        for s in np.flatnonzero(cost > 0):
            if qty[s] != 0 and a + 1 <= b:
                out.append((s, a + 1, k, 0.0, 0.0, -cost[s], np.sign(new[s])))
            else:
                out.append((s, a, k - 1, 0.0, 0.0, -cost[s], np.sign(old[s])))
        if not len(held) or a + 1 > b:
            continue
        seg = slice(a + 1, b + 1)
        c = p.close_ff[a:b + 1][:, held]
        pnl = qty[held] * np.diff(c, axis=0)                              # 每天的價格損益（帳戶比例）
        fund = -qty[held] * c[:-1] * p.fund[seg][:, held]                 # 多單付正費率
        days_idx = np.arange(a + 1, b + 1)
        for j, s in enumerate(held):
            for t in range(len(days_idx)):
                out.append((s, days_idx[t], k, pnl[t, j], fund[t, j], 0.0, np.sign(qty[s])))
    if not out:
        return pd.DataFrame()
    arr = np.array(out, dtype=float)
    df = pd.DataFrame({"sym_i": arr[:, 0].astype(int), "day_i": arr[:, 1].astype(int), "period": arr[:, 2].astype(int),
                       "pnl": arr[:, 3], "fund": arr[:, 4], "cost": arr[:, 5], "side": arr[:, 6].astype(int)})
    df = df.groupby(["sym_i", "day_i"], as_index=False).agg(
        period=("period", "max"), pnl=("pnl", "sum"), fund=("fund", "sum"), cost=("cost", "sum"), side=("side", "last"))
    df["symbol"] = np.array(p.syms)[df["sym_i"]]
    df["time"] = p.days[df["day_i"]]
    df["exit_t"] = df["time"] + pd.Timedelta(seconds=86_399)
    df["r"] = df["pnl"] / RISK
    df["funding_r"] = df["fund"] / RISK
    df["cost_r"] = df["cost"] / RISK
    df["r_net"] = df["r"] + df["funding_r"] + df["cost_r"]
    df["hedge"] = df["symbol"] == HEDGE
    df.loc[df["side"] == 0, "side"] = 1
    df = df[df["time"] >= pd.Timestamp(start, tz="UTC")]
    return df.sort_values(["time", "symbol"]).reset_index(drop=True)[
        ["symbol", "side", "time", "exit_t", "period", "r", "funding_r", "cost_r", "r_net", "hedge"]]


def daily(rows: pd.DataFrame, col: str = "r_net") -> pd.Series:
    """組合日報酬（帳戶比例）。"""
    s = (rows[col] * RISK).groupby(rows["time"].dt.floor("D")).sum()
    return s.reindex(pd.date_range(s.index.min(), s.index.max(), freq="D"), fill_value=0.0)


def position_weeks(rows: pd.DataFrame) -> pd.DataFrame:
    """逐 (幣, 再平衡週) 加總成「一筆交易」：R 以一週一個標準差為 1R（不含 BTC 對沖部位）。"""
    legs = rows[~rows["hedge"]]
    g = legs.groupby(["symbol", "period"])
    out = g.agg(side=("side", "last"), time=("time", "min"), exit_t=("exit_t", "max"),
                r=("r", "sum"), r_net=("r_net", "sum")).reset_index()
    return out


def sharpe(x: pd.Series) -> float:
    return float(x.mean() / x.std() * math.sqrt(365)) if x.std() > 0 else float("nan")
