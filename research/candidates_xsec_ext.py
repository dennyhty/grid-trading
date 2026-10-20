"""候選 B 延伸版：用現貨把橫斷面多空組合往前補到 2017-09，並比較三種對沖（BTC、等權幣池指數、不對沖只做金額中性）。

背景（report_candidates_20261020.md）：B（每週動能／低波動多空、對 BTC 做 beta 中性）樣本外 2024+ 夏普 +1.02，
但樣本內 2020-23 幾乎是零、2023 年虧 33%、BTC 對沖虧 139R、獲利集中在 XRP/BNB/TRX/SHIB/DOGE 多單。
兩個疑問：(1) 樣本內只有一個多空循環，太短；(2) 獲利可能只是「BTC 對沖 vs 大型山寨」這段行情。
所以這裡：

1. 現貨補長（grid_trading.spot_pit_universe）：2017-09 ~ 2019-12 用現貨時點幣池（上個月現貨 USDT 成交額前
   min(100, 可選數量)、上個月至少 20 天有量、含已下架、排除穩定幣/法幣/槓桿代幣）和現貨日線收盤價；
   2020-01 起用原本的合約時點幣池和合約日線（和 PR #53 完全相同的面板）。
   - 現貨與合約當成「不同的商品」（現貨欄位代號加 "S:" 前綴），各自算波動、beta、成交額、歷史天數，
     2020-01 第一個週日再平衡時現貨部位全部平倉（記換手成本），之後只用合約。合約幣池 2020-02 才有、
     歷史滿 61 天要到 2020-03，所以 2020-01 ~ 2020-02 空手（和 PR #53 相同），這是接縫的不連續。
   - 現貨時期沒有資金費率（設為 0）；而且當時大部分幣沒辦法放空（幣安槓桿 2019-07 才開、標的很少，
     合約 2019-09 才有 BTC），所以 2017-2019 是「紙上多空組合」，不是當時做得出來的報酬，分開報告。
   - 成本照舊：手續費每邊 0.05%（當時現貨實際是 0.075～0.1%，另做敏感度）＋依現貨 30 天日均成交額的滑價分級。
2. 對沖（hedge）三選一：
   - btc：原本的做法，各部位對 BTC 的 60 天 beta x 名目加總，用 BTC 反向對沖（現貨期用現貨 BTC、合約期用合約 BTC）。
   - index：對「等權幣池指數」做 beta 中性。指數 = 當月幣池成員日報酬的等權平均（每天再平衡，只用來估 beta）；
     各部位對這個指數的 60 天 beta x 名目加總 = 組合的 beta，用一籃子「這次合格的所有幣、每個等額」反向對沖。
     成本假設：這一籃子是真的逐幣持有（每週再平衡回等額，週內數量固定），換手成本 = 每個幣
     |新名目 - 舊名目（含漂移）| x（手續費 + 該幣滑價）、合約期每個幣照付資金費率；和多空腿的同一個幣不互相抵銷
     （偏保守）。籃子對指數的 beta 當成 1。
   - none：不對沖，只做金額中性：波動縮放後把多空兩邊的總名目都調成兩邊的平均（總曝險不變）。
3. 預先登記的網格（8 組，跑之前就固定）：
   - combo q0.3、combo q0.2 x {btc, index, none} = 6 組；
   - mom28 q0.3、lowvol q0.3 只配 index 對沖 = 2 組（拆解 combo 兩個成分）。
   挑主設定只看新的樣本內 2017-09 ~ 2023-12 扣成本日報酬夏普；樣本外 2024-01 起不變。
   鄰近設定（參數高原）：combo 換另一個 q（同對沖）；mom28/lowvol 用同對沖的 combo q0.3（較弱，事先說明）。

選股、部位大小（一週一個標準差 = 帳戶 0.25%）、每週日收盤再平衡、合格條件（幣池、61 天歷史、30 天波動、
60 天 beta）都和 research/candidates_xsec.py 相同；btc 對沖在合約期的結果和 PR #53 逐列相同（測試有檢查）。
"""

from __future__ import annotations

import math
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from grid_trading.costs import slippage_bps
from research.candidates_common import FEE, RISK
from research.candidates_xsec import (HEDGE, MIN_HIST, Panel, rebalance_days, rolling_beta, scores, select,
                                      target_notional)

FAMILY = "xsec_ext"
SPOT_PREFIX = "S:"
SPOT_HEDGE = SPOT_PREFIX + HEDGE
IDX = "EWIDX"                      # 等權指數對沖籃子（合成代號）
SPLICE_MONTH = "2020-01"           # 這個月（含）起用合約幣池
START = pd.Timestamp("2017-09-01", tz="UTC")
IS_END = pd.Timestamp("2024-01-01", tz="UTC")
OLD_IS_START = pd.Timestamp("2020-01-01", tz="UTC")
HEDGES = ("btc", "index", "none")
EXT_PANEL_CACHE = Path("data/candidates/xsec_ext_panel.pkl")


def configs() -> dict[str, dict]:
    """預先登記的 8 組設定。名稱：分數|q|對沖。"""
    out = {}
    for h in HEDGES:
        for q in (0.3, 0.2):
            out[f"combo|q{q:g}|{h}"] = {"signal": "combo", "q": q, "hedge": h}
    for sig in ("mom28", "lowvol"):
        out[f"{sig}|q0.3|index"] = {"signal": sig, "q": 0.3, "hedge": "index"}
    return out


def neighbors(name: str) -> list[str]:
    """combo：同對沖、另一個 q；mom28/lowvol：同對沖的 combo q0.3。"""
    p = configs()[name]
    if p["signal"] == "combo":
        q2 = 0.2 if p["q"] == 0.3 else 0.3
        return [f"combo|q{q2:g}|{p['hedge']}"]
    return [f"combo|q0.3|{p['hedge']}"]


def era_of_month(month: str) -> str:
    return "spot" if month < SPLICE_MONTH else "fut"


def coin_key(symbol: str) -> str:
    """去掉現貨前綴：XRP 的現貨期和合約期當成同一個幣（集中度檢查用）。"""
    return symbol[len(SPOT_PREFIX):] if symbol.startswith(SPOT_PREFIX) else symbol


# ---------------------------------------------------------------- 面板

@dataclass
class ExtPanel(Panel):
    """Panel（見 candidates_xsec）加上：beta_idx（對等權幣池指數的 60 天 beta）、idx_ret（指數日對數報酬）、
    is_spot（每個欄位是不是現貨）。beta 是對 BTC 的 beta：現貨欄位對現貨 BTC、合約欄位對合約 BTC。"""

    beta_idx: np.ndarray = None
    idx_ret: np.ndarray = None
    is_spot: np.ndarray = None


def spot_frames(spot_daily: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """現貨長表 → (收盤, 成交額) 寬表，欄位加 "S:" 前綴；成交量 0 的日子當成沒有資料。"""
    d = spot_daily.copy()
    d["time"] = pd.to_datetime(d["time"], utc=True).dt.floor("D")
    d.loc[d["volume"] <= 0, ["close", "quote_volume"]] = np.nan
    d["symbol"] = SPOT_PREFIX + d["symbol"]
    close = d.pivot_table(index="time", columns="symbol", values="close", aggfunc="last", dropna=False)
    qv = d.pivot_table(index="time", columns="symbol", values="quote_volume", aggfunc="last", dropna=False)
    return close.sort_index(), qv.reindex(close.index)[close.columns]


def pool_mask(days: pd.DatetimeIndex, syms: list, pool: set) -> np.ndarray:
    """mask[t, j] = 第 t 天所在月份 (syms[j], 月) 在 pool 裡。"""
    months = days.strftime("%Y-%m")
    col = {s: j for j, s in enumerate(syms)}
    mask = np.zeros((len(days), len(syms)), bool)
    by_month: dict[str, list[int]] = {}
    for s, m in pool:
        if s in col:
            by_month.setdefault(m, []).append(col[s])
    for m, cols in by_month.items():
        rows = np.flatnonzero(months == m)
        if len(rows):
            mask[np.ix_(rows, cols)] = True
    return mask


def ew_index_returns(close: pd.DataFrame, mask: np.ndarray) -> pd.Series:
    """等權指數的日對數報酬：當天所在月份的幣池成員裡、當天有報酬的幣，簡單報酬取平均後取 log。沒有成員為 NaN。"""
    with np.errstate(invalid="ignore", divide="ignore"):
        r = (close / close.shift(1) - 1).to_numpy()
    r = np.where(mask & np.isfinite(r), r, np.nan)
    with np.errstate(invalid="ignore"):
        cnt = np.isfinite(r).sum(axis=1)
        m = np.where(cnt > 0, np.nansum(r, axis=1) / np.maximum(cnt, 1), np.nan)
    return pd.Series(np.log1p(m), index=close.index)


def build_ext_panel(fut: Panel, spot_daily: pd.DataFrame, trade_pool: set, index_pool: set,
                    cache: bool = False) -> ExtPanel:
    """把合約面板（PR #53 的 Panel，原封不動）和現貨面板接起來。

    trade_pool：交易用的時點幣池（現貨代號帶 "S:"、只到 2019-12；合約 2020-01 起）。
    index_pool：估等權指數 beta 用的成員（現貨到 2020-01、合約 2020-02 起，讓 2020-01 也有指數）。"""
    if cache and EXT_PANEL_CACHE.exists():
        with EXT_PANEL_CACHE.open("rb") as f:
            p = pickle.load(f)
        if p.syms[:len(fut.syms)] == list(fut.syms):
            return p
    s_close, s_qv = spot_frames(spot_daily)
    days = pd.date_range(min(s_close.index.min(), fut.days.min()), max(s_close.index.max(), fut.days.max()),
                         freq="D")
    s_close, s_qv = s_close.reindex(days), s_qv.reindex(days)
    s_syms = list(s_close.columns)
    if SPOT_HEDGE not in s_syms:
        raise ValueError("現貨資料沒有 BTCUSDT")
    ret_s = np.log(s_close / s_close.shift(1))
    s_vol = ret_s.rolling(30, min_periods=20).std()
    s_beta = rolling_beta(ret_s, ret_s[SPOT_HEDGE])
    s_adv = s_qv.rolling(30, min_periods=5).mean()
    s_hist = s_close.notna().cumsum()

    # 合約區塊：直接沿用 PR #53 面板的數值（2020-01-01 以前補 NaN / 0），保證合約期完全相同
    pos = days.get_indexer(fut.days)
    if (pos < 0).any():
        raise ValueError("合約面板的日期不在合併後的日期裡")

    def place(arr, fill):
        out = np.full((len(days), arr.shape[1]), fill, dtype=float)
        out[pos] = arr
        return out

    f_close = place(fut.close, np.nan)
    f_close_ff = place(fut.close_ff, np.nan)
    f_close_ff[pos[-1] + 1:] = fut.close_ff[-1]
    f_hist = place(fut.hist, 0.0)
    f_hist[pos[-1] + 1:] = fut.hist[-1]
    syms = list(fut.syms) + s_syms
    close = np.hstack([f_close, s_close.to_numpy()])
    close_df = pd.DataFrame(close, index=days, columns=syms)
    idx_mask = pool_mask(days, syms, index_pool)
    idx_ret = ew_index_returns(close_df, idx_mask)
    ret_all = np.log(close_df / close_df.shift(1))
    beta_idx = rolling_beta(ret_all, idx_ret).to_numpy()
    p = ExtPanel(days=days, syms=syms, close=close,
                 close_ff=np.hstack([f_close_ff, s_close.ffill().to_numpy()]),
                 adv=np.hstack([place(fut.adv, np.nan), s_adv.to_numpy()]),
                 vol=np.hstack([place(fut.vol, np.nan), s_vol.to_numpy()]),
                 beta=np.hstack([place(fut.beta, np.nan), s_beta.to_numpy()]),
                 hist=np.hstack([f_hist, s_hist.to_numpy()]),
                 fund=np.hstack([place(fut.fund, 0.0), np.zeros((len(days), len(s_syms)))]),
                 beta_idx=beta_idx, idx_ret=idx_ret.to_numpy(),
                 is_spot=np.array([s.startswith(SPOT_PREFIX) for s in syms]))
    if cache:
        EXT_PANEL_CACHE.parent.mkdir(parents=True, exist_ok=True)
        with EXT_PANEL_CACHE.open("wb") as f:
            pickle.dump(p, f)
    return p


# ---------------------------------------------------------------- 部位

def leg_notional(longs, shorts, vol: np.ndarray, n: int, risk: float = RISK) -> np.ndarray:
    """只有多空腿的名目（和 target_notional 相同的波動縮放，不含對沖）。"""
    w = np.zeros(n)
    w[longs] = risk / (vol[longs] * math.sqrt(7))
    w[shorts] = -risk / (vol[shorts] * math.sqrt(7))
    return w


def dollar_neutral(w: np.ndarray) -> np.ndarray:
    """把多空兩邊的總名目都調成兩邊的平均（總曝險不變，淨名目 = 0）。某一邊是空的就原樣回傳。"""
    w = w.copy()
    L, S = w[w > 0].sum(), -w[w < 0].sum()
    if L <= 0 or S <= 0:
        return w
    g = (L + S) / 2
    w[w > 0] *= g / L
    w[w < 0] *= g / S
    return w


def index_hedge(w: np.ndarray, beta_idx: np.ndarray, basket: np.ndarray, n: int) -> np.ndarray:
    """等權指數對沖籃子的名目：組合 beta = sum(名目 x 對指數 beta)（beta 缺值當 1），
    籃子裡每個幣 = -組合 beta / 籃子幣數。"""
    h = np.zeros(n)
    if len(basket) == 0:
        return h
    net = float((w * np.nan_to_num(beta_idx, nan=1.0)).sum())
    h[basket] = -net / len(basket)
    return h


# ---------------------------------------------------------------- 組合模擬

def run_ext(p: ExtPanel, pool: set, cfg: dict, rng: np.random.Generator | None = None, fee: float = FEE,
            start: pd.Timestamp | str = START, spot_fee: float | None = None) -> pd.DataFrame:
    """回傳逐 (商品, 日) 的列，欄位同 candidates_xsec.run，另加 era（spot / fut，依再平衡所在月份）。
    cfg：signal、q、hedge（btc / index / none）。等權指數對沖籃子合成一個代號 EWIDX（hedge=True）。
    spot_fee：現貨期另用的手續費（敏感度用，預設同 fee）。"""
    hedge = cfg.get("hedge", "btc")
    if hedge not in HEDGES:
        raise ValueError(hedge)
    n = len(p.syms)
    btc_idx = {"fut": p.syms.index(HEDGE), "spot": p.syms.index(SPOT_HEDGE)}
    start = pd.Timestamp(start)
    start = start.tz_localize("UTC") if start.tz is None else start
    reb = rebalance_days(p.days)
    reb = reb[p.days[reb] >= start - pd.Timedelta(days=1)]
    months = (p.days + pd.Timedelta(days=1)).strftime("%Y-%m")
    eras = [era_of_month(m) for m in months[reb]]
    in_pool = pool_mask(p.days[reb] + pd.Timedelta(days=1), p.syms, pool)
    qty = np.zeros(n)            # 多空腿（btc 對沖時含 BTC 對沖）的持有數量
    hq = np.zeros(n)             # 等權指數籃子的持有數量
    out, hout = [], []           # hout：籃子的 (日, 週期, 損益, 資金費, 成本, 方向)
    slip = np.vectorize(slippage_bps)
    # 現貨商品可另給手續費（敏感度）；合約商品一律用 fee
    fee_vec = np.where(p.is_spot, fee if spot_fee is None else spot_fee, fee)
    for k, a in enumerate(reb):
        px = p.close_ff[a]
        old = np.nan_to_num(qty * px)
        hold = np.nan_to_num(hq * px)
        elig = in_pool[k] & np.isfinite(p.close[a]) & (p.hist[a] >= MIN_HIST) & np.isfinite(p.vol[a]) \
            & (p.vol[a] > 0) & np.isfinite(p.beta[a]) & np.isfinite(p.beta_idx[a])
        longs, shorts = select(scores(p, a, cfg["signal"]), elig, cfg["q"], rng)
        hnew = np.zeros(n)
        if not len(longs):
            new = np.zeros(n)
        elif hedge == "btc":
            new = target_notional(longs, shorts, p.vol[a], p.beta[a], btc_idx[eras[k]], n)
        elif hedge == "index":
            new = leg_notional(longs, shorts, p.vol[a], n)
            hnew = index_hedge(new, p.beta_idx[a], np.flatnonzero(elig), n)
        else:
            new = dollar_neutral(leg_notional(longs, shorts, p.vol[a], n))
        unit = fee_vec + slip(p.adv[a]) / 1e4
        cost = np.abs(new - old) * unit
        hcost = float((np.abs(hnew - hold) * unit).sum())
        with np.errstate(invalid="ignore", divide="ignore"):
            qty = np.where(new != 0, new / px, 0.0)
            hq = np.where(hnew != 0, hnew / px, 0.0)
        b = reb[k + 1] if k + 1 < len(reb) else len(p.days) - 1
        held = np.flatnonzero(qty != 0)
        for s in np.flatnonzero(cost > 0):
            if qty[s] != 0 and a + 1 <= b:
                out.append((s, a + 1, k, 0.0, 0.0, -cost[s], np.sign(new[s])))
            else:
                out.append((s, a, k - 1, 0.0, 0.0, -cost[s], np.sign(old[s])))
        hsum_new = hnew.sum()
        if hcost > 0:
            if hsum_new != 0 and a + 1 <= b:
                hout.append((a + 1, k, 0.0, 0.0, -hcost, np.sign(hsum_new)))
            else:
                hout.append((a, k - 1, 0.0, 0.0, -hcost, np.sign(hold.sum()) or 1.0))
        if a + 1 > b:
            continue
        seg = slice(a + 1, b + 1)
        days_idx = np.arange(a + 1, b + 1)
        if len(held):
            c = p.close_ff[a:b + 1][:, held]
            pnl = qty[held] * np.diff(c, axis=0)
            fund = -qty[held] * c[:-1] * p.fund[seg][:, held]
            for j, s in enumerate(held):
                for t in range(len(days_idx)):
                    out.append((s, days_idx[t], k, pnl[t, j], fund[t, j], 0.0, np.sign(qty[s])))
        hb = np.flatnonzero(hq != 0)
        if len(hb):
            c = p.close_ff[a:b + 1][:, hb]
            hp = (hq[hb] * np.diff(c, axis=0)).sum(axis=1)
            hf = (-hq[hb] * c[:-1] * p.fund[seg][:, hb]).sum(axis=1)
            for t in range(len(days_idx)):
                hout.append((days_idx[t], k, hp[t], hf[t], 0.0, np.sign(hsum_new)))
    cols = ["sym_i", "day_i", "period", "pnl", "fund", "cost", "side"]
    frames = []
    if out:
        frames.append(pd.DataFrame(np.array(out, dtype=float), columns=cols))
    if hout:
        h = np.array(hout, dtype=float)
        frames.append(pd.DataFrame({"sym_i": float(n), "day_i": h[:, 0], "period": h[:, 1], "pnl": h[:, 2],
                                    "fund": h[:, 3], "cost": h[:, 4], "side": h[:, 5]}))
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    for c in ("sym_i", "day_i", "period", "side"):
        df[c] = df[c].astype(int)
    df = df.groupby(["sym_i", "day_i"], as_index=False).agg(
        period=("period", "max"), pnl=("pnl", "sum"), fund=("fund", "sum"), cost=("cost", "sum"), side=("side", "last"))
    names = np.array(list(p.syms) + [IDX])
    df["symbol"] = names[df["sym_i"]]
    df["time"] = p.days[df["day_i"]]
    df["exit_t"] = df["time"] + pd.Timedelta(seconds=86_399)
    df["r"] = df["pnl"] / RISK
    df["funding_r"] = df["fund"] / RISK
    df["cost_r"] = df["cost"] / RISK
    df["r_net"] = df["r"] + df["funding_r"] + df["cost_r"]
    if hedge == "btc":
        df["hedge"] = df["symbol"].isin([HEDGE, SPOT_HEDGE])
    else:
        df["hedge"] = df["symbol"] == IDX
    df.loc[df["side"] == 0, "side"] = 1
    per_era = np.array(eras + [eras[-1]] if eras else ["fut"])
    df["era"] = per_era[np.clip(df["period"].to_numpy(), 0, len(per_era) - 1)]
    df = df[df["time"] >= start]
    return df.sort_values(["time", "symbol"]).reset_index(drop=True)[
        ["symbol", "side", "time", "exit_t", "period", "r", "funding_r", "cost_r", "r_net", "hedge", "era"]]


# ---------------------------------------------------------------- 統計

def daily_ret(rows: pd.DataFrame, col: str = "r_net", lo=None, hi=None) -> pd.Series:
    """組合日報酬（帳戶比例）；lo/hi 給的話補 0 補滿這段期間（含 lo，不含 hi）。"""
    s = (rows[col] * RISK).groupby(rows["time"].dt.floor("D")).sum() if len(rows) else pd.Series(dtype=float)
    a = pd.Timestamp(lo) if lo is not None else s.index.min()
    b = (pd.Timestamp(hi) - pd.Timedelta(days=1)) if hi is not None else s.index.max()
    return s.reindex(pd.date_range(a, b, freq="D"), fill_value=0.0)


def perf(day: pd.Series) -> dict:
    """夏普（年化，365）、年化報酬（日報酬平均 x 365）、年化波動、最大回撤（複利淨值）、週報酬 t、天數。"""
    if len(day) < 2 or day.std() == 0:
        return {"days": len(day), "sharpe": float("nan"), "ann_ret_pct": float("nan"), "ann_vol_pct": float("nan"),
                "max_dd_pct": float("nan"), "t_weekly": float("nan")}
    eq = np.cumprod(1 + day.to_numpy())
    w = day.resample("W").sum()
    return {"days": len(day), "sharpe": float(day.mean() / day.std() * math.sqrt(365)),
            "ann_ret_pct": float(day.mean() * 365 * 100), "ann_vol_pct": float(day.std() * math.sqrt(365) * 100),
            "max_dd_pct": float((eq / np.maximum.accumulate(eq) - 1).min() * 100),
            "t_weekly": float(w.mean() / w.std() * math.sqrt(len(w))) if w.std() > 0 else float("nan")}


def first_active(rows: pd.DataFrame) -> pd.Timestamp:
    return rows["time"].min().floor("D")
