"""階段 2 候選策略的共用工具：日線資料、時點幣池、出場模擬、帳本、隨機進場、和實盤維加斯比較。

給 research/candidates_daily_trend.py、candidates_squeeze.py、candidates_xsec.py 和 candidates_run.py 用。

- 日線：用已快取的幣安合約 4h K 線（grid_trading.pit_universe.fetch_symbol）在 UTC 00:00 重取樣成 1D，
  不另外下載；最後一天不滿 6 根 4h 的丟掉。結果快取在 data/candidates/daily.pkl（data/ 不進 git）。
- 幣池：grid_trading.pit_universe.membership（上個月成交額前 100 名），以「進場時間所在月份」判斷。
- 出場：重用 research.stop_research.sim_many（numba，和實盤 tracker 同一套），手續費每邊 0.05%，
  滑價和資金費率之後由 grid_trading.costs.apply_costs 統一加（slip 參數固定 0，避免重複扣）。
- 同一個幣同時只一張（不分多空），新訊號的 K 棒必須在前一張出場那根之後或同一根（i >= j），
  比共用的 busy_filter 嚴格一根（那個邊界問題見 report_honest_backtest_20261020.md）。
"""

from __future__ import annotations

import math
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from grid_trading.costs import apply_costs
from grid_trading.pit_universe import coin_of, fetch_symbol, membership

FEE = 0.0005
RISK = 0.0025
IS_END = pd.Timestamp("2024-01-01", tz="UTC")   # 樣本內 2020-2023，樣本外 2024 起
START = pd.Timestamp("2020-01-01", tz="UTC")
DAILY_CACHE = Path("data/candidates/daily.pkl")
DAY = pd.Timedelta(days=1)


# ---------------------------------------------------------------- 日線資料

def to_daily(df4h: pd.DataFrame) -> pd.DataFrame:
    """4h K 線 → UTC 日線：open/high/low/close/volume、qv（成交額 USDT，用每根 4h 的 volume*close 加總）、
    n（這天有幾根 4h）。time 是當天 00:00（開盤時間）。最後一天不滿 6 根視為還沒收完，丟掉。"""
    if df4h is None or len(df4h) == 0:
        return pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume", "qv", "n"])
    d = df4h.copy()
    d["qv"] = d["volume"] * d["close"]
    g = d.set_index("time").resample("1D")
    out = pd.DataFrame({"open": g["open"].first(), "high": g["high"].max(), "low": g["low"].min(),
                        "close": g["close"].last(), "volume": g["volume"].sum(), "qv": g["qv"].sum(),
                        "n": g["close"].count()})
    out = out[out["n"] > 0]
    if len(out) and out["n"].iloc[-1] < 6:
        out = out.iloc[:-1]
    return out.rename_axis("time").reset_index()


def pool_table() -> pd.DataFrame:
    mem = membership(n=100)
    mem["month"] = mem["month"].astype(str)
    return mem


def load_daily(symbols: list[str] | None = None, refresh: bool = False) -> dict[str, pd.DataFrame]:
    """所有曾經進過時點幣池的合約的日線（快取）。"""
    if DAILY_CACHE.exists() and not refresh:
        with DAILY_CACHE.open("rb") as f:
            data = pickle.load(f)
        if symbols is None or set(symbols) <= set(data):
            return data if symbols is None else {s: data[s] for s in symbols}
    if symbols is None:
        symbols = sorted(set(pool_table()["symbol"]))
    data = {}
    for s in symbols:
        d = to_daily(fetch_symbol(s, "4h"))
        if len(d) > 30:
            data[s] = d
    DAILY_CACHE.parent.mkdir(parents=True, exist_ok=True)
    with DAILY_CACHE.open("wb") as f:
        pickle.dump(data, f)
    return data


def wilder_atr(high, low, close, n: int = 20) -> np.ndarray:
    """Wilder ATR（和 grid_trading.indicators.atr 同算法）。"""
    h, l, c = (pd.Series(np.asarray(x, float)) for x in (high, low, close))
    prev = c.shift(1)
    tr = pd.concat([h - l, (h - prev).abs(), (l - prev).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean().to_numpy()


def adv30(qv) -> np.ndarray:
    """近 30 天平均日成交額（含當天；訊號在當天收盤才用，不偷看）。"""
    return pd.Series(np.asarray(qv, float)).rolling(30, min_periods=5).mean().to_numpy()


# ---------------------------------------------------------------- 出場模擬

@dataclass(frozen=True)
class ExitPlan:
    """出場規則。stop_atr：初始止損 = 幾倍 ATR（也是 1R）；tps/portions：TP 的 R 倍數和平倉比例；
    trail_atr：TP 全碰到後（沒有 TP 就從進場起）剩餘倉位的追蹤止損 ATR 倍數，0 = 不追蹤；
    chan_exit：收盤跌破（空單為突破）前 chan_exit 天的最低（最高）價就收盤出場，0 = 不用；
    max_bars：最多持有幾根日 K。TP1 後保本、TP2 後移到 TP1（實盤規則，protect=0）。"""

    stop_atr: float
    tps: tuple = ()
    portions: tuple = ()
    trail_atr: float = 0.0
    chan_exit: int = 0
    max_bars: int = 365


LADDER = dict(tps=(1.0, 3.0, 5.0), portions=(0.25, 0.25, 0.25), trail_atr=3.0)   # 實盤 1/3/5R 各平 1/4
HALF_1R = dict(tps=(1.0,), portions=(0.5,), trail_atr=3.0)                        # 1R 平一半、保本、追蹤 3 ATR


def channel_levels(high, low, n: int) -> tuple[np.ndarray, np.ndarray]:
    """前 n 天（不含當天）的最低價、最高價：收盤出場用。"""
    h, l = pd.Series(np.asarray(high, float)), pd.Series(np.asarray(low, float))
    return l.rolling(n).min().shift(1).to_numpy(), h.rolling(n).max().shift(1).to_numpy()


def simulate(arr: dict, idx: np.ndarray, ds: np.ndarray, plan: ExitPlan, fee: float = FEE) -> dict:
    """訊號 K 棒收盤進場、下一根開始算出場。arr 需 open/high/low/close/atr（numpy）。
    回傳 r（含手續費，不含滑價和資金費率）、hit、j（出場 K 棒）、kind（見 sim_many）、risk_pct。"""
    from research.stop_research import MIN_STOP_PCT, sim_many

    idx = np.asarray(idx, np.int64)
    ds = np.asarray(ds, np.int64)
    n = len(idx)
    if n == 0:
        e = np.zeros(0)
        return {"r": e, "hit": e.astype(np.int64), "j": e.astype(np.int64), "kind": e.astype(np.int64), "risk_pct": e}
    price = arr["close"][idx]
    atr = arr["atr"][idx]
    risk = np.maximum(plan.stop_atr * atr, MIN_STOP_PCT * price)
    stop0 = price - ds * risk
    if plan.tps:
        tps = np.stack([price + ds * k * risk for k in plan.tps], axis=1)
    else:
        tps = np.zeros((n, 0))
    portions = np.array(plan.portions, float) if plan.portions else np.zeros(0)
    zero = np.zeros(n)
    m = len(arr["close"])
    if plan.chan_exit > 0:
        inv_lo, inv_hi = channel_levels(arr["high"], arr["low"], plan.chan_exit)
        inv_lo = np.where(np.isfinite(inv_lo), inv_lo, -np.inf)
        inv_hi = np.where(np.isfinite(inv_hi), inv_hi, np.inf)
        inv_mode = 1
    else:
        inv_lo = inv_hi = np.zeros(m)
        inv_mode = 0
    isc = np.ones(m, np.bool_)
    r, hit, j, kind, _ = sim_many(arr["open"], arr["high"], arr["low"], arr["close"], isc, idx + 1, ds, price, risk,
                                  stop0, tps, portions, 0, plan.trail_atr * atr, fee, 0.0, plan.max_bars, 0, 0, zero,
                                  0.0, 0.0, zero, inv_mode, inv_lo, inv_hi, True)
    return {"r": r, "hit": hit, "j": j, "kind": kind, "risk_pct": risk / price}


def trade_frame(sym: str, d: pd.DataFrame, idx, ds, sim: dict, adv: np.ndarray) -> pd.DataFrame:
    """把模擬結果整理成逐筆交易表。time = 訊號日收盤（= 隔天 00:00）；exit_t：收盤規則出場記當天收盤前一秒，
    盤中止損/TP 記當天 12:00（出場日的歸屬才對；資金費率會少算或多算不到半天）。"""
    t = pd.DatetimeIndex(pd.to_datetime(d["time"], utc=True))
    n = len(t)
    j = np.minimum(sim["j"], n - 1)
    close_exit = np.isin(sim["kind"], (4, 5))
    off = pd.to_timedelta(np.where(close_exit, 86_399, 12 * 3600), unit="s")
    return pd.DataFrame({"symbol": sym, "coin": coin_of(sym), "i": np.asarray(idx), "side": np.asarray(ds),
                         "time": t[idx] + DAY, "exit_t": t[j] + off,
                         "r": sim["r"], "hit": sim["hit"], "kind": sim["kind"], "j": sim["j"],
                         "risk_pct": sim["risk_pct"], "adv_usdt": adv[idx]})


def one_at_a_time(tr: pd.DataFrame) -> pd.DataFrame:
    """同一個幣同時只一張（不分方向）：依 (幣, i) 排序，前一張出場那根（j）之前的訊號跳過；
    資料結束還沒出場（kind -1）的丟掉，也不擋後面。"""
    if tr.empty:
        return tr
    tr = tr[tr["kind"] != -1].sort_values(["symbol", "i"]).reset_index(drop=True)
    keep = np.zeros(len(tr), bool)
    busy, prev = -1, None
    for k, (s, i, j) in enumerate(zip(tr["symbol"].to_numpy(), tr["i"].to_numpy(), tr["j"].to_numpy())):
        if s != prev:
            busy, prev = -1, s
        if i < busy:
            continue
        keep[k] = True
        busy = j
    return tr[keep].reset_index(drop=True)


def in_pool(tr: pd.DataFrame, mem: pd.DataFrame) -> np.ndarray:
    pit = set(zip(mem["symbol"], mem["month"]))
    return np.array([(s, m) in pit for s, m in zip(tr["symbol"], pd.to_datetime(tr["time"], utc=True).dt.strftime("%Y-%m"))],
                    bool)


def finalize(tr: pd.DataFrame, mem: pd.DataFrame, costs: bool = True) -> pd.DataFrame:
    """篩時點幣池 + 2020 起 → 一個幣一次一張 → 加滑價和資金費率（r_net）。"""
    if tr.empty:
        return tr
    tr = tr.copy()
    tr["time"] = pd.to_datetime(tr["time"], utc=True)
    tr["exit_t"] = pd.to_datetime(tr["exit_t"], utc=True)
    tr = tr[(tr["time"] >= START) & in_pool(tr, mem)]
    tr = one_at_a_time(tr)
    return apply_costs(tr) if costs else tr


# ---------------------------------------------------------------- 統計

def stats(tr: pd.DataFrame, rcol: str = "r_net", risk: float = RISK) -> dict:
    """筆數、勝率、平均 R、群集（進場日+方向）標準誤與 t、日報酬年化夏普、多空拆分、最大回撤。"""
    from research.stop_research import cluster_key, cluster_stats

    if tr is None or len(tr) == 0:
        return {"trades": 0}
    r = tr[rcol].to_numpy(float)
    mean, se, *_ = cluster_stats(r, cluster_key(tr))
    day = daily_series(tr, rcol, risk)
    sh = float(day.mean() / day.std() * math.sqrt(365)) if day.std() > 0 else float("nan")
    eq = np.cumprod(1 + risk * tr.sort_values("exit_t")[rcol].to_numpy(float))
    side = tr["side"].to_numpy()
    return {"trades": len(r), "win_rate": float((r > 0).mean()), "exp_r": float(mean), "se_cluster": float(se),
            "t": float(mean / se) if se > 0 else float("nan"), "sharpe": sh,
            "long_n": int((side > 0).sum()), "long_r": float(r[side > 0].mean()) if (side > 0).any() else float("nan"),
            "short_n": int((side < 0).sum()), "short_r": float(r[side < 0].mean()) if (side < 0).any() else float("nan"),
            "max_dd_pct": float((eq / np.maximum.accumulate(eq) - 1).min() * 100),
            "hold_days": float(((tr["exit_t"] - tr["time"]).dt.total_seconds() / 86400).mean())}


def daily_series(tr: pd.DataFrame, rcol: str = "r_net", risk: float = RISK, start=None, end=None) -> pd.Series:
    """每筆 R x 風險，依出場日加總（和 validation_gate.daily_returns 相同）；沒交易的日子補 0。"""
    from research.validation_gate import daily_returns

    df = tr.copy()
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df["exit_t"] = pd.to_datetime(df["exit_t"], utc=True)
    return daily_returns(df, rcol, risk, start, end)


def split(tr: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    t = pd.to_datetime(tr["time"], utc=True)
    return tr[t < IS_END], tr[t >= IS_END]


# ---------------------------------------------------------------- 實盤維加斯帳本、相關性

def vegas_book(path: str = "output/pit_trades.csv.gz") -> pd.DataFrame:
    """實盤維加斯（4H+2H）時點幣池帳本，r_net 已扣滑價與資金費率。"""
    from research.pit_backtest import book

    tr = pd.read_csv(path, parse_dates=["time", "exit_t"])
    return book(tr, "in_pit")


def compare_with(base_daily: pd.Series, cand_daily: pd.Series) -> dict:
    """和實盤的關係：日／週／月報酬相關、實盤最差 5% 日子裡候選的平均、合併後夏普（各自 0.25% 風險直接相加）。"""
    lo = max(base_daily.index.min(), cand_daily.index.min())
    hi = min(base_daily.index.max(), cand_daily.index.max())
    a = base_daily.loc[lo:hi]
    b = cand_daily.reindex(a.index, fill_value=0.0)
    out = {"days": len(a), "corr_daily": float(a.corr(b))}
    for f, k in (("W", "corr_weekly"), ("ME", "corr_monthly")):
        out[k] = float(a.resample(f).sum().corr(b.resample(f).sum()))
    worst = a <= a.quantile(0.05)
    best = a >= a.quantile(0.95)
    out["cand_on_base_worst5pct_bp"] = float(b[worst].mean() * 1e4)
    out["cand_on_base_best5pct_bp"] = float(b[best].mean() * 1e4)
    out["cand_other_days_bp"] = float(b[~worst & ~best].mean() * 1e4)
    sh = lambda x: float(x.mean() / x.std() * math.sqrt(365)) if x.std() > 0 else float("nan")  # noqa: E731
    out["sharpe_base"], out["sharpe_cand"], out["sharpe_combined"] = sh(a), sh(b), sh(a + b)
    return out


# ---------------------------------------------------------------- 隨機進場基準

def random_entries(trades: pd.DataFrame, data: dict[str, pd.DataFrame], mem: pd.DataFrame, plan_of, draws: int = 3,
                   seed: int = 7, min_bar: int = 60) -> pd.DataFrame:
    """隨機進場基準：每筆策略交易換成「同一個幣、同一年、同方向、在幣池內」的隨機一天收盤進場，出場規則相同。
    plan_of(symbol) 回傳 (arr, ExitPlan)。重複 draws 次合併（每次各自套一個幣一次一張和成本）。"""
    rng = np.random.default_rng(seed)
    pit = set(zip(mem["symbol"], mem["month"]))
    parts = []
    for k in range(draws):
        rows = []
        for sym, g in trades.groupby("symbol"):
            d = data[sym]
            arr, plan = plan_of(sym)
            t = pd.to_datetime(d["time"], utc=True)
            ent = t + DAY
            ok = np.array([(sym, m) in pit for m in ent.dt.strftime("%Y-%m")]) & (np.arange(len(d)) >= min_bar) \
                & np.isfinite(arr["atr"]) & (np.arange(len(d)) + 1 < len(d))
            years = ent.dt.year.to_numpy()
            idx, ds = [], []
            for y, side in zip(pd.to_datetime(g["time"], utc=True).dt.year, g["side"]):
                cand = np.flatnonzero(ok & (years == y))
                if len(cand):
                    idx.append(int(rng.choice(cand)))
                    ds.append(int(side))
            if not idx:
                continue
            idx, ds = np.array(idx), np.array(ds)
            o = np.argsort(idx, kind="stable")
            idx, ds = idx[o], ds[o]
            sim = simulate(arr, idx, ds, plan)
            rows.append(trade_frame(sym, d, idx, ds, sim, adv30(d["qv"].to_numpy())))
        if rows:
            tr = pd.concat(rows, ignore_index=True)
            tr["symbol_draw"] = k
            parts.append(finalize(tr, mem))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


# ---------------------------------------------------------------- 網格與挑選

@dataclass
class GridResult:
    """一個家族的網格結果：每組設定的交易表、樣本內／外統計、挑出的主設定。"""

    family: str
    configs: dict = field(default_factory=dict)      # name -> params dict
    trades: dict = field(default_factory=dict)       # name -> 帳本 DataFrame
    table: pd.DataFrame | None = None
    headline: str | None = None


def grid_table(res: GridResult) -> pd.DataFrame:
    rows = []
    for name, tr in res.trades.items():
        ins, oos = split(tr)
        si, so, sa = stats(ins), stats(oos), stats(tr)
        rows.append({"config": name, **{f"is_{k}": v for k, v in si.items()},
                     **{f"oos_{k}": v for k, v in so.items()}, **{f"all_{k}": v for k, v in sa.items()}})
    return pd.DataFrame(rows)


def pick_headline(table: pd.DataFrame, min_trades: int = 100, col: str = "is_t") -> str:
    """只用樣本內挑主設定：樣本內筆數 >= min_trades 的設定裡 col（預設群集 t）最高。"""
    ok = table[table["is_trades"] >= min_trades]
    if ok.empty:
        ok = table
    return str(ok.sort_values(col, ascending=False)["config"].iloc[0])
