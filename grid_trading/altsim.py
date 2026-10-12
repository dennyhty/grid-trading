"""其他主流策略（強弱輪動、均值回歸、VWAP/成交量分布）共用的回測工具。

- 資料：幣安 1H 月檔（grid_trading/binance_history.py），合成 2H/4H/日線。
- 出場模擬 sim() 和 grid_trading/signals/tracker.py 的規則一致（同一根先停損、TP 各平 1/3、TP1 後保本、
  TP2 後移到 TP1、超過 max_bars 收盤平倉），只是改成純數字迴圈，回測幾萬筆才跑得動。
  單一目標價的策略把 3 個 TP 設成同一價位即可。
- 隨機進場對照：同幣同年份隨機挑 K 線，方向和止損距離（占價格的比例）、出場規則都和真實交易一樣。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .binance_history import load as binance_load
from .history import resample
from .universe import TOP100

FEE = 0.0005
SLIP = 0.0005
RISK = 0.0025  # 每筆單願意賠總資金的比例
TRAIN = (pd.Timestamp("2020-01-01", tz="UTC"), pd.Timestamp("2024-01-01", tz="UTC"))
VALID = (pd.Timestamp("2024-01-01", tz="UTC"), pd.Timestamp("2026-10-01", tz="UTC"))
BAR_HOURS = {"1h": 1, "2h": 2, "4h": 4, "1d": 24}


@dataclass
class Bars:
    sym: str
    tf: str
    t: np.ndarray        # 開盤時間，ns
    o: np.ndarray
    h: np.ndarray
    l: np.ndarray
    c: np.ndarray
    v: np.ndarray
    hl: list             # h、l、c 的 list 版本，sim() 用
    ll: list
    cl: list
    df: pd.DataFrame

    @property
    def bar_ns(self) -> int:
        return BAR_HOURS[self.tf] * 3_600_000_000_000


def coins() -> list[str]:
    return [c for c in TOP100 if c.isascii() and not binance_load(c, "1h").empty]


def load_bars(coin: str, tf: str) -> Bars | None:
    """已收盤的 K 線。1h 是原始檔，2h/4h/1d 由 1h 合成；最後湊不滿的一根丟掉。"""
    raw = binance_load(coin, "1h")
    if raw.empty:
        return None
    df = raw if tf == "1h" else resample(raw, f"{BAR_HOURS[tf]}h", "start_day")
    last_close = raw["time"].iloc[-1] + pd.Timedelta(hours=1)
    df = df[df["time"] + pd.Timedelta(hours=BAR_HOURS[tf]) <= last_close].reset_index(drop=True)
    t = df["time"].dt.as_unit("ns").astype("int64").to_numpy()
    a = {k: df[k].to_numpy(float) for k in ("open", "high", "low", "close", "volume")}
    return Bars(coin, tf, t, a["open"], a["high"], a["low"], a["close"], a["volume"],
                a["high"].tolist(), a["low"].tolist(), a["close"].tolist(), df)


def sim(b: Bars, i: int, d: int, risk: float, tps: list[float], fee: float = FEE, max_bars: int = 240,
        price: float | None = None):
    """第 i 根收盤進場，從 i+1 根開始追蹤。回傳 (r, 出場 K 線索引, 碰到的 TP 數)，沒結束回傳 None。

    tps 是 TP 的價位（0~3 個，各占 1/n；空的話只靠止損和時間出場）。
    """
    price = b.cl[i] if price is None else price
    n = len(tps)
    portion = 1.0 / n if n else 1.0
    be = price * (1 + d * 2 * fee)
    stop = price - d * risk
    hit = 0
    remaining = 1.0
    r = 0.0
    hs, ls, cs = b.hl, b.ll, b.cl
    end = min(len(cs), i + 1 + max_bars)
    for j in range(i + 1, end):
        adverse = ls[j] if d > 0 else hs[j]
        if d * (adverse - stop) <= 0:
            r += remaining * (d * (stop - price) - fee * (price + stop)) / risk
            return r, j, hit
        favorable = hs[j] if d > 0 else ls[j]
        while hit < n and d * (favorable - tps[hit]) >= 0:
            tp = tps[hit]
            r += portion * (d * (tp - price) - fee * (price + tp)) / risk
            remaining -= portion
            hit += 1
            if hit == 1:
                stop = be
            elif hit == 2:
                stop = tps[0]
        if n and hit == n:
            return r, j, hit
        if j - i >= max_bars:
            r += remaining * (d * (cs[j] - price) - fee * (price + cs[j])) / risk
            return r, j, hit
    return None


COLS = ["cfg", "sym", "side", "i", "time", "exit_time", "r", "risk_pct", "tpk", "bars"]


def run_signals(b: Bars, cfg: str, sigs, fee: float = FEE, max_bars: int = 240,
                one_at_a_time: bool = True) -> list[tuple]:
    """sigs：[(i, side(+1/-1), risk(價格距離), tps(價位 list))]，依 i 排序。同一個幣前一張單沒結束就略過新訊號。"""
    out = []
    busy = -1
    for i, d, risk, tps in sigs:
        if one_at_a_time and i < busy:
            continue
        if not (risk > 0) or not math.isfinite(risk):
            continue
        res = sim(b, i, d, risk, tps, fee, max_bars)
        if res is None:
            continue
        r, j, hit = res
        busy = j
        tpk = abs(tps[0] - b.cl[i]) / risk if tps else 0.0
        out.append((cfg, b.sym, d, i, b.t[i] + b.bar_ns, b.t[j], r, risk / b.cl[i], tpk, j - i))
    return out


def to_frame(rows: list[tuple]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=COLS)
    df["time"] = pd.to_datetime(df["time"], unit="ns", utc=True)
    df["exit_time"] = pd.to_datetime(df["exit_time"], unit="ns", utc=True)
    # 滑價：進出場各 0.05%，換算成 R
    df["r_slip"] = df["r"] - 2 * SLIP / df["risk_pct"]
    return df


def window(t: pd.DataFrame, w: tuple) -> pd.DataFrame:
    return t[(t["time"] >= w[0]) & (t["time"] < w[1])]


def stats(t: pd.DataFrame, risk: float = RISK) -> dict:
    """筆數、勝率、期望值（R）、含滑價期望值、年化報酬（每筆賠 risk 複利）、最大回撤、月夏普。"""
    if len(t) == 0:
        return {"n": 0}
    r = t["r"]
    tt = t.sort_values("exit_time")
    eq = (1 + risk * tt["r"]).cumprod()
    span = (t["exit_time"].max() - t["time"].min()).total_seconds() / (365.25 * 86400)
    cagr = (eq.iloc[-1] ** (1 / span) - 1) * 100 if span > 0 and eq.iloc[-1] > 0 else float("nan")
    m = (tt["r"] * risk).groupby(tt["exit_time"].dt.to_period("M")).sum()
    sharpe = m.mean() / m.std() * math.sqrt(12) if len(m) > 2 and m.std() > 0 else float("nan")
    return {"n": int(len(r)), "win": float((r > 0).mean()), "exp_r": float(r.mean()),
            "se": float(r.std() / math.sqrt(len(r))) if len(r) > 1 else float("nan"),
            "t": float(r.mean() / (r.std() / math.sqrt(len(r)))) if len(r) > 1 and r.std() > 0 else float("nan"),
            "exp_r_slip": float(t["r_slip"].mean()), "cagr_pct": float(cagr),
            "max_dd_pct": float((eq / eq.cummax() - 1).min() * 100), "sharpe": float(sharpe),
            "per_day": len(r) / max(span * 365.25, 1)}


def random_null(trades: pd.DataFrame, data: dict[str, Bars], draws: int = 30, seed: int = 0,
                fee: float = FEE, max_bars: int = 240, tps_mult: tuple | None = None, masks: dict | None = None) -> dict:
    """同幣同年份隨機挑 K 線進場，方向、止損距離比例、TP 倍數、持有上限都和真實交易一樣。

    tps_mult 為 None 時用交易自己的 tpk 做單一目標（3 個 TP 同價位，tpk=0 代表沒有 TP）；
    傳 (1,2,3) 則是 1R/2R/3R 的三段出場。masks：{幣: {1: 布林陣列, -1: 布林陣列}}，只從符合該方向條件的 K 線裡抽
    （例如「趨勢同向」的對照，用來分辨訊號的價值是來自趨勢還是來自進場時機）。
    """
    rng = np.random.default_rng(seed)
    pools: dict[str, dict] = {}
    for sym, b in data.items():
        years = pd.to_datetime(b.t, unit="ns", utc=True).year.to_numpy()
        idx = np.arange(len(b.t))
        pools[sym] = {int(y): np.flatnonzero((years == y) & (idx >= 200)) for y in np.unique(years)}
    tr = list(zip(trades["sym"], trades["side"], trades["time"].dt.year, trades["risk_pct"], trades["tpk"]))
    means = []
    for _ in range(draws):
        rs = []
        for sym, d, year, rp, tpk in tr:
            pool = pools[sym].get(int(year))
            if pool is not None and masks is not None:
                pool = pool[masks[sym][d][pool]]
            if pool is None or len(pool) == 0:
                continue
            b = data[sym]
            i = int(pool[rng.integers(len(pool))])
            price = b.cl[i]
            risk = rp * price
            ks = tps_mult or ((tpk, tpk, tpk) if tpk > 0 else ())
            res = sim(b, i, d, risk, [price + d * k * risk for k in ks], fee, max_bars)
            if res is not None:
                rs.append(res[0])
        means.append(float(np.mean(rs)))
    return {"mean": float(np.mean(means)), "sd": float(np.std(means)), "draws": means}
