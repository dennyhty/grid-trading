"""SMC 觀察組：4H 掃流動性 → 結構轉變，突破那根收盤進場，只做日線結構的方向。

定義見 grid_trading/smc.py，回測見 research/smc_backtest.py 和專案的 report_smc_ict_20261012.md。
以做多為例（做空反過來）：

1. 日線結構向上：日線最近一次收盤突破的是擺動高點（擺動點 = 左右各 3 根都不比它高 / 低）。
2. 4H 影線跌破最近 100 根內還沒被碰過的擺動低點，但收盤收回上方（掃流動性）。
3. 掃完 20 根內，4H 收盤突破最近一個擺動高點，中間沒有收盤跌破掃出的低點。
4. 突破那根收盤進場；停損在波段最低點下方 0.1 ATR（至少 0.5 ATR）；TP 1R/2R/3R，出場規則和維加斯相同。

只在 4H 上跑。加密貨幣用幣安 2020 年起的資料驗證過，2024 年後的優勢在統計上還不夠確定，所以標成觀察組。
2026-10-13 起也在美股、商品上跑（派網約一年的資料：美股 -0.06R、商品 +0.40R 但只有 55 筆），只觀察、不推播。
美股、商品另外有 2H 版本（SmcSweep2H，Denny 要求補齊 4H/2H 觀察組）；加密貨幣 2H 在研究裡是負的，所以不跑。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import indicators as ind
from .. import smc
from ..history import resample
from .strategy import LONG, SHORT, Strategy, register


@register
class SmcSweep(Strategy):
    name = "smc_sweep"
    title = "SMC 掃流動性 + 結構轉變（觀察組）"
    description = ("4H 影線掃掉前低（前高）後收回，接著收盤突破最近的擺動高點（低點），順著日線結構方向進場。"
                   "停損在掃出的極值外側，通常比維加斯寬很多；TP 為 1R/2R/3R。")
    tag = "SMC觀察組"
    intervals = ("4H",)
    markets = ("crypto", "stock", "commodity")
    min_bars = 200

    def __init__(self, k: int = 3, liq_lookback: int = 100, sweep_window: int = 20,
                 tp_r: tuple[float, float, float] = (1.0, 2.0, 3.0)):
        self.k = k
        self.liq_lookback = liq_lookback
        self.sweep_window = sweep_window
        self.tp_r = tuple(tp_r)

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["atr"] = ind.atr(df)
        hours = _bar_hours(df)
        daily = _closed_daily(df, hours)
        htf = smc.htf_trend(df, daily, hours, 24, self.k) if len(daily) > 2 * self.k + 1 else np.zeros(len(df), int)
        _, breaks = smc.structure(df, self.k, self.liq_lookback)
        setups = smc.find_setups(df, breaks, htf, setup="sweep", entry="market", htf=True, discount=False,
                                 sweep_window=self.sweep_window)
        by_bar = {b.bar: b for b in breaks}
        n = len(df)
        cols = {c: np.full(n, np.nan) for c in ("smc_dir", "smc_sl", "smc_sweep", "smc_pivot", "smc_leg_low", "smc_leg_high")}
        kind = np.full(n, "", dtype=object)
        low, high = df["low"].to_numpy(), df["high"].to_numpy()
        for s in setups:
            b = by_bar[s.bar]
            cols["smc_dir"][s.bar] = s.direction
            cols["smc_sl"][s.bar] = s.sl
            cols["smc_sweep"][s.bar] = low[b.sweep] if s.direction > 0 else high[b.sweep]
            cols["smc_pivot"][s.bar] = high[b.pivot] if s.direction > 0 else low[b.pivot]
            cols["smc_leg_low"][s.bar] = s.leg_low
            cols["smc_leg_high"][s.bar] = s.leg_high
            kind[s.bar] = s.kind
        for c, v in cols.items():
            df[c] = v
        df["smc_kind"] = kind
        df["htf_trend"] = htf
        return df

    def candidates(self, df: pd.DataFrame):
        return df["smc_dir"].notna().to_numpy()

    def check(self, df: pd.DataFrame, i: int):
        d = df["smc_dir"].iat[i]
        if not d == d:  # NaN：這根沒有進場計畫
            return None
        r = df.iloc[i]
        entry, sl = float(r.close), float(r.smc_sl)
        risk = abs(entry - sl)
        if risk <= 0:
            return None
        d = int(d)
        side = LONG if d > 0 else SHORT
        swept = "前低" if d > 0 else "前高"
        broke = "擺動高點" if d > 0 else "擺動低點"
        info = {
            "reason": f"日線結構{'向上' if d > 0 else '向下'}；{_bar_hours(df):.0f}H 掃掉{swept} {r.smc_sweep:.6g} 後收回，"
                      f"收盤突破{broke} {r.smc_pivot:.6g}（{'轉向' if r.smc_kind == smc.CHOCH else '順勢'}）",
            "atr": float(r.atr),
            "details": [
                ["掃出的極值", f"{r.smc_sweep:.6g}"],
                [f"突破的{broke}", f"{r.smc_pivot:.6g}"],
                ["結構", "CHoCH 轉向" if r.smc_kind == smc.CHOCH else "BOS 順勢"],
                ["日線結構", "向上" if r.htf_trend > 0 else "向下"],
                ["停損距離", f"{risk / entry * 100:.1f}%（停損寬，倉位要算小）"],
            ],
            "candle": {"open": float(r.open), "high": float(r.high), "low": float(r.low),
                       "close": float(r.close), "time": int(r.time.timestamp() * 1000)},
        }
        return side, sl, [entry + d * k * risk for k in self.tp_r], info


@register
class SmcSweep2H(SmcSweep):
    """同一套 SMC 規則跑在 2H，只給美股和商品當觀察組（派網約一年資料：美股 -0.03R、商品 +0.02R）。"""
    name = "smc_sweep_2h"
    title = "SMC 掃流動性 + 結構轉變 2H（觀察組）"
    description = SmcSweep.description.replace("4H", "2H")
    intervals = ("2H",)
    markets = ("stock", "commodity")


def _bar_hours(df: pd.DataFrame) -> float:
    if len(df) < 2:
        return 4.0
    return df["time"].diff().dropna().median().total_seconds() / 3600


def _closed_daily(df: pd.DataFrame, hours: float) -> pd.DataFrame:
    """由 K 線合成 UTC 日線，只留已經收完的那幾天。"""
    if df.empty:
        return df
    daily = resample(df[["time", "open", "high", "low", "close", "volume"]], "1D", origin="epoch")
    last_close = df["time"].iloc[-1] + pd.Timedelta(hours=hours)
    return daily[daily["time"] + pd.Timedelta(days=1) <= last_close].reset_index(drop=True)
