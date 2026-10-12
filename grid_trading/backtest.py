"""滾動視窗回測：檢驗「篩選器的開單規則」本身在歷史上賺不賺錢。

每隔 hold 天在時間點 t：
  1. 只用 t 之前的 1H K 線合成 4H，丟進 screener.analyze() 得到方向、區間、格數、停損
  2. 用 t 之後 hold 天的 1H K 線模擬網格
所以每個視窗用到的參數都不會偷看未來。
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from .gridsim import LONG, NEUTRAL, SHORT, simulate
from .history import resample
from .screener import MAX_RANGE_WIDTH_PCT, ScreenConfig, analyze


ANCHOR = pd.Timestamp("2025-01-01", tz="UTC")


@dataclass(frozen=True)
class Variant:
    direction: str = "vegas"      # vegas：用維加斯通道定方向；neutral：一律中性
    filtered: bool = True         # True：只在震盪條件成立時開單
    hold_days: int = 7
    stop_atr_mult: float = 1.0
    max_adx: float = 25.0
    max_efficiency: float = 0.15

    def label(self) -> str:
        return (f"{'順勢' if self.direction == 'vegas' else '中性'}"
                f"|{'過濾' if self.filtered else '不過濾'}"
                f"|{self.hold_days}天|停損{self.stop_atr_mult:g}ATR")


def windows(h1: pd.DataFrame, variant: Variant, cfg: ScreenConfig, symbol: str, fee: float) -> list[dict]:
    h1 = h1.reset_index(drop=True)
    times = h1["time"]
    warmup = times.iloc[0] + pd.Timedelta(hours=4 * (cfg.min_bars + 10))
    step = pd.Timedelta(days=variant.hold_days)
    # 所有幣用同一套日曆切視窗，組合報酬才能把同一期的幣放在一起算
    t = ANCHOR + step * int(np.ceil((warmup - ANCHOR) / step))
    cfg = replace(cfg, stop_atr_mult=variant.stop_atr_mult)
    h4_all = resample(h1)  # t 都對齊 4H 整點，所以 time < t 的 4H K 線都已收盤
    out = []
    while t + step <= times.iloc[-1]:
        h4 = h4_all[h4_all["time"] < t].iloc[-500:]
        r = analyze(h4, cfg)
        future = h1[(times >= t) & (times < t + step)]
        t_start = t
        t += step
        if r is None or len(future) < 24:
            continue
        mode = NEUTRAL if variant.direction == "neutral" else r["direction"]
        tradable = (
            r["in_range"]
            and r["range_width_pct"] <= MAX_RANGE_WIDTH_PCT
            and r["adx"] <= variant.max_adx
            and r["efficiency"] <= variant.max_efficiency
        )
        row = {"symbol": symbol, "start": t_start, "mode": mode, "tradable": tradable,
               "adx": r["adx"], "efficiency": r["efficiency"], "range_width_pct": r["range_width_pct"]}
        if variant.filtered and not tradable:
            out.append({**row, "traded": False, "ret": 0.0})
            continue
        # 做多只設下方停損、做空只設上方停損、中性兩邊都設（強制中性時也一樣）
        atr_abs = r["atr_pct"] / 100 * r["price"]
        stop_low = r["lower"] - variant.stop_atr_mult * atr_abs if mode != SHORT else None
        stop_high = r["upper"] + variant.stop_atr_mult * atr_abs if mode != LONG else None
        p0 = float(future["open"].iloc[0])
        if (stop_low is not None and p0 <= stop_low) or (stop_high is not None and p0 >= stop_high):
            out.append({**row, "traded": False, "ret": 0.0})
            continue
        res = simulate(future, r["lower"], r["upper"], r["grids"], mode, stop_low, stop_high, fee)
        out.append({**row, "traded": True, "ret": res.ret, "grid_profit": res.grid_profit,
                    "fees": res.fees, "max_dd": res.max_drawdown, "trades": res.trades,
                    "stopped": res.stopped})
    return out


def summarize(df: pd.DataFrame, hold_days: int) -> dict:
    traded = df[df["traded"]]
    if traded.empty:
        return {"開單次數": 0}
    # 投資組合：每個視窗把資金平均分給當期有開單的幣；沒開單的資金報酬為 0
    per_window = df.groupby("start")["ret"].mean()
    equity = (1 + per_window).cumprod()
    years = len(per_window) * hold_days / 365
    total = equity.iloc[-1] - 1
    dd = (equity / equity.cummax() - 1).min()
    return {
        "開單次數": len(traded),
        "開單率": len(traded) / len(df),
        "單次平均報酬%": traded["ret"].mean() * 100,
        "單次中位數%": traded["ret"].median() * 100,
        "勝率": (traded["ret"] > 0).mean(),
        "最差單次%": traded["ret"].min() * 100,
        "停損率": traded["stopped"].mean(),
        "組合總報酬%": total * 100,
        "組合年化%": ((1 + total) ** (1 / years) - 1) * 100 if years > 0 and total > -1 else np.nan,
        "組合最大回撤%": dd * 100,
    }
