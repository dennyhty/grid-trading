"""追蹤一個訊號之後的走勢：TP 觸及、保護止損移動、止損出場。回測和即時卡片共用。

出場規則（固定，策略不用管）：
- 在 TP1、TP2、TP3 各平 1/3。
- TP1 觸及後，剩餘倉位的止損移到 BE（進場價加上來回手續費，出場不賠錢）。
- TP2 觸及後，止損再移到 TP1。
- 同一根 K 線同時碰到止損和 TP 時，保守假設先碰止損。
- 超過 max_bars 根還沒結束，就用收盤價平掉剩下的倉位。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .strategy import Signal

PORTIONS = (1 / 3, 1 / 3, 1 / 3)


@dataclass
class Event:
    kind: str      # TP1 / TP2 / TP3 / STOP_MOVED / SL / BE_EXIT / PROTECT_EXIT / TIMEOUT
    time: pd.Timestamp
    price: float
    text: str

    def to_dict(self) -> dict:
        return {"kind": self.kind, "time": int(self.time.timestamp() * 1000), "price": self.price, "text": self.text}


@dataclass
class TrackResult:
    status: str               # OPEN / CLOSED
    events: list[Event] = field(default_factory=list)
    stop: float = 0.0         # 目前的止損價
    remaining: float = 1.0    # 剩餘倉位比例
    realized_r: float = 0.0   # 已實現（扣手續費），以 R 計
    realized_pct: float = 0.0  # 已實現（扣手續費），以 1 倍槓桿下的倉位報酬 % 計
    tps_hit: int = 0
    bars: int = 0
    be: float = 0.0

    def to_dict(self) -> dict:
        return {"status": self.status, "events": [e.to_dict() for e in self.events], "stop": self.stop,
                "remaining": self.remaining, "realized_r": self.realized_r, "realized_pct": self.realized_pct,
                "tps_hit": self.tps_hit, "bars": self.bars, "be": self.be}


def breakeven(sig: Signal, fee: float) -> float:
    return sig.price * (1 + sig.direction * 2 * fee)


def track(sig: Signal, candles: pd.DataFrame, fee: float = 0.0005, max_bars: int = 240) -> TrackResult:
    """candles：K 線（time 是開盤時間），由舊到新。只看開盤時間 >= 訊號時間的部分。"""
    d = sig.direction
    risk = abs(sig.price - sig.sl)
    be = breakeven(sig, fee)
    res = TrackResult(status="OPEN", stop=sig.sl, be=be)
    # 只取訊號後最多 max_bars 根，轉成 list 逐根跑（iterrows 很慢，回測時會被呼叫上千次）
    start = int(candles["time"].searchsorted(sig.time, side="left"))
    end = len(candles) if max_bars is None else min(len(candles), start + max_bars)
    times = candles["time"].iloc[start:end].tolist()
    highs = candles["high"].iloc[start:end].tolist()
    lows = candles["low"].iloc[start:end].tolist()
    closes = candles["close"].iloc[start:end].tolist()

    def close_part(portion: float, price: float):
        move = d * (price - sig.price)
        cost = fee * (sig.price + price)  # 進出場各一次手續費
        res.realized_r += portion * (move - cost) / risk
        res.realized_pct += portion * (move - cost) / sig.price * 100
        res.remaining -= portion

    for t, high, low, close in zip(times, highs, lows, closes):
        res.bars += 1
        adverse = low if d > 0 else high
        if d * (adverse - res.stop) <= 0:
            price = res.stop
            if res.tps_hit == 0:
                kind, text = "SL", f"止損出場 {_fmt(price)}"
            elif res.tps_hit == 1:
                kind, text = "BE_EXIT", f"保本出場 {_fmt(price)}"
            else:
                kind, text = "PROTECT_EXIT", f"保護止損出場 {_fmt(price)}"
            close_part(res.remaining, price)
            res.events.append(Event(kind, t, price, text))
            res.status = "CLOSED"
            break
        favorable = high if d > 0 else low
        while res.tps_hit < len(sig.tps) and d * (favorable - sig.tps[res.tps_hit]) >= 0:
            n = res.tps_hit
            tp = sig.tps[n]
            close_part(PORTIONS[n], tp)
            res.tps_hit += 1
            res.events.append(Event(f"TP{n + 1}", t, tp, f"TP{n + 1} 觸及 {_fmt(tp)}"))
            new_stop = be if n == 0 else (sig.tps[n - 1] if n == 1 else None)
            if new_stop is not None:
                res.stop = new_stop
                res.events.append(Event("STOP_MOVED", t, new_stop, f"保護止損設於 {_fmt(new_stop)}"))
        if res.tps_hit == len(sig.tps):
            res.remaining = 0.0
            res.status = "CLOSED"
            break
        if res.bars >= max_bars:
            close_part(res.remaining, float(close))
            res.events.append(Event("TIMEOUT", t, float(close), f"超過 {max_bars} 根未結束，收盤平倉"))
            res.status = "CLOSED"
            break
    return res


def _fmt(p: float) -> str:
    return f"{p:.6g}"
