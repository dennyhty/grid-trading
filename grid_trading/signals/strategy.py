"""策略介面。新策略只要繼承 Strategy、實作 prepare() 和 check()，再用 @register 登記即可。

規則：check(df, i) 只能看 df 第 i 根（含）以前的資料，第 i 根是剛收盤的 K 線。
這樣同一份程式碼可以拿來回測，也可以拿來即時掃描，不會偷看未來。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

LONG = "LONG"
SHORT = "SHORT"


@dataclass
class Signal:
    symbol: str
    interval: str
    side: str            # LONG / SHORT
    time: pd.Timestamp   # 訊號 K 線的收盤時間（UTC）
    price: float         # 訊號價格 = 收盤價 = 建議進場價
    sl: float
    tps: list[float]     # TP1、TP2、TP3
    strategy: str
    info: dict = field(default_factory=dict)  # RSI、MACD 等，顯示在「RSI 與策略資訊」

    @property
    def direction(self) -> int:
        return 1 if self.side == LONG else -1

    def to_dict(self) -> dict:
        d = asdict(self)
        d["time"] = int(self.time.timestamp() * 1000)
        return d


class Strategy:
    name = "base"
    title = "未命名策略"
    description = ""
    # 圖表上要畫的均線，例如 {"EMA12": 12, "EMA144": 144}
    chart_emas: dict[str, int] = {}
    min_bars = 200

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        """一次算好整段指標（只能用 ewm/rolling 這類不看未來的算法），回傳加了欄位的 df。"""
        return df

    def check(self, df: pd.DataFrame, i: int) -> tuple[str, float, list[float], dict] | None:
        """第 i 根收盤時是否觸發。觸發就回傳 (side, sl, [tp1, tp2, tp3], info)，否則 None。"""
        raise NotImplementedError

    def candidates(self, df: pd.DataFrame):
        """（可選）用向量運算先挑出可能觸發的 K 線，回傳布林陣列；只有 True 的那幾根才呼叫 check()。

        必須涵蓋 check() 可能觸發的每一根（只能多不能少），結果才會和逐根檢查一樣。回傳 None 表示每根都檢查。
        """
        return None

    def scan(self, df: pd.DataFrame, symbol: str, interval: str, start: int | None = None) -> list[Signal]:
        """回傳 df 中從 start 起所有觸發的訊號（依時間）。df 每一根都必須已收盤。"""
        df = self.prepare(df.reset_index(drop=True))
        bar = _bar_length(df)
        out = []
        first = max(start or 0, self.min_bars)
        mask = self.candidates(df)
        rows = range(first, len(df)) if mask is None else (i for i in np.flatnonzero(mask) if i >= first)
        for i in rows:
            hit = self.check(df, int(i))
            if hit is None:
                continue
            side, sl, tps, info = hit
            out.append(Signal(symbol, interval, side, df["time"].iloc[i] + bar, float(df["close"].iloc[i]),
                              float(sl), [float(t) for t in tps], self.name, info))
        return out


def _bar_length(df: pd.DataFrame) -> pd.Timedelta:
    if len(df) < 2:
        return pd.Timedelta(0)
    return (df["time"].diff().dropna()).median()


REGISTRY: dict[str, type[Strategy]] = {}


def register(cls: type[Strategy]) -> type[Strategy]:
    REGISTRY[cls.name] = cls
    return cls


def get(name: str, **params) -> Strategy:
    return REGISTRY[name](**params)
