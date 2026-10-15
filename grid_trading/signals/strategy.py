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
    # 出場方式（tracker 用）：portions 每個 TP 平幾成；trail 是 TP 都碰到後，剩下倉位的追蹤止損距離（價格）。
    # 空的就是預設：TP1/2/3 各平 1/3、不追蹤
    exit: dict = field(default_factory=dict)

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
    # 觀察組之類的標籤，會加在網站卡片和推播文字前面；空字串就不加
    tag = ""
    # 只在某些類別當觀察組時用，例如 {"stock": "美股觀察組"}；沒列到的類別用 tag
    market_tags: dict[str, str] = {}
    # 只在這些週期 / 類別上跑（engine 的週期寫法，例如 "4H"；類別 crypto/stock/commodity），None 表示全部
    intervals: tuple[str, ...] | None = None
    markets: tuple[str, ...] | None = None
    # 實盤（沒有標籤）時，同一個幣、同方向只做一張：另一個週期的單還沒出場，新訊號就不做（engine 處理）
    one_per_coin = False

    def tag_for(self, market: str) -> str:
        """這個類別的標籤；有標籤就是觀察組（網站顯示、不推播）。"""
        return self.market_tags.get(market, self.tag)

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        """一次算好整段指標（只能用 ewm/rolling 這類不看未來的算法），回傳加了欄位的 df。"""
        return df

    def check(self, df: pd.DataFrame, i: int) -> tuple[str, float, list[float], dict] | None:
        """第 i 根收盤時是否觸發。觸發就回傳 (side, sl, [tp1, tp2, tp3], info)，否則 None。"""
        raise NotImplementedError

    def exit_plan(self, interval: str) -> dict | None:
        """（可選）某個週期要用不同的出場方式時回傳 {"tp_r": [...], "portions": [...], "trail_atr": k}。

        tp_r 是各 TP 的 R 倍數（取代 check() 給的 TP），portions 是各 TP 平倉比例，trail_atr 是 TP 都碰到後
        剩下倉位用幾倍 ATR（訊號 K 棒的 ATR）追蹤止損。None 表示照 check() 的 TP、各平 1/3。
        """
        return None

    def accept(self, df: pd.DataFrame, i: int, interval: str) -> bool:
        """（可選）只在某些週期才用的進場條件：check() 觸發後再檢查一次，回傳 False 就不出訊號。"""
        return True

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
            if not self.accept(df, int(i), interval):
                continue
            side, sl, tps, info = hit
            price = float(df["close"].iloc[i])
            exit_ = {}
            plan = self.exit_plan(interval)
            if plan:
                d = 1 if side == LONG else -1
                risk = abs(price - float(sl))
                tps = [price + d * k * risk for k in plan["tp_r"]]
                exit_ = {"portions": list(plan["portions"])}
                if plan.get("trail_atr"):
                    exit_["trail"] = float(plan["trail_atr"] * info["atr"])
                    exit_["trail_atr"] = plan["trail_atr"]
            out.append(Signal(symbol, interval, side, df["time"].iloc[i] + bar, price,
                              float(sl), [float(t) for t in tps], self.name, info, exit_))
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
