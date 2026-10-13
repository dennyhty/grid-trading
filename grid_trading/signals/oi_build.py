"""訂單流觀察組：價漲倉增做多、價跌倉增做空（4H，看過去 24 小時）。

規則（和 orderflow_backtest.py 的「oi n6 k2.0 build」相同）：
- 過去 6 根 4H（24 小時）的漲跌幅，和該幣過去 200 根的平常值比，超過 2 個標準差；
- 同一段時間的持倉量變化，也比平常多 2 個標準差以上（新資金進場）；
- 漲 + 倉增 → 做多，跌 + 倉增 → 做空。只在條件剛成立那一根觸發。
- 停損 1.5 ATR = 1R，TP 1R/2R/3R，出場規則和維加斯相同。

持倉量來自幣安（派網沒有），即時掃描時用 www.binance.com 的 openInterestHist（只有最近 30 天，
所以持倉量的標準化最多用約 180 根，回測是 200 根）。抓不到持倉量的幣不會出訊號。

回測見專案的 report_orderflow_20261012.md：逐筆 2024 年後 +0.09R，但同一天同方向算一注時只剩 +0.03R，
還不能當成已驗證的策略，所以只當觀察組、不推播。

2026-10-13 起也跑美股和商品（Denny 要求，觀察組）：幣安有這些標的的 USDT 永續（股票代號不帶 X，例如 NVDAX → NVDAUSDT），
持倉量一樣從幣安抓；幣安沒有的（WTI、布蘭特、USOX…）不出訊號。STXX（希捷）和幣安的 STX（Stacks 幣）撞名，不跑。
美股、商品另外有 2H 版本（OiBuild2H）。幣安這些合約大多 2026 年才上市，回測樣本很小。
"""

from __future__ import annotations

import pandas as pd

from ..binance_live import attach_open_interest, fetch_open_interest
from ..markets import STOCK, classify
from .orderflow import OpenInterestPrice
from .strategy import LONG, register

# 派網代號對到幣安時會撞到別的東西的，直接不跑
NO_BINANCE = {"STXX"}


def binance_coin(symbol: str) -> str | None:
    """派網合約 → 幣安的幣名（binance_symbols 用）。美股去掉結尾的 X；撞名的回 None。"""
    base = symbol.split("_")[0]
    if base in NO_BINANCE:
        return None
    return base[:-1] if classify(symbol) == STOCK else base


@register
class OiBuild(OpenInterestPrice):
    name = "oi_build"
    title = "價漲倉增（訂單流觀察組）"
    description = ("過去 24 小時漲幅（跌幅）和持倉量增加都異常大時，順著方向做多（做空）。"
                   "持倉量用幣安的資料。停損 1.5 ATR，TP 為 1R/2R/3R。")
    tag = "訂單流觀察組"
    intervals = ("4H",)
    markets = ("crypto", "stock", "commodity")
    chart_emas = {"EMA144": 144, "EMA169": 169}

    def __init__(self, n: int = 6, k: float = 2.0, **kw):
        super().__init__(n=n, k=k, mode="build", **kw)

    def scan(self, df: pd.DataFrame, symbol: str, interval: str, start: int | None = None):
        if "oi" not in df:  # 即時掃描：派網 K 線沒有持倉量，接上幣安的
            coin = binance_coin(symbol)
            if coin is None:
                return []
            oi = fetch_open_interest(coin, interval)
            if oi.empty:
                return []
            df = attach_open_interest(df, oi, interval)
        return super().scan(df, symbol, interval, start)

    def check(self, df: pd.DataFrame, i: int):
        hit = super().check(df, i)
        if hit is None:
            return None
        side, sl, tps, info = hit
        r = df.iloc[i]
        n = self.n
        hours = n * (df["time"].iat[i] - df["time"].iat[i - 1]) / pd.Timedelta(hours=1)
        up = side == LONG
        info["reason"] = (f"過去 {hours:g} 小時{'大漲' if up else '大跌'}，持倉量同時明顯增加（新資金進場）")
        info["details"] = [
            [f"{hours:g} 小時漲跌", f"{r[f'ret{n}'] * 100:+.2f}%（{r[f'ret{n}_z']:+.1f}σ）"],
            [f"{hours:g} 小時持倉量變化", f"{r[f'doi{n}'] * 100:+.2f}%（{r[f'doi{n}_z']:+.1f}σ）"],
        ]
        return side, sl, tps, info



@register
class OiBuild2H(OiBuild):
    """同一套規則跑在 2H（過去 6 根 = 12 小時），只給美股和商品當觀察組。"""
    name = "oi_build_2h"
    title = "價漲倉增 2H（訂單流觀察組）"
    description = OiBuild.description.replace("24 小時", "12 小時")
    intervals = ("2H",)
    markets = ("stock", "commodity")
