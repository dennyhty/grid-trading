"""幣安 USDT 本位永續合約的公開行情（訊號網站掃描用，不需要金鑰）。

fapi.binance.com 會擋美國的連線（GitHub Actions 的機器在美國，回 HTTP 451），
www.binance.com 底下的同一組 /fapi/v1/* 不擋（2026-10-20 在 Actions 上確認過 klines、ticker、exchangeInfo 都是 200）。

網站內部的代號沿用 <幣>_USDT_PERP，<幣> 直接是幣安合約名稱去掉 USDT（例如 1000PEPE_USDT_PERP → 1000PEPEUSDT），
和 VPS 上自動交易程式（grid_trading/trader）用同一份幣安合約名單。
"""

from __future__ import annotations

import time

import pandas as pd
import requests

from .binance_history import binance_symbols
from .universe import TOP100

BASE_URL = "https://www.binance.com"
INTERVALS = {"15M": "15m", "30M": "30m", "60M": "1h", "2H": "2h", "4H": "4h", "8H": "8h", "12H": "12h", "1D": "1d"}
MAX_LIMIT = 1500
# exchangeInfo 抓不到時用的備用名單：市值前 100 名裡幣安沒有合約的（2026-10-19 查過）
NOT_LISTED = {"CRO", "OKB", "MNT", "PI"}


def to_binance(symbol: str) -> str:
    """BTC_USDT_PERP → BTCUSDT。"""
    return symbol.split("_")[0] + "USDT"


def from_binance(symbol: str) -> str:
    """BTCUSDT → BTC_USDT_PERP。"""
    return symbol[:-4] + "_USDT_PERP"


class BinancePublic:
    def __init__(self, base_url: str = BASE_URL, timeout: float = 15.0, pause: float = 0.05):
        self.base_url = base_url
        self.timeout = timeout
        self.pause = pause
        self.session = requests.Session()

    def _get(self, path: str, params: dict | None = None):
        resp = self.session.get(self.base_url + path, params=params, timeout=self.timeout)
        resp.raise_for_status()
        time.sleep(self.pause)
        return resp.json()

    def kline_rows(self, symbol: str, interval: str, limit: int, end: int | None = None) -> list[list]:
        """一頁 K 線（最多 1500 根），由舊到新。symbol 用網站代號，interval 用網站寫法（4H、2H…）。"""
        params = {"symbol": to_binance(symbol), "interval": INTERVALS[interval], "limit": min(MAX_LIMIT, limit)}
        if end is not None:
            params["endTime"] = end
        return self._get("/fapi/v1/klines", params)

    def prices(self) -> dict[str, float]:
        """所有合約最新價，key 是網站代號。"""
        return {from_binance(r["symbol"]): float(r["price"]) for r in self._get("/fapi/v1/ticker/price")
                if r["symbol"].endswith("USDT")}

    def usdt_perps(self) -> set[str]:
        """交易中的 USDT 本位永續合約（幣安名稱，例如 BTCUSDT）。"""
        info = self._get("/fapi/v1/exchangeInfo")
        return {s["symbol"] for s in info["symbols"]
                if s.get("contractType") == "PERPETUAL" and s.get("quoteAsset") == "USDT" and s.get("status") == "TRADING"}


def universe(api: BinancePublic | None = None) -> list[str]:
    """要掃描的幣：市值前 100 名（grid_trading/universe.py）裡幣安有 USDT 永續合約的，網站代號。"""
    try:
        listed = (api or BinancePublic()).usdt_perps()
        syms = [b for b in (binance_symbols(c)[-1] for c in TOP100) if b in listed]
    except Exception as e:  # 抓不到合約清單：用備用名單，K 線抓不到的幣掃描時會自己跳過
        print(f"注意：幣安合約清單抓不到（{e}），用備用名單", flush=True)
        syms = [binance_symbols(c)[-1] for c in TOP100 if c not in NOT_LISTED]
    return sorted(from_binance(b) for b in syms)


def rows_to_frame(rows: list[list]) -> pd.DataFrame:
    df = pd.DataFrame([r[:6] for r in rows], columns=["time", "open", "high", "low", "close", "volume"])
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["time"] = pd.to_datetime(df["time"].astype("int64"), unit="ms", utc=True)
    return df
