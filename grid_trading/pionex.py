"""派網 (Pionex) 公開行情 API 客戶端。

只呼叫不需要 API Key 的公開端點，程式碼裡不存放任何金鑰。
"""

from __future__ import annotations

import time

import pandas as pd
import requests

BASE_URL = "https://api.pionex.com"


class PionexError(RuntimeError):
    pass


class PionexPublic:
    def __init__(self, base_url: str = BASE_URL, timeout: float = 15.0, pause: float = 0.15):
        self.base_url = base_url
        self.timeout = timeout
        self.pause = pause  # 每次請求之間稍停，避免觸發頻率限制
        self.session = requests.Session()

    def _get(self, path: str, params: dict | None = None) -> dict:
        resp = self.session.get(self.base_url + path, params=params, timeout=self.timeout)
        resp.raise_for_status()
        body = resp.json()
        if not body.get("result", False):
            raise PionexError(f"{path} 回傳錯誤: {body.get('code')} {body.get('message')}")
        time.sleep(self.pause)
        return body.get("data", {})

    def perp_symbols(self) -> list[str]:
        """所有 USDT 本位永續合約的交易對，例如 BTC_USDT_PERP。"""
        data = self._get("/api/v1/common/symbols", {"type": "PERP"})
        out = []
        for s in data.get("symbols", []):
            name = s.get("symbol", "")
            if s.get("quoteCurrency", "USDT") != "USDT" or not name.endswith("_USDT_PERP"):
                continue
            if s.get("status", "TRADING") != "TRADING" or s.get("enable") is False:
                continue
            out.append(name)
        return out

    def perp_specs(self) -> dict[str, dict]:
        """每個永續合約的下單限制：min_size 最小數量（顆）、step 數量級距、min_notional 最小名目金額（USDT）。"""
        data = self._get("/api/v1/common/symbols", {"type": "PERP"})
        out = {}
        for s in data.get("symbols", []):
            try:
                out[s["symbol"]] = {"min_size": float(s.get("minSizeMarket") or s.get("minSizeLimit") or 0),
                                    "step": float(s.get("baseStep") or 0),
                                    "min_notional": float(s.get("minNotional") or 0)}
            except (KeyError, TypeError, ValueError):
                continue
        return out

    def perp_tickers(self) -> pd.DataFrame:
        """24 小時行情，amount 欄位是以 USDT 計的成交額。"""
        data = self._get("/api/v1/market/tickers", {"type": "PERP"})
        df = pd.DataFrame(data.get("tickers", []))
        for col in ("open", "close", "high", "low", "volume", "amount"):
            if col in df:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

    def klines(self, symbol: str, interval: str = "4H", limit: int = 500) -> pd.DataFrame:
        """K 線，依時間由舊到新排序。interval 可用 1M/5M/15M/30M/60M/4H/8H/12H/1D。"""
        data = self._get("/api/v1/market/klines", {"symbol": symbol, "interval": interval, "limit": limit})
        df = pd.DataFrame(data.get("klines", []))
        if df.empty:
            return df
        for col in ("open", "high", "low", "close", "volume"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True)
        return df.sort_values("time").reset_index(drop=True)
