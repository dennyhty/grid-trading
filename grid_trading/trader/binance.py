"""幣安 USDT 本位合約 REST API 的最小用戶端：行情（不用金鑰）和下單（要金鑰）。

- 行情一律走正式站 fapi.binance.com（訊號要用真實價格算）；測試網模式的下單和持倉管理走測試網。
- 美國的 IP 會被擋（HTTP 451），所以只能在美國以外的主機上跑（GitHub Actions 不行）。
- 條件單（止損、止盈）從 2025-12 起改走 Algo Order：POST/DELETE /fapi/v1/algoOrder、GET /fapi/v1/openAlgoOrders。
"""

from __future__ import annotations

import hashlib
import hmac
import time
import urllib.parse

import pandas as pd
import requests

MAINNET = "https://fapi.binance.com"
TESTNET = "https://testnet.binancefuture.com"
INTERVALS = {"2H": "2h", "4H": "4h"}


class BinanceError(RuntimeError):
    def __init__(self, status: int, code, msg: str):
        super().__init__(f"HTTP {status} code={code} {msg}")
        self.status, self.code, self.msg = status, code, msg


class BinanceFutures:
    def __init__(self, base: str = MAINNET, key: str = "", secret: str = "", timeout: float = 15,
                 session: requests.Session | None = None):
        self.base = base.rstrip("/")
        self.key, self.secret = key, secret
        self.timeout = timeout
        self.session = session or requests.Session()
        self._offset = 0  # 本機時間和幣安伺服器時間的差（毫秒），簽名用

    # ---------- 底層 ----------
    def _request(self, method: str, path: str, params: dict | None = None, signed: bool = False, tries: int = 3):
        params = {k: v for k, v in (params or {}).items() if v is not None}
        headers = {}
        for i in range(tries):
            q = dict(params)
            if signed:
                if not (self.key and self.secret):
                    raise RuntimeError("沒有設定 BINANCE_API_KEY / BINANCE_API_SECRET")
                q["timestamp"] = int(time.time() * 1000) + self._offset
                q["recvWindow"] = 10_000
                query = urllib.parse.urlencode(q)
                q["signature"] = hmac.new(self.secret.encode(), query.encode(), hashlib.sha256).hexdigest()
                headers["X-MBX-APIKEY"] = self.key
            try:
                resp = self.session.request(method, self.base + path, params=q, headers=headers, timeout=self.timeout)
            except requests.RequestException:
                if i == tries - 1:
                    raise
                time.sleep(2 ** i)
                continue
            if resp.status_code in (418, 429) or resp.status_code >= 500:
                if i < tries - 1:  # 被限流或幣安自己出錯，等一下再試
                    time.sleep(float(resp.headers.get("Retry-After", 2 ** (i + 1))))
                    continue
            try:
                data = resp.json()
            except ValueError:
                data = {"msg": resp.text[:200]}
            if resp.status_code >= 400:
                code = data.get("code") if isinstance(data, dict) else None
                if code == -1021 and signed and i < tries - 1:  # 時間差太大：對時後重送
                    self.sync_time()
                    continue
                raise BinanceError(resp.status_code, code, (data.get("msg") if isinstance(data, dict) else "") or "")
            return data
        raise RuntimeError("unreachable")

    def sync_time(self) -> None:
        server = self._request("GET", "/fapi/v1/time")["serverTime"]
        self._offset = int(server) - int(time.time() * 1000)

    # ---------- 行情（不用金鑰） ----------
    def exchange_info(self) -> dict:
        return self._request("GET", "/fapi/v1/exchangeInfo")

    def klines(self, symbol: str, interval: str, limit: int = 500) -> pd.DataFrame:
        """最近 limit 根 K 線（time 是開盤時間，UTC）。最後一根通常還沒收盤。"""
        rows = self._request("GET", "/fapi/v1/klines",
                             {"symbol": symbol, "interval": INTERVALS.get(interval, interval), "limit": limit})
        df = pd.DataFrame([r[:6] for r in rows], columns=["time", "open", "high", "low", "close", "volume"])
        for col in ("open", "high", "low", "close", "volume"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["time"] = pd.to_datetime(df["time"].astype("int64"), unit="ms", utc=True)
        return df

    def price(self, symbol: str) -> float:
        return float(self._request("GET", "/fapi/v1/ticker/price", {"symbol": symbol})["price"])

    # ---------- 帳戶（要金鑰） ----------
    def balance_usdt(self) -> float:
        """USDT 的帳戶權益（錢包餘額 + 未實現損益）。"""
        for row in self._request("GET", "/fapi/v2/balance", signed=True):
            if row.get("asset") == "USDT":
                return float(row["balance"]) + float(row.get("crossUnPnl") or 0)
        return 0.0

    def positions(self) -> dict[tuple[str, str], float]:
        """(代號, LONG/SHORT) → 持倉數量（正數）。只列有倉位的。"""
        out = {}
        for p in self._request("GET", "/fapi/v2/positionRisk", signed=True):
            amt = abs(float(p.get("positionAmt") or 0))
            if amt > 0:
                side = p.get("positionSide") or ("LONG" if float(p["positionAmt"]) > 0 else "SHORT")
                out[(p["symbol"], side)] = amt
        return out

    def hedge_mode(self) -> bool:
        return bool(self._request("GET", "/fapi/v1/positionSide/dual", signed=True).get("dualSidePosition"))

    def set_hedge_mode(self) -> None:
        self._request("POST", "/fapi/v1/positionSide/dual", {"dualSidePosition": "true"}, signed=True)

    def set_leverage(self, symbol: str, leverage: int) -> None:
        self._request("POST", "/fapi/v1/leverage", {"symbol": symbol, "leverage": leverage}, signed=True)

    def set_isolated(self, symbol: str) -> None:
        try:
            self._request("POST", "/fapi/v1/marginType", {"symbol": symbol, "marginType": "ISOLATED"}, signed=True)
        except BinanceError as e:
            if e.code != -4046:  # -4046：本來就是逐倉
                raise

    def market_order(self, symbol: str, side: str, position_side: str, quantity: str, client_id: str) -> dict:
        """市價單。side 是 BUY/SELL，position_side 是 LONG/SHORT（雙向持倉）。回傳含 avgPrice、executedQty。"""
        return self._request("POST", "/fapi/v1/order", {
            "symbol": symbol, "side": side, "positionSide": position_side, "type": "MARKET",
            "quantity": quantity, "newClientOrderId": client_id, "newOrderRespType": "RESULT"}, signed=True)

    def conditional_order(self, symbol: str, side: str, position_side: str, kind: str, trigger: str,
                          quantity: str | None = None, client_id: str | None = None) -> dict:
        """條件單：kind 是 STOP_MARKET 或 TAKE_PROFIT_MARKET，用標記價格觸發。quantity 為 None 時全部平倉。"""
        params = {"algoType": "CONDITIONAL", "symbol": symbol, "side": side, "positionSide": position_side,
                  "type": kind, "triggerPrice": trigger, "workingType": "MARK_PRICE",
                  "clientAlgoId": client_id}
        if quantity is None:
            params["closePosition"] = "true"
        else:
            params["quantity"] = quantity
        return self._request("POST", "/fapi/v1/algoOrder", params, signed=True)

    def cancel_conditional(self, algo_id) -> None:
        try:
            self._request("DELETE", "/fapi/v1/algoOrder", {"algoId": algo_id}, signed=True)
        except BinanceError as e:
            if e.status != 400:  # 已經觸發或取消過的單會回 400，當作取消成功
                raise

    def algo_order(self, algo_id) -> dict:
        """查一張條件單（含已觸發、已取消的）；algoStatus 是 TRIGGERED / FINISHED 表示已成交。"""
        return self._request("GET", "/fapi/v1/algoOrder", {"algoId": algo_id}, signed=True)

    def open_conditionals(self, symbol: str) -> list[dict]:
        data = self._request("GET", "/fapi/v1/openAlgoOrders", {"symbol": symbol}, signed=True)
        return data if isinstance(data, list) else data.get("orders", [])

    def realized_since(self, symbol: str, position_side: str, start_ms: int) -> float:
        """某個代號、某一邊（LONG/SHORT）從 start_ms 起的已實現損益減手續費（USDT）。"""
        total = 0.0
        for t in self._request("GET", "/fapi/v1/userTrades",
                               {"symbol": symbol, "startTime": start_ms, "limit": 1000}, signed=True):
            if t.get("positionSide", position_side) != position_side:
                continue
            total += float(t.get("realizedPnl") or 0) - float(t.get("commission") or 0)
        return total
