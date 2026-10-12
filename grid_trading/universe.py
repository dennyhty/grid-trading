"""CoinMarketCap 市值前 100 名（扣掉穩定幣）中，派網有 USDT 永續合約的幣種。

TOP100 是 2026-10-12 從 CoinMarketCap 抓下來的快照，依市值排序。前 100 名裡派網沒有合約的
（LEO、XAUt、M、BGB、MORPHO、GT、JST、KCS、XDC、FLR、NEXO）不遞補，所以只有 88 個。
要更新名單：python -c "from grid_trading.universe import fetch_cmc_top; print(fetch_cmc_top())"
"""

from __future__ import annotations

import requests

CMC_LISTING = "https://api.coinmarketcap.com/data-api/v3/cryptocurrency/listing"

TOP100 = [
    "BTC", "ETH", "BNB", "XRP", "SOL", "TRX", "ZEC", "HYPE", "DOGE", "XMR",
    "LINK", "ADA", "NEAR", "XLM", "BCH", "LTC", "CC", "AVAX", "UNI", "SUI",
    "HBAR", "GRAM", "TAO", "SHIB", "ENA", "CRO", "QNT", "OKB", "PUMP", "AAVE",
    "ONDO", "DOT", "WLD", "ASTER", "ICP", "MNT", "SKY", "PAXG", "WLFI", "PEPE",
    "ETC", "ARB", "VVV", "JUP", "ALGO", "POL", "KAS", "RENDER", "FIL", "ATOM",
    "ZRO", "LIT", "PI", "AERO", "INJ", "NIGHT", "APT", "CAKE", "STABLE", "STRK",
    "ETHFI", "STX", "PYTH", "DASH", "VET", "RAY", "CRV", "币安人生", "TRUMP", "FET",
    "TIA", "SEI", "PENGU", "VIRTUAL", "XPL", "BSV", "PIEVERSE", "FF", "PENDLE", "IMX",
    "LDO", "XTZ", "SPX", "KITE", "CFX", "SUN", "OP", "KAIA",
]


def fetch_cmc_top(n: int = 100) -> list[str]:
    """CoinMarketCap 市值前 n 名（不含穩定幣）的代號，依市值排序。不需要 API Key。"""
    params = {"start": 1, "limit": n + 50, "sortBy": "market_cap", "sortType": "desc", "convert": "USD"}
    data = requests.get(CMC_LISTING, params=params, timeout=20).json()["data"]["cryptoCurrencyList"]
    coins = [c["symbol"] for c in data if not {"stablecoin", "fiat-stablecoin"} & set(c.get("tags") or [])]
    return coins[:n]


def perp_symbols(coins: list[str] = TOP100) -> list[str]:
    return [f"{c}_USDT_PERP" for c in coins]
