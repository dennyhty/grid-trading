"""把派網永續合約分成加密貨幣、美股（代幣化股票/ETF）、商品三類。

派網的交易對資料沒有類別欄位，所以用名稱判斷：
- 代幣化美股/ETF 的代號是「股票代號 + X」，例如 NVDAX、AAPLX、QQQX、MUX（美光）。
  但有些加密貨幣本來就以 X 結尾（AVAX、IMX、TRX…），列在 CRYPTO_ENDING_X 排除。
- 商品：貴金屬、原油、天然氣、銅。
"""

from __future__ import annotations

CRYPTO = "crypto"
STOCK = "stock"
COMMODITY = "commodity"
LABELS = {CRYPTO: "加密貨幣", STOCK: "美股", COMMODITY: "商品"}

COMMODITIES = {
    "XAU": "黃金", "XAUT": "黃金（XAUT）", "PAXG": "黃金（PAXG）", "XAG": "白銀", "XPT": "鉑金", "XPD": "鈀金",
    "WTI": "WTI 原油", "BRENTOIL": "布蘭特原油", "NATGAS": "天然氣", "COPPER": "銅",
}

# 以 X 結尾、但其實是加密貨幣的代號
CRYPTO_ENDING_X = {
    "AVAX", "BEAMX", "CFX", "FRAX", "GMX", "IMX", "INX", "MYX", "POLYX", "SLX", "SNX", "SPX", "STX", "TRX",
    "ZRX", "ARX", "CVX", "APEX", "FLUX", "OX", "HMX", "ELX", "AIX", "XAIX", "DEXX", "LUX", "NMX", "REX",
}


def base(symbol: str) -> str:
    return symbol.split("_")[0]


def classify(symbol: str) -> str:
    b = base(symbol)
    if b in COMMODITIES:
        return COMMODITY
    if b.endswith("X") and len(b) >= 2 and b.isascii() and b.isalpha() and b not in CRYPTO_ENDING_X:
        return STOCK
    return CRYPTO


def display_name(symbol: str) -> str:
    """畫面上顯示的名稱：股票去掉結尾的 X，商品加中文名。"""
    b = base(symbol)
    m = classify(symbol)
    if m == COMMODITY:
        return f"{b}（{COMMODITIES[b]}）" if not COMMODITIES[b].startswith("黃金（") else f"{b}（黃金）"
    if m == STOCK:
        return b[:-1]
    return b
