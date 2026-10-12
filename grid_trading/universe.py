"""市值前 100 名（不含穩定幣）中派網有永續合約的幣種，依市值大約排序。

雲端環境連不到 CoinGecko 等市值 API，這份名單是依 2026 年中的市值排名手動整理，
再扣掉派網沒有永續合約的幣（TON、LEO、XDC、FLR、KCS、GT、NEXO、FLOW、EOS、MORPHO、IP、AB、ZK、DYDX、MINA、KSM）往後遞補。
"""

TOP100 = [
    "BTC", "ETH", "XRP", "BNB", "SOL", "TRX", "DOGE", "ADA", "HYPE", "LINK",
    "BCH", "XLM", "SUI", "AVAX", "HBAR", "LTC", "SHIB", "DOT", "XMR", "UNI",
    "PEPE", "AAVE", "APT", "NEAR", "ENA", "ONDO", "TAO", "ICP", "ETC", "WLD",
    "POL", "ARB", "KAS", "ALGO", "VET", "ATOM", "FIL", "RENDER", "TRUMP", "BONK",
    "OP", "SEI", "JUP", "FET", "INJ", "TIA", "STX", "IMX", "QNT", "CRV",
    "GRT", "WIF", "PENGU", "MNT", "CRO", "OKB", "VIRTUAL", "PUMP", "WLFI", "LDO",
    "SAND", "MANA", "THETA", "AXS", "GALA", "IOTA", "NEO", "XTZ", "KAIA", "PYTH",
    "JASMY", "CAKE", "ENS", "RAY", "ZEC", "DASH", "CFX", "AR", "PENDLE", "SPX",
    "FARTCOIN", "AERO", "S", "ZRO", "STRK", "FLOKI", "POPCAT", "KAITO", "BERA", "SYRUP",
    "XPL", "ASTER", "2Z", "LINEA", "APE", "COMP", "SNX", "CHZ", "1INCH", "RUNE",
]


def perp_symbols(coins: list[str] = TOP100) -> list[str]:
    return [f"{c}_USDT_PERP" for c in coins]
