from grid_trading.markets import COMMODITY, CRYPTO, STOCK, classify, display_name


def test_tokenized_stocks_end_with_x():
    for s in ("NVDAX_USDT_PERP", "AAPLX_USDT_PERP", "MUX_USDT_PERP", "QQQX_USDT_PERP", "LRCXX_USDT_PERP"):
        assert classify(s) == STOCK, s
    assert display_name("MUX_USDT_PERP") == "MU"
    assert display_name("LRCXX_USDT_PERP") == "LRCX"


def test_crypto_ending_in_x_stays_crypto():
    for s in ("AVAX_USDT_PERP", "TRX_USDT_PERP", "IMX_USDT_PERP", "BTC_USDT_PERP", "PUMP_USDT_PERP", "龙虾_USDT_PERP"):
        assert classify(s) == CRYPTO, s


def test_commodities():
    for s in ("XAU_USDT_PERP", "XAG_USDT_PERP", "WTI_USDT_PERP", "BRENTOIL_USDT_PERP", "PAXG_USDT_PERP"):
        assert classify(s) == COMMODITY, s
    assert display_name("XAU_USDT_PERP") == "XAU（黃金）"
