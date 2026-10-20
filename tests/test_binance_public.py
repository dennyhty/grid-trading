from grid_trading import binance_public as bp


class FakeApi:
    def usdt_perps(self):
        return {"BTCUSDT", "ETHUSDT", "1000PEPEUSDT", "POLUSDT"}


def test_universe_uses_binance_contract_names():
    syms = bp.universe(FakeApi())
    assert "BTC_USDT_PERP" in syms and "1000PEPE_USDT_PERP" in syms and "POL_USDT_PERP" in syms
    assert "PEPE_USDT_PERP" not in syms and "XRP_USDT_PERP" not in syms
    assert bp.to_binance("1000PEPE_USDT_PERP") == "1000PEPEUSDT"
    assert bp.from_binance("BTCUSDT") == "BTC_USDT_PERP"


def test_universe_falls_back_when_exchange_info_fails():
    class Down:
        def usdt_perps(self):
            raise RuntimeError("timeout")
    syms = bp.universe(Down())
    assert "BTC_USDT_PERP" in syms and "CRO_USDT_PERP" not in syms and "1000SHIB_USDT_PERP" in syms
