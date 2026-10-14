import signal_snapshot
from grid_trading.pionex import PionexPublic


def test_perp_specs_parses_limits(monkeypatch):
    data = {"symbols": [
        {"symbol": "BTC_USDT_PERP", "minSizeMarket": "0.0001", "baseStep": "0.0001", "minNotional": "1"},
        {"symbol": "BAD_USDT_PERP", "minSizeMarket": "x"},
    ]}
    monkeypatch.setattr(PionexPublic, "_get", lambda self, path, params=None: data)
    specs = PionexPublic(pause=0).perp_specs()
    assert specs == {"BTC_USDT_PERP": {"min_size": 0.0001, "step": 0.0001, "min_notional": 1.0}}


def test_order_specs_keeps_previous_on_error(monkeypatch):
    def boom(self):
        raise RuntimeError("429")
    monkeypatch.setattr(PionexPublic, "perp_specs", boom)
    prev = {"ETH_USDT_PERP": {"min_size": 0.001, "step": 0.001, "min_notional": 1.0}}
    assert signal_snapshot.order_specs({"ETH_USDT_PERP", "XRP_USDT_PERP"}, prev) == prev
