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


def test_closing_intervals():
    h = 3_600_000
    assert signal_snapshot.closing_intervals(["4H", "2H"], 8 * h + 20_000) == ["4H", "2H"]
    assert signal_snapshot.closing_intervals(["4H", "2H"], 10 * h + 20_000) == ["2H"]
    assert signal_snapshot.closing_intervals(["4H", "2H"], 11 * h + 20_000) == []
    assert signal_snapshot.closing_intervals(["4H", "2H"], 10 * h + 15 * 60_000) == []


def test_retract_late_removes_signals_that_showed_up_after_their_bar_was_scanned():
    from signal_snapshot import LATE_SINCE_MS, retract_late

    h = 3_600_000
    created = LATE_SINCE_MS + 50 * h
    late = "VIRTUAL_USDT_PERP|2H|vegas_trend|1"
    ok = "POL_USDT_PERP|2H|vegas_trend|2"
    rwa = "NVDAX_USDT_PERP|2H|vegas_trend|3"
    notes = [
        {"signal_id": late, "kind": "NEW", "time": created - 140 * h, "created": created, "initial": True, "observe": False},
        {"signal_id": late, "kind": "TP1", "time": created - 120 * h, "created": created, "initial": True, "observe": False},
        {"signal_id": ok, "kind": "NEW", "time": created - 60_000, "created": created, "initial": False, "observe": False},
        {"signal_id": rwa, "kind": "NEW", "time": created - 140 * h, "created": created, "initial": True, "observe": True},
    ]
    ledger = [{"id": late}, {"id": ok}]
    prev = {"signals": [{"id": ok}]}
    assert retract_late(prev, notes, ledger) == {late}
    assert {n["signal_id"] for n in notes} == {ok, rwa}
    assert ledger == [{"id": ok}]
    # 上一次的 state 已經有 scanned：不再清
    assert retract_late({**prev, "scanned": {}}, notes, ledger) == set()
