import signal_snapshot


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
