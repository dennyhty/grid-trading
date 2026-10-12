import pandas as pd

from grid_trading.signals import ledger


def _sig(sid, status="OPEN", r=0.0, strict=True, time=0):
    events = [{"kind": "TP3", "time": time + 1, "price": 1, "text": ""}] if status == "CLOSED" else []
    return {"id": sid, "symbol": "BTC_USDT_PERP", "interval": "4H", "strategy": "vegas_trend", "market": "crypto",
            "side": "LONG", "time": time, "price": 100.0, "sl": 90.0, "tps": [110.0, 120.0, 130.0],
            "info": {"strict_rsi": strict},
            "track": {"status": status, "realized_r": r, "tps_hit": 3 if status == "CLOSED" else 0, "events": events}}


def _candles(start_ms, highs, lows):
    t = pd.to_datetime([start_ms + i * 14_400_000 for i in range(len(highs))], unit="ms", utc=True)
    return pd.DataFrame({"time": t, "open": 100.0, "high": highs, "low": lows, "close": 100.0})


def test_update_keeps_closed_and_retracks_aged_out_open():
    book = ledger.update([], [_sig("a", "CLOSED", 2.0), _sig("b", strict=False)], lambda s, i: None)
    assert [e["status"] for e in book] == ["CLOSED", "OPEN"]
    # 「b」離開掃描範圍後，用 K 線重新追蹤：一路漲到 TP3
    book = ledger.update(book, [], lambda s, i: _candles(0, [105, 111, 121, 131], [99, 106, 112, 122]))
    b = next(e for e in book if e["id"] == "b")
    assert b["status"] == "CLOSED" and b["exit"] == "TP3" and b["r"] > 1.9
    # 已結束的不會被這次掃描覆蓋
    book = ledger.update(book, [_sig("a", "OPEN")], lambda s, i: None)
    assert next(e for e in book if e["id"] == "a")["status"] == "CLOSED"


def test_summary_groups():
    book = ledger.update([], [_sig("a", "CLOSED", 2.0, True), _sig("b", "CLOSED", -1.0, False, 5),
                              _sig("c", "OPEN", 0, True, 9)], lambda s, i: None)
    g = ledger.summary(book)["groups"]["vegas_trend|crypto|4H"]
    assert g["all"]["trades"] == 2 and g["all"]["open"] == 1 and abs(g["all"]["expectancy_r"] - 0.5) < 1e-9
    assert g["strict"]["trades"] == 1 and g["strict"]["expectancy_r"] == 2.0 and g["strict"]["open"] == 1
    assert g["excluded"]["trades"] == 1 and g["excluded"]["win_rate"] == 0
