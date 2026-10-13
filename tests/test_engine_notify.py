import time

from grid_trading.signals.engine import Engine, fetch_klines


def _sig(sid_time, interval="2H", events=()):
    sid = f"BTC_USDT_PERP|{interval}|vegas_trend|{sid_time // 1000}"
    return sid, {"id": sid, "label": interval, "market": "crypto", "name": "BTC", "symbol": "BTC_USDT_PERP",
                 "interval": interval, "side": "LONG", "time": sid_time, "price": 100.0,
                 "track": {"events": list(events)}}


def test_old_signals_from_a_new_interval_are_recorded_quietly(tmp_path):
    eng = Engine([], [], ["4H", "2H"], state_dir=tmp_path)
    eng.notifications = [{"id": "x", "signal_id": "x", "kind": "NEW", "time": 0, "created": 0, "text": "", "initial": False}]
    now = int(time.time() * 1000)
    old_id, old = _sig(now - 24 * 3_600_000, events=[{"kind": "TP1", "time": now - 20 * 3_600_000, "text": "TP1"}])
    new_id, new = _sig(now - 2 * 3_600_000)
    eng._notify({old_id: old, new_id: new})
    by_sig = {n["signal_id"]: n for n in eng.notifications if n["kind"] == "NEW"}
    assert by_sig[old_id]["initial"] is True
    assert all(n["initial"] for n in eng.notifications if n["signal_id"] == old_id)
    assert by_sig[new_id]["initial"] is False

    # 下一輪：舊訊號有新事件時照常推播
    old["track"]["events"].append({"kind": "TP2", "time": now, "text": "TP2"})
    eng.signals = {}
    eng._notify({old_id: old, new_id: new})
    tp2 = [n for n in eng.notifications if n["signal_id"] == old_id and n["kind"] == "TP2"]
    assert len(tp2) == 1 and tp2[0]["initial"] is False


def test_2h_klines_are_built_from_1h():
    import pandas as pd

    class FakeApi:
        def _get(self, path, params):
            assert params["interval"] == "60M"
            start = 1_700_000_000_000 - 1_700_000_000_000 % 7_200_000
            return {"klines": [{"time": start + i * 3_600_000, "open": i, "high": i + 1, "low": i - 1,
                                "close": i + 0.5, "volume": 1} for i in range(6)]}

    df = fetch_klines(FakeApi(), "BTC_USDT_PERP", "2H", 3)
    assert len(df) == 3
    assert list(df["open"]) == [0, 2, 4] and list(df["close"]) == [1.5, 3.5, 5.5]
    assert list(df["high"]) == [2, 4, 6] and (df["time"].diff().dropna() == pd.Timedelta(hours=2)).all()


def test_fetch_retries_then_marks_incomplete(monkeypatch):
    import grid_trading.signals.engine as eng_mod
    monkeypatch.setattr(eng_mod.time, "sleep", lambda s: None)

    class Flaky:
        def __init__(self, fail_first):
            self.calls = 0
            self.fail_first = fail_first

        def _get(self, path, params):
            self.calls += 1
            if self.calls <= self.fail_first:
                raise RuntimeError("429")
            return {"klines": [{"time": 1_700_000_000_000 + i * 3_600_000, "open": 1, "high": 1, "low": 1,
                                "close": 1, "volume": 1} for i in range(10)]}

    ok = fetch_klines(Flaky(2), "BTC_USDT_PERP", "60M", 10)
    assert len(ok) == 10 and not ok.attrs["incomplete"]
    bad = fetch_klines(Flaky(99), "BTC_USDT_PERP", "2H", 10)
    assert len(bad) == 0 and bad.attrs["incomplete"]


def test_failed_fetch_keeps_previous_signals(tmp_path, monkeypatch):
    import pandas as pd

    eng = Engine([], ["BTC_USDT_PERP"], ["2H"], state_dir=tmp_path)
    now = int(time.time() * 1000)
    sid, old = _sig(now - 5 * 3_600_000)
    eng.previous = {sid: old}
    bad = pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])
    bad.attrs.update(incomplete=True, error="HTTPError: 429")
    monkeypatch.setattr(eng, "klines_many", lambda jobs, bars=1000: {j: bad for j in jobs})
    eng.scan_once()
    assert eng.failed == [("BTC_USDT_PERP", "2H")] and eng.carried == 1
    assert sid in eng.signals
