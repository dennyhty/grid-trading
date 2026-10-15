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


def test_events_are_matched_by_id_not_by_position(tmp_path):
    eng = Engine([], [], ["4H"], state_dir=tmp_path)
    now = int(time.time() * 1000)
    sid, s = _sig(now - 8 * 3_600_000, "4H")
    eng._notify({sid: s})  # 第一輪：只有 NEW
    eng.notifications.append({"id": "other", "signal_id": "other", "kind": "NEW", "time": 0, "created": 0,
                              "text": "", "initial": False})
    # TP1、TP2 同一根 K 線觸及：兩個同時間的 STOP_MOVED 都要通知，而且只通知一次
    t = now - 3_600_000
    s["track"]["events"] = [{"kind": "TP1", "time": t, "text": "TP1"},
                            {"kind": "STOP_MOVED", "time": t, "text": "保護止損設於 101"},
                            {"kind": "TP2", "time": t, "text": "TP2"},
                            {"kind": "STOP_MOVED", "time": t, "text": "保護止損設於 105"}]
    eng._notify({sid: s})
    eng._notify({sid: s})
    texts = [n["text"] for n in eng.notifications if n["signal_id"] == sid and n["kind"] != "NEW"]
    assert len(texts) == 4 and texts[-1].endswith("保護止損設於 105")

    # 出場規則改了、事件清單變短：之後真正的新事件還是要推播（舊做法用數量比對會漏掉）
    s["track"]["events"] = [{"kind": "TP1", "time": t, "text": "TP1"},
                            {"kind": "SL", "time": now, "text": "止損出場"}]
    eng._notify({sid: s})
    sl = [n for n in eng.notifications if n["signal_id"] == sid and n["kind"] == "SL"]
    assert len(sl) == 1 and sl[0]["initial"] is False

    # 改規則後補出來、比已通知事件還早的事件：記錄但不推播
    s["track"]["events"].insert(1, {"kind": "TRAIL_MOVED", "time": t + 60_000, "text": "追蹤止損移到 103"})
    eng._notify({sid: s})
    trail = [n for n in eng.notifications if n["signal_id"] == sid and n["kind"] == "TRAIL_MOVED"]
    assert len(trail) == 1 and trail[0]["initial"] is True


def test_trim_keeps_notifications_of_signals_still_on_the_site(tmp_path):
    eng = Engine([], [], ["4H"], state_dir=tmp_path)
    now = int(time.time() * 1000)
    sid, s = _sig(now - 30 * 3_600_000, "4H", events=[{"kind": "TP1", "time": now - 20 * 3_600_000, "text": "TP1"}])
    eng._notify({sid: s})
    eng.notifications += [{"id": f"x{i}", "signal_id": f"x{i}", "kind": "NEW", "time": 0, "created": 0,
                           "text": "", "initial": False} for i in range(400)]
    s["track"]["events"].append({"kind": "TP2", "time": now, "text": "TP2"})
    eng._notify({sid: s})
    mine = [n for n in eng.notifications if n["signal_id"] == sid]
    assert [n["kind"] for n in mine] == ["NEW", "TP1", "TP2"]
    assert mine[-1]["initial"] is False
    assert len(eng.notifications) == 300 + 2  # 其他的只留最近 300 則


def _candles(n, start_ms, bar_ms, price=100.0, highs=None):
    import pandas as pd
    rows = []
    for i in range(n):
        h = (highs or {}).get(i, price + 0.5)
        rows.append({"time": pd.Timestamp(start_ms + i * bar_ms, unit="ms", tz="UTC"), "open": price,
                     "high": h, "low": price - 0.5, "close": price, "volume": 1.0})
    df = pd.DataFrame(rows)
    df.attrs.update(incomplete=False, error="")
    return df


def test_open_signal_older_than_keep_days_is_still_tracked(tmp_path, monkeypatch):
    from grid_trading.signals.strategy import Strategy

    class Quiet(Strategy):
        name = "vegas_trend"

        def scan(self, df, symbol, interval, start=None):
            return []

    bar = 14_400_000
    now = int(time.time() * 1000) // bar * bar + 60_000
    df = _candles(300, now - 300 * bar, bar, highs={295: 106.0})  # 第 295 根碰到 TP1
    sig_time = int(df["time"].iloc[200].timestamp() * 1000)  # 約 16 天前，已經超過 keep_days=7
    sid = f"BTC_USDT_PERP|4H|vegas_trend|{sig_time // 1000}"
    old = {"id": sid, "label": "4H", "market": "crypto", "name": "BTC", "symbol": "BTC_USDT_PERP", "interval": "4H",
           "side": "LONG", "time": sig_time, "price": 100.0, "sl": 95.0, "tps": [105.0, 110.0, 115.0],
           "strategy": "vegas_trend", "info": {}, "track": {"status": "OPEN", "events": []}}
    eng = Engine([Quiet()], ["BTC_USDT_PERP"], ["4H"], state_dir=tmp_path)
    eng.notifications = [{"id": f"{sid}|NEW|{sig_time}", "signal_id": sid, "kind": "NEW", "time": sig_time,
                          "created": sig_time, "text": "", "initial": False}]
    eng.previous = {sid: old}
    monkeypatch.setattr(eng, "klines_many", lambda jobs, bars=1000: {j: df for j in jobs})
    eng.scan_once()
    assert sid in eng.signals and eng.signals[sid]["track"]["tps_hit"] == 1
    tp1 = [n for n in eng.notifications if n["signal_id"] == sid and n["kind"] == "TP1"]
    assert len(tp1) == 1 and tp1[0]["initial"] is False

    # 已結束的單：這一輪推完出場通知，下一輪就不再追蹤
    eng.previous = {sid: {**old, "track": {"status": "CLOSED", "events": []}}}
    eng.scan_once()
    assert sid not in eng.signals


def test_open_signal_dropped_by_a_rule_change_is_still_tracked(tmp_path, monkeypatch):
    import pandas as pd

    from grid_trading.signals.strategy import Signal, Strategy

    bar = 14_400_000
    now = int(time.time() * 1000) // bar * bar + 60_000
    df = _candles(300, now - 300 * bar, bar, highs={295: 106.0})  # 第 295 根碰到 TP1
    sig_time = int(df["time"].iloc[290].timestamp() * 1000)  # 約 2 天前，還在掃描範圍內
    sid = f"BTC_USDT_PERP|4H|vegas_trend|{sig_time // 1000}"
    old = {"id": sid, "label": "4H", "market": "crypto", "name": "BTC", "symbol": "BTC_USDT_PERP", "interval": "4H",
           "side": "LONG", "time": sig_time, "price": 100.0, "sl": 95.0, "tps": [105.0, 110.0, 115.0],
           "strategy": "vegas_trend", "info": {}, "track": {"status": "OPEN", "events": []}}
    t = lambda i: df["time"].iloc[i] + pd.Timedelta(milliseconds=bar)  # noqa: E731

    class NewRule(Strategy):
        """新規則不再出現 sig_time 那一單；另外多了兩單：一單會開到那一單之後，一單在那一單持有期間。"""
        name = "vegas_trend"

        def scan(self, df, symbol, interval, start=None):
            mk = lambda i: Signal(symbol, interval, "LONG", t(i), 100.0, 95.0, [105.0, 110.0, 115.0], self.name)  # noqa: E731
            return [mk(280), mk(292)]

    eng = Engine([NewRule()], ["BTC_USDT_PERP"], ["4H"], state_dir=tmp_path)
    eng.previous = {sid: old}
    monkeypatch.setattr(eng, "klines_many", lambda jobs, bars=1000: {j: df for j in jobs})
    eng.scan_once()
    assert sid in eng.signals and eng.signals[sid]["track"]["tps_hit"] == 1
    # 兩單新的都會和已經發出、還沒結束的那一單重疊，不出
    assert set(eng.signals) == {sid}


def _pos(interval, t, side="LONG", status="OPEN", end=None, market="crypto", sym="BTC_USDT_PERP"):
    sid = f"{sym}|{interval}|vegas_trend|{t // 1000}"
    events = [{"kind": "SL", "time": end, "text": ""}] if status == "CLOSED" else []
    return sid, {"id": sid, "symbol": sym, "interval": interval, "strategy": "vegas_trend", "market": market,
                 "side": side, "time": t, "track": {"status": status, "events": events}}


def _one_per_coin_engine(tmp_path, intervals=("4H", "2H")):
    import pandas as pd
    from grid_trading.signals import get
    eng = Engine([get("vegas_trend")], [], list(intervals), state_dir=tmp_path)
    return eng, pd.Timestamp.now(tz="UTC")


def test_one_position_per_coin_and_side_across_2h_and_4h(tmp_path):
    eng, now = _one_per_coin_engine(tmp_path)
    t = int(now.timestamp() * 1000) - 10 * 3_600_000
    h = 3_600_000
    a_id, a = _pos("4H", t)                                     # 已經發出、還開著
    eng.previous = {a_id: a}
    b_id, b = _pos("2H", t + 2 * h)                             # 同幣同方向 → 不做
    c_id, c = _pos("2H", t + 2 * h, sym="ETH_USDT_PERP")         # 別的幣 → 照做
    d_id, d = _pos("2H", t + 4 * h, side="SHORT")                # 反方向 → 照做
    e_id, e = _pos("2H", t + 2 * h, market="stock", sym="NVDAX_USDT_PERP")  # 觀察組不管
    found = {a_id: a, b_id: b, c_id: c, d_id: d, e_id: e}
    eng._one_per_coin(found, now)
    assert set(found) == {a_id, c_id, d_id, e_id}
    assert b_id in eng.blocked

    # 擋住它的單出場、甚至離開掃描範圍以後，被擋掉的訊號也不會再出現
    eng.previous = {c_id: c, d_id: d}
    found = {b_id: b, c_id: c, d_id: d}
    eng._one_per_coin(found, now)
    assert b_id not in found


def test_one_per_coin_ties_go_to_2h_and_closed_blockers_free_the_coin(tmp_path):
    eng, now = _one_per_coin_engine(tmp_path)
    t = int(now.timestamp() * 1000) - 20 * 3_600_000
    a_id, a = _pos("4H", t)
    b_id, b = _pos("2H", t)
    found = {a_id: a, b_id: b}
    eng._one_per_coin(found, now)
    assert set(found) == {b_id}

    eng2, _ = _one_per_coin_engine(tmp_path)
    c_id, c = _pos("4H", t, status="CLOSED", end=t + 4 * 3_600_000)
    d_id, d = _pos("2H", t + 6 * 3_600_000)
    eng2.previous = {c_id: c}
    found = {c_id: c, d_id: d}
    eng2._one_per_coin(found, now)
    assert set(found) == {c_id, d_id}


def test_one_per_coin_uses_previous_signals_of_intervals_not_scanned(tmp_path):
    # 收盤搶先掃描只掃 2H，4H 還開著的單從上一次的訊號來
    eng, now = _one_per_coin_engine(tmp_path, intervals=("2H",))
    t = int(now.timestamp() * 1000) - 10 * 3_600_000
    a_id, a = _pos("4H", t)
    b_id, b = _pos("2H", t + 4 * 3_600_000)
    eng.previous = {a_id: a}
    found = {b_id: b}
    eng._one_per_coin(found, now)
    assert found == {}
