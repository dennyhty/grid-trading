import json

from signal_telegram import build_outbox, send


def _write(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False))


def test_outbox_only_new_and_not_initial(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    old = {"id": "a|NEW|1", "signal_id": "a", "kind": "NEW", "text": "舊", "initial": False}
    _write(tmp_path / "prev.json", [old])
    _write(data / "notifications.json", [
        old,
        {"id": "b|NEW|2", "signal_id": "b", "kind": "NEW", "text": "BTC/USDT 4H 做多訊號 @ 100", "initial": False},
        {"id": "a|TP1|3", "signal_id": "a", "kind": "TP1", "text": "到 TP1", "initial": False},
        {"id": "c|NEW|4", "signal_id": "c", "kind": "NEW", "text": "啟動時補的", "initial": True},
    ])
    _write(data / "state.json", {"signals": [{"id": "b", "sl": 95.0, "tps": [105.0, 110.0, 115.0]}]})
    out = build_outbox(data, tmp_path / "prev.json")
    assert out == ["BTC/USDT 4H 做多訊號 @ 100\n止損 95\nTP1 105 / TP2 110 / TP3 115", "到 TP1"]


def test_outbox_empty_without_previous(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    _write(data / "notifications.json", [{"id": "x", "signal_id": "x", "kind": "NEW", "text": "t", "initial": False}])
    assert build_outbox(data, tmp_path / "missing.json") == []


def test_send_caps_messages(monkeypatch):
    sent = []
    monkeypatch.setattr("urllib.request.urlopen", lambda url, body, timeout: sent.append(body) or _Resp())
    monkeypatch.setattr("time.sleep", lambda s: None)
    ok, failed = send([f"m{i}" for i in range(40)], "TOKEN", "1")
    assert (ok, failed) == (30, 0)
    assert "%E9%82%84%E6%9C%89+11" in sent[-1].decode()  # 「還有 11」


class _Resp:
    def read(self):
        return b"{}"


def test_outbox_skips_observation_groups(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    _write(tmp_path / "prev.json", [{"id": "old"}])
    smc = "ETH_USDT_PERP|4H|smc_sweep|1"
    oi = "SOL_USDT_PERP|4H|oi_build|2"
    live = "BTC_USDT_PERP|4H|vegas_trend|3"
    _write(data / "notifications.json", [
        {"id": f"{sid}|NEW|9", "signal_id": sid, "kind": "NEW", "text": sid, "initial": False}
        for sid in (smc, oi, live)])
    _write(data / "state.json", {"signals": [], "strategies": {
        "vegas_trend": {"tag": ""}, "smc_sweep": {"tag": "SMC觀察組"}, "oi_build": {"tag": "訂單流觀察組"}}})
    assert build_outbox(data, tmp_path / "prev.json") == [live]


def test_send_retries_after_telegram_rate_limit(monkeypatch):
    import io
    import urllib.error

    calls = []

    def fake(url, body, timeout):
        calls.append(body)
        if len(calls) == 1:
            raise urllib.error.HTTPError(url, 429, "Too Many Requests", {},
                                         io.BytesIO(b'{"ok":false,"parameters":{"retry_after":3}}'))
        return _Resp()

    waits = []
    monkeypatch.setattr("urllib.request.urlopen", fake)
    monkeypatch.setattr("time.sleep", waits.append)
    assert send(["m1", "m2"], "TOKEN", "1") == (2, 0)
    assert len(calls) == 3 and 4.0 in waits


def test_send_does_not_retry_bad_requests(monkeypatch):
    import io
    import urllib.error

    calls = []

    def fake(url, body, timeout):
        calls.append(body)
        raise urllib.error.HTTPError(url, 400, "Bad Request", {}, io.BytesIO(b"{}"))

    monkeypatch.setattr("urllib.request.urlopen", fake)
    monkeypatch.setattr("time.sleep", lambda s: None)
    assert send(["m1"], "TOKEN", "1") == (0, 1)
    assert len(calls) == 1


def test_send_does_not_resend_after_timeout(monkeypatch):
    calls = []

    def fake(url, body, timeout):
        calls.append(body)
        raise TimeoutError("timed out")  # 訊息可能已送達，只是沒等到回應

    monkeypatch.setattr("urllib.request.urlopen", fake)
    monkeypatch.setattr("time.sleep", lambda s: None)
    assert send(["m1"], "TOKEN", "1") == (1, 0)
    assert len(calls) == 1


def test_outbox_skips_notifications_marked_observe(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    _write(tmp_path / "prev.json", [{"id": "old"}])
    stock = "NVDAX_USDT_PERP|4H|vegas_trend|1"
    live = "BTC_USDT_PERP|4H|vegas_trend|3"
    _write(data / "notifications.json", [
        {"id": f"{stock}|NEW|9", "signal_id": stock, "kind": "NEW", "text": stock, "initial": False, "observe": True},
        {"id": f"{live}|NEW|9", "signal_id": live, "kind": "NEW", "text": live, "initial": False, "observe": False}])
    _write(data / "state.json", {"signals": [], "strategies": {"vegas_trend": {"tag": ""}}})
    assert build_outbox(data, tmp_path / "prev.json") == [live]


def test_outbox_skips_notifications_already_pushed_early(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    _write(tmp_path / "prev.json", [{"id": "old"}])
    a, b = "BTC_USDT_PERP|4H|vegas_trend|1", "ETH_USDT_PERP|2H|vegas_trend|2"
    _write(data / "notifications.json", [
        {"id": f"{sid}|NEW|9", "signal_id": sid, "kind": "NEW", "text": sid, "initial": False} for sid in (a, b)])
    _write(data / "state.json", {"signals": [], "strategies": {"vegas_trend": {"tag": ""}}})
    assert build_outbox(data, tmp_path / "prev.json", {f"{a}|NEW|9"}) == [b]
