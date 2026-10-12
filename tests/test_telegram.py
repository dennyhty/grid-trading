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
    _write(data / "state.json", {"signals": [{"id": "b", "sl": 95.0, "tps": [105.0, 110.0, 115.0],
                                               "info": {"strict_rsi": False}}]})
    out = build_outbox(data, tmp_path / "prev.json")
    assert out == ["BTC/USDT 4H 做多訊號 @ 100\n止損 95\nTP1 105 / TP2 110 / TP3 115\nRSI 收緊觀察組：會剔除（只有現行 RSI 範圍才進）", "到 TP1"]


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
