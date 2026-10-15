import time

import numpy as np
import pandas as pd

from grid_trading import signals
from grid_trading.signals.engine import Engine


def _klines(n, start_ms, step_ms=3_600_000, price=None):
    price = np.linspace(100, 130, n) if price is None else price
    return {"klines": [{"time": start_ms + i * step_ms, "open": p, "high": p * 1.002, "low": p * 0.998,
                        "close": p, "volume": 1} for i, p in enumerate(price)]}


def test_1h_and_2h_share_one_1h_fetch(tmp_path, monkeypatch):
    import grid_trading.signals.engine as eng_mod

    calls = []

    class FakeApi:
        def _get(self, path, params):
            calls.append((params["symbol"], params["interval"]))
            end = params.get("endTime") or int(time.time() * 1000)
            end -= end % 7_200_000
            return _klines(params["limit"], end - params["limit"] * 3_600_000)

    eng = Engine([], ["BTC_USDT_PERP"], ["2H", "60M"], state_dir=tmp_path)
    monkeypatch.setattr(eng, "client", lambda: FakeApi())
    monkeypatch.setattr(eng_mod.time, "sleep", lambda s: None)
    frames = eng.klines_many([("BTC_USDT_PERP", "2H"), ("BTC_USDT_PERP", "60M")])
    assert {iv for _, iv in calls} == {"60M"}  # 2H 不另外抓
    assert len(calls) == 2  # 1000 根 1H = 兩頁，1H 和 2H 共用
    assert len(frames[("BTC_USDT_PERP", "60M")]) == 1000 and len(frames[("BTC_USDT_PERP", "2H")]) == 500

    # 只掃 2H 時也把抓到的 1H 留下，下一個 Engine（完整掃描）不用再抓
    calls.clear()
    quick = Engine([], ["BTC_USDT_PERP"], ["2H"], state_dir=tmp_path)
    monkeypatch.setattr(quick, "client", lambda: FakeApi())
    got = quick.klines_many([("BTC_USDT_PERP", "2H")])
    assert ("BTC_USDT_PERP", "60M") in got
    n = len(calls)
    eng.prefetched = got
    eng.klines_many([("BTC_USDT_PERP", "2H"), ("BTC_USDT_PERP", "60M")])
    assert len(calls) == n


def test_breakout_is_an_observation_group_and_vegas_skips_1h():
    b = signals.get("breakout_momentum")
    assert b.tag and b.intervals == ("60M",) and b.markets == ("crypto",)
    assert not b.one_per_coin
    v = signals.get("vegas_trend")
    assert "60M" not in v.intervals


def test_breakout_fires_on_a_strong_volume_breakout():
    n = 1200
    t = pd.date_range("2025-01-01", periods=n, freq="h", tz="UTC")
    rng = np.random.default_rng(0)
    close = 100 * np.exp(np.cumsum(0.0008 + rng.normal(0, 0.002, n)))  # 慢慢往上，4H 在通道上方
    close[-1] = close[-21:-1].max() * 1.04  # 最後一根大漲突破
    df = pd.DataFrame({"time": t, "open": np.r_[close[0], close[:-1]], "close": close})
    df["high"] = df[["open", "close"]].max(axis=1) * 1.001
    df["low"] = df[["open", "close"]].min(axis=1) * 0.999
    df["volume"] = 100.0
    df.loc[n - 1, "volume"] = 500.0
    strat = signals.get("breakout_momentum")
    p = strat.prepare(df)
    assert strat.candidates(p)[-1]
    side, sl, tps, info = strat.check(p, n - 1)
    c = close[-1]
    assert side == "LONG" and sl < c and np.allclose(tps, [c + (c - sl), c + 2 * (c - sl)])
    assert info["details"]

    # 量不夠就不做
    df.loc[n - 1, "volume"] = 150.0
    assert strat.check(strat.prepare(df), n - 1) is None
