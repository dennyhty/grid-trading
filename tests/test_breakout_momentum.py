import numpy as np
import pandas as pd

from grid_trading import signals


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
