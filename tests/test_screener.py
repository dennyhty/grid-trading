import numpy as np
import pandas as pd

from grid_trading import indicators as ind
from grid_trading.screener import analyze, rank


def make_df(close: np.ndarray, noise: float = 0.002) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    close = pd.Series(close)
    wick = close * noise * (1 + rng.random(len(close)))
    return pd.DataFrame({
        "open": close.shift(1).fillna(close.iloc[0]),
        "high": close + wick,
        "low": close - wick,
        "close": close,
        "volume": 1.0,
    })


def ranging(n=500):
    t = np.arange(n)
    return 100 + 3 * np.sin(t / 4)


def uptrend(n=500):
    return 100 * 1.004 ** np.arange(n)


def downtrend(n=500):
    return 100 * 0.996 ** np.arange(n)


def test_efficiency_ratio_separates_chop_from_trend():
    assert ind.efficiency_ratio(pd.Series(ranging()), 120) < 0.1
    assert ind.efficiency_ratio(pd.Series(uptrend()), 120) > 0.9


def test_adx_high_in_trend_low_in_range():
    assert ind.adx(make_df(uptrend())).iloc[-1] > 40
    assert ind.adx(make_df(ranging())).iloc[-1] < ind.adx(make_df(uptrend())).iloc[-1]


def test_direction_follows_vegas_channel():
    assert analyze(make_df(uptrend()))["direction"] == "做多"
    assert analyze(make_df(downtrend()))["direction"] == "做空"
    assert analyze(make_df(ranging()))["direction"] == "中性"


def test_range_and_stops_bracket_price():
    r = analyze(make_df(ranging()))
    assert r["lower"] < r["price"] < r["upper"]
    assert r["stop_low"] < r["lower"] and r["stop_high"] > r["upper"]
    assert 5 <= r["grids"] <= 100
    assert r["grid_spacing_pct"] >= 0.3 - 1e-9


def test_long_grid_has_no_upper_stop():
    r = analyze(make_df(uptrend()))
    assert np.isnan(r["stop_high"]) and not np.isnan(r["stop_low"])


def test_too_few_bars_skipped():
    assert analyze(make_df(ranging(100))) is None


def test_ranging_coin_ranks_above_trending():
    table = rank({"CHOP": analyze(make_df(ranging())), "TREND": analyze(make_df(uptrend()))})
    assert list(table.index) == ["CHOP", "TREND"]
