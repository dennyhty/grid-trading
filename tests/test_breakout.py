import numpy as np
import pandas as pd

from grid_trading.breakout import BreakoutConfig, add_indicators, signal_at


def _uptrend_with_breakout(n: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    close = 100 * np.exp(np.cumsum(0.001 + 0.004 * rng.standard_normal(n)))
    close[-1] = close[-21:-1].max() * 1.02  # 最後一根突破
    time = pd.date_range("2026-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame({"time": time, "open": close, "high": close * 1.002,
                         "low": close * 0.998, "close": close, "volume": 1.0})


def test_swing_low_does_not_peek_ahead():
    df = _uptrend_with_breakout()
    full = add_indicators(df)
    cut = add_indicators(df.iloc[:300])
    assert np.allclose(full["swing_low"].iloc[:300].dropna(), cut["swing_low"].dropna())


def test_signal_levels_are_ordered():
    df = add_indicators(_uptrend_with_breakout())
    cfg = BreakoutConfig(rsi_max=100, use_4h_filter=False, min_risk=0.0)
    sig = signal_at(df.iloc[-1], cfg)
    assert sig is not None
    assert sig["stop"] < sig["entry"] < sig["tp1"] < sig["tp2"]
    assert abs((sig["tp1"] - sig["entry"]) - (sig["entry"] - sig["stop"])) < 1e-9
