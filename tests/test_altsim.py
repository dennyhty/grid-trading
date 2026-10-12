import numpy as np
import pandas as pd
import pytest

from grid_trading.altsim import Bars, sim
from grid_trading.signals import LONG, SHORT, Signal
from grid_trading.signals.tracker import track


def make_bars(seed: int, n: int = 400) -> Bars:
    rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    h, l = c * (1 + rng.uniform(0, 0.01, n)), c * (1 - rng.uniform(0, 0.01, n))
    t = pd.date_range("2026-01-01", periods=n, freq="4h", tz="UTC")
    df = pd.DataFrame({"time": t, "open": c, "high": h, "low": l, "close": c, "volume": 1.0})
    return Bars("X", "4h", t.as_unit("ns").astype("int64").to_numpy(), c, h, l, c, np.ones(n),
                h.tolist(), l.tolist(), c.tolist(), df)


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("d", [1, -1])
def test_sim_matches_tracker(seed, d):
    b = make_bars(seed)
    for i in range(10, 300, 7):
        price, risk = b.cl[i], 0.02 * b.cl[i]
        tps = [price + d * k * risk for k in (1, 2, 3)]
        sig = Signal("X", "4H", LONG if d > 0 else SHORT, b.df["time"].iloc[i] + pd.Timedelta(hours=4),
                     price, price - d * risk, tps, "t")
        ref = track(sig, b.df, 0.0005, 30)
        got = sim(b, i, d, risk, tps, 0.0005, 30)
        if ref.status == "CLOSED":
            assert got is not None and got[0] == pytest.approx(ref.realized_r)
        else:
            assert got is None
