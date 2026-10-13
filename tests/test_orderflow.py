import numpy as np
import pandas as pd

from grid_trading.orderflow import add_features, prior_peak
from grid_trading.signals import get
from grid_trading.signals import orderflow  # noqa: F401


def test_prior_peak_looks_only_at_the_window_before_gap():
    v = np.array([1, 5, 2, 3, 9, 4, 0], dtype=float)
    out = prior_peak(v, lookback=4, gap=2, high=True)
    # i=4：看 v[0..2] → 位置 1；i=6：看 v[2..4] → 位置 4
    assert list(out[:4]) == [-1, -1, -1, -1]
    assert out[4] == 1 and out[5] == 1 and out[6] == 4
    assert prior_peak(v, 4, 2, high=False)[6] == 2


def _flow(close, delta, oi=None):
    n = len(close)
    df = pd.DataFrame({"time": pd.date_range("2024-01-01", periods=n, freq="4h", tz="UTC"), "close": close})
    df["open"] = df["close"].shift(1).fillna(df["close"])
    df["high"] = df[["open", "close"]].max(axis=1) * 1.002
    df["low"] = df[["open", "close"]].min(axis=1) * 0.998
    df["volume"] = 100.0
    df["quote_volume"] = 1000.0
    df["taker_buy_quote"] = (1000 + np.asarray(delta)) / 2
    df["delta"] = np.asarray(delta, dtype=float)
    df["cvd"] = df["delta"].cumsum()
    df["oi"] = np.nan if oi is None else oi
    for c in ("oi_value", "account_ratio", "top_account_ratio", "top_position_ratio", "taker_ratio"):
        df[c] = np.nan
    return df


def test_bearish_cvd_divergence_gives_short_reversal():
    rng = np.random.default_rng(1)
    n = 260
    close = 100 + np.cumsum(rng.normal(0, 0.3, n))
    delta = rng.normal(0, 10, n)
    # 第 240 根是前高，第 255 根價格更高但這段主動賣比較多（CVD 較低）
    close[240] = close[:255].max() + 1
    delta[240] = 300
    close[255] = close[240] + 1
    delta[241:256] = -40
    sig = get("of_cvd_div", lookback=20, mode="reversal").prepare(_flow(close, delta))["sig"]
    assert sig.iat[255] == -1


def test_taker_follow_and_fade_are_mirrors():
    rng = np.random.default_rng(2)
    n = 300
    close = 100 + np.cumsum(rng.normal(0, 0.3, n))
    delta = rng.normal(0, 20, n)
    delta[280] = 600
    df = _flow(close, delta)
    follow = get("of_taker", n=1, k=2.0, mode="follow").prepare(df)["sig"]
    fade = get("of_taker", n=1, k=2.0, mode="fade").prepare(df)["sig"]
    assert follow.iat[280] == 1 and fade.iat[280] == -1
    assert (follow == -fade).all()


def test_features_do_not_peek_ahead():
    rng = np.random.default_rng(3)
    n = 300
    df = _flow(100 + np.cumsum(rng.normal(0, 0.3, n)), rng.normal(0, 20, n), oi=1000 + np.cumsum(rng.normal(0, 5, n)))
    full = add_features(df)
    cut = add_features(df.iloc[:250].copy())
    cols = ["imb3_z", "doi3_z", "ret6_z", "trend"]
    pd.testing.assert_frame_equal(full.loc[:249, cols], cut[cols])
