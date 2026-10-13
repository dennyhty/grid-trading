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


def test_attach_open_interest_uses_snapshot_at_bar_close():
    from grid_trading.binance_live import attach_open_interest

    t0 = pd.Timestamp("2026-10-01", tz="UTC")
    bars = pd.DataFrame({"time": pd.date_range(t0, periods=3, freq="4h", tz="UTC")})
    oi = pd.DataFrame({"time": pd.date_range(t0, periods=4, freq="4h", tz="UTC"), "oi": [1.0, 2.0, 3.0, 4.0]})
    out = attach_open_interest(bars, oi.iloc[:3], "4H")
    # 第 0 根（00:00-04:00）用 timestamp 04:00 那筆；最後一根收盤那筆還沒出來 → NaN
    assert out["oi"].iloc[0] == 2.0 and out["oi"].iloc[1] == 3.0 and np.isnan(out["oi"].iloc[2])


def test_oi_build_matches_research_rule_and_fetches_live_oi(monkeypatch):
    from grid_trading.signals import oi_build

    rng = np.random.default_rng(5)
    n = 400
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    oi = 1000 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    close[380:386] *= np.linspace(1.02, 1.15, 6)
    oi[380:386] *= np.linspace(1.03, 1.25, 6)
    df = _flow(close, rng.normal(0, 20, n), oi=oi)[["time", "open", "high", "low", "close", "volume", "oi"]]
    research = get("of_oi", n=6, k=2.0, mode="build").scan(df, "X_USDT_PERP", "4H")
    live = get("oi_build").scan(df, "X_USDT_PERP", "4H")
    assert [(s.time, s.side) for s in live] == [(s.time, s.side) for s in research]
    assert any(s.side == "LONG" and s.time > df["time"].iloc[380] for s in live)
    assert all(s.info["details"] for s in live)

    # 沒有 oi 欄位時（派網 K 線），向幣安要持倉量；timestamp = K 線收盤時間
    snap = pd.DataFrame({"time": df["time"] + pd.Timedelta(hours=4), "oi": oi})
    monkeypatch.setattr(oi_build, "fetch_open_interest", lambda coin, interval: snap)
    fetched = get("oi_build").scan(df.drop(columns="oi"), "X_USDT_PERP", "4H")
    assert [(s.time, s.side) for s in fetched] == [(s.time, s.side) for s in live]
    monkeypatch.setattr(oi_build, "fetch_open_interest", lambda coin, interval: snap.iloc[:0])
    assert get("oi_build").scan(df.drop(columns="oi"), "X_USDT_PERP", "4H") == []
