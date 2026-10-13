import numpy as np
import pandas as pd
import pytest

from grid_trading import indicators as ind
from grid_trading.signals import LONG, SHORT, Signal, get
from grid_trading.signals.tracker import track

T0 = pd.Timestamp("2026-01-01", tz="UTC")


def candles(rows):
    """rows: (high, low, close)；time 是每小時的開盤時間，從 T0 開始。"""
    df = pd.DataFrame(rows, columns=["high", "low", "close"])
    df["open"] = df["close"]
    df["time"] = pd.date_range(T0, periods=len(df), freq="h", tz="UTC")
    return df


def sig(side=LONG, price=100.0, sl=90.0, tps=(110.0, 120.0, 130.0)):
    return Signal("X_USDT_PERP", "60M", side, T0, price, sl, list(tps), "test")


def test_stop_loss_first_costs_one_r_plus_fees():
    r = track(sig(), candles([(101, 89, 92)]), fee=0)
    assert r.status == "CLOSED" and r.events[-1].kind == "SL"
    assert r.realized_r == pytest.approx(-1.0)


def test_tp1_moves_stop_to_breakeven_then_be_exit():
    r = track(sig(), candles([(111, 99, 108), (109, 95, 96)]), fee=0)
    kinds = [e.kind for e in r.events]
    assert kinds == ["TP1", "STOP_MOVED", "BE_EXIT"]
    assert r.realized_r == pytest.approx(1 / 3)


def test_all_tps_hit_gives_two_r():
    r = track(sig(), candles([(111, 99, 110), (121, 109, 120), (131, 119, 130)]), fee=0)
    assert [e.kind for e in r.events if e.kind.startswith("TP")] == ["TP1", "TP2", "TP3"]
    assert r.status == "CLOSED" and r.remaining == pytest.approx(0, abs=1e-12)
    assert r.realized_r == pytest.approx((1 + 2 + 3) / 3)


def test_after_tp2_stop_is_tp1():
    r = track(sig(), candles([(121, 99, 120), (119, 105, 106)]), fee=0)
    assert r.events[-1].kind == "PROTECT_EXIT" and r.events[-1].price == 110
    assert r.realized_r == pytest.approx((1 + 2 + 1) / 3)


def test_short_mirrors_long():
    s = sig(SHORT, 100, 110, (90, 80, 70))
    r = track(s, candles([(101, 89, 92)]), fee=0)
    assert [e.kind for e in r.events] == ["TP1", "STOP_MOVED"] and r.status == "OPEN"
    assert r.stop == pytest.approx(100)


def test_fees_reduce_result_and_raise_breakeven():
    r = track(sig(), candles([(101, 89, 92)]), fee=0.0005)
    assert r.realized_r < -1.0
    assert r.be == pytest.approx(100.1)


def test_candles_before_signal_are_ignored():
    df = candles([(200, 1, 100), (101, 99, 100)])
    s = sig()
    s.time = df["time"].iloc[1]
    r = track(s, df, fee=0)
    assert r.status == "OPEN" and r.bars == 1


def test_rsi_bounds_and_macd_columns():
    close = pd.Series(np.linspace(1, 2, 100))
    assert ind.rsi(close).iloc[-1] == pytest.approx(100)
    m = ind.macd(close)
    assert list(m.columns) == ["macd", "signal", "hist"]
    assert (m["hist"] == m["macd"] - m["signal"]).all()


def test_vegas_trend_only_uses_past_data():
    rng = np.random.default_rng(0)
    close = 100 * np.exp(np.cumsum(rng.normal(0.001, 0.01, 800)))
    df = candles([(c * 1.005, c * 0.995, c) for c in close])
    strat = get("vegas_trend")
    full = strat.scan(df, "X", "60M")
    cut = strat.scan(df.iloc[:600], "X", "60M")
    assert full, "這段上漲走勢應該至少有一個訊號"
    # 只看前 600 根得到的訊號，和看完整資料時前 600 根內的訊號必須一模一樣
    assert [s.time for s in cut] == [s.time for s in full if s.time <= df["time"].iloc[599] + pd.Timedelta("1h")]
    for s in full:
        assert (s.sl < s.price < s.tps[0]) if s.side == LONG else (s.sl > s.price > s.tps[0])


def _trend_hours(n=1200, drift=0.0015, seed=1, freq="h"):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(drift + 0.004 * rng.standard_normal(n)))
    df = pd.DataFrame({"close": close, "open": close, "high": close * 1.003, "low": close * 0.997, "volume": 1.0})
    df["time"] = pd.date_range(T0, periods=n, freq=freq, tz="UTC")
    return df


def test_vegas_breakout_only_trades_with_the_4h_trend():
    up = get("vegas_breakout", min_risk=0.0).scan(_trend_hours(drift=0.0015), "X_USDT_PERP", "60M")
    down = get("vegas_breakout", min_risk=0.0).scan(_trend_hours(drift=-0.0015), "X_USDT_PERP", "60M")
    assert up and all(s.side == LONG for s in up)
    assert down and all(s.side == SHORT for s in down)
    for s in up:
        assert s.sl < s.price < s.tps[0] < s.tps[1] < s.tps[2]


def test_vegas_breakout_does_not_peek_ahead():
    df = _trend_hours()
    strat = get("vegas_breakout", min_risk=0.0)
    full = [s.time for s in strat.scan(df, "X", "60M")]
    cut = [s.time for s in strat.scan(df.iloc[:900], "X", "60M")]
    assert [t for t in full if t <= df["time"].iloc[899]] == cut


def test_vegas_breakout_on_daily_bars_waits_for_trend_warmup_and_does_not_peek():
    df = _trend_hours(n=1000, drift=0.0015, freq="D")
    strat = get("vegas_breakout", min_risk=0.0, max_risk=0.5)
    full = strat.scan(df, "X", "1D")
    assert full and all(s.side == LONG for s in full)
    # 大週期是 4 日線，要先有 169 根（676 天）才判斷方向
    assert min(s.time for s in full) > df["time"].iloc[0] + pd.Timedelta(days=676)
    cut = [s.time for s in strat.scan(df.iloc[:850], "X", "1D")]
    assert [s.time for s in full if s.time <= df["time"].iloc[849] + pd.Timedelta(days=1)] == cut


def test_psar_flips_below_price_in_an_uptrend():
    close = pd.Series(100 * 1.002 ** np.arange(300))
    df = pd.DataFrame({"close": close, "high": close * 1.001, "low": close * 0.999})
    sar = ind.psar(df)
    assert (sar.iloc[-50:] < df["close"].iloc[-50:]).all()


def test_hma_lags_far_less_than_an_ema():
    s = pd.Series(np.arange(400, dtype=float))
    assert 399 - ind.hma(s, 55).iloc[-1] < 2 < 399 - ind.ema(s, 55).iloc[-1]


def test_candidate_prefilter_gives_the_same_signals_as_checking_every_bar():
    # 趨勢一下上一下下的隨機走勢，兩個策略都會有多有空
    rng = np.random.default_rng(3)
    n = 3000
    close = 100 * np.exp(np.cumsum(0.0005 * np.sin(np.arange(n) / 200) + 0.01 * rng.standard_normal(n)))
    df = pd.DataFrame({"close": close, "open": np.r_[close[0], close[:-1]], "high": close * 1.004,
                       "low": close * 0.996, "volume": 1.0})
    df["time"] = pd.date_range(T0, periods=n, freq="h", tz="UTC")
    for strat in (get("vegas_breakout", min_risk=0.0), get("vegas_trend")):
        fast = strat.scan(df, "X", "60M")
        strat.candidates = lambda d: None  # 退回逐根檢查
        slow = strat.scan(df, "X", "60M")
        assert fast and [(s.time, s.side, s.sl, s.tps) for s in fast] == [(s.time, s.side, s.sl, s.tps) for s in slow]


def test_binance_clean_drops_delisted_flat_bars_and_keeps_the_last_listing():
    from grid_trading.binance_history import clean
    day = 86_400_000
    old = pd.DataFrame({"time": [i * day for i in range(5)], "volume": [1, 1, 1, 0, 0]})
    new = pd.DataFrame({"time": [(100 + i) * day for i in range(3)], "volume": [1, 1, 1]})
    out = clean(pd.concat([old, new]).assign(open=1.0, high=1.0, low=1.0, close=1.0))
    assert out["time"].tolist() == [100 * day, 101 * day, 102 * day]


def test_half_at_tp1_then_trailing_stop():
    # 1R 平一半，之後追蹤距離 15：最高 140 → 追蹤止損 125，回落到 124 出場
    s = Signal("X_USDT_PERP", "4H", LONG, T0, 100.0, 90.0, [110.0], "test", exit={"portions": [0.5], "trail": 15.0})
    r = track(s, candles([(111, 99, 110), (140, 120, 135), (138, 124, 126)]), fee=0)
    kinds = [e.kind for e in r.events]
    assert kinds == ["TP1", "STOP_MOVED", "TRAIL_MOVED", "TRAIL_EXIT"]
    assert r.events[2].price == pytest.approx(125.0)
    assert r.realized_r == pytest.approx(0.5 * 1 + 0.5 * 2.5)


def test_trailing_never_below_breakeven():
    s = Signal("X_USDT_PERP", "4H", LONG, T0, 100.0, 90.0, [110.0], "test", exit={"portions": [0.5], "trail": 15.0})
    r = track(s, candles([(111, 101, 110), (105, 99, 100)]), fee=0)
    assert [e.kind for e in r.events] == ["TP1", "STOP_MOVED", "BE_EXIT"]
    assert r.realized_r == pytest.approx(0.5)


def test_vegas_exit_plans_by_interval():
    v = get("vegas_trend")
    assert v.exit_plan("4H")["trail_atr"] == 3.0
    assert v.exit_plan("2H")["tp_r"] == [1.0, 3.0, 5.0]
    assert v.exit_plan("2H")["trail_atr"] == 3.0 and sum(v.exit_plan("2H")["portions"]) == 0.75
    assert v.exit_plan("60M") is None
