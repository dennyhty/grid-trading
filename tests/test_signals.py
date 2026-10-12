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
