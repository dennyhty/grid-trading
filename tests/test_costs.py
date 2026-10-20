"""資金費率與滑價成本的測試（不連網）。"""
import numpy as np
import pandas as pd
import pytest

from grid_trading import costs
from grid_trading.costs import apply_costs, funding_r, funding_r_many, slippage_bps, slippage_r


def make_funding(rate=0.0001, start="2024-01-01", n=30):
    t = pd.date_range(start, periods=n, freq="8h", tz="UTC")
    return pd.DataFrame({"time": t, "rate": rate})


def test_long_pays_short_receives():
    f = make_funding(0.0001)
    # 01-01 00:00 到 01-02 00:00，(entry, exit] 內有 08:00、16:00、24:00 三次
    long_r = funding_r(f, "2024-01-01 00:00", "2024-01-02 00:00", +1, 0.03)
    short_r = funding_r(f, "2024-01-01 00:00", "2024-01-02 00:00", -1, 0.03)
    assert long_r == pytest.approx(-3 * 0.0001 / 0.03)
    assert short_r == pytest.approx(+3 * 0.0001 / 0.03)


def test_negative_rate_flips_sign():
    f = make_funding(-0.0002)
    assert funding_r(f, "2024-01-01 00:00", "2024-01-02 00:00", +1, 0.02) > 0


def test_window_boundaries():
    f = make_funding(0.0001)
    # 進場剛好在結算時刻：該次不算；出場剛好在結算時刻：該次算
    one = funding_r(f, "2024-01-01 08:00", "2024-01-01 16:00", +1, 0.01)
    assert one == pytest.approx(-0.0001 / 0.01)
    # 兩個結算點之間：沒有費用
    assert funding_r(f, "2024-01-01 08:00:01", "2024-01-01 15:59:59", +1, 0.01) == 0.0
    # 區間在資料之外
    assert funding_r(f, "2030-01-01", "2030-01-05", +1, 0.01) == 0.0
    assert funding_r(f, "2019-01-01", "2019-01-05", +1, 0.01) == 0.0


def test_many_matches_single_and_held_fraction():
    rng = np.random.default_rng(0)
    f = pd.DataFrame({"time": pd.date_range("2024-01-01", periods=100, freq="8h", tz="UTC"),
                      "rate": rng.normal(0.0001, 0.0002, 100)})
    entries = pd.to_datetime(["2024-01-02 03:00", "2024-01-05 00:00", "2024-01-10 12:00"], utc=True)
    exits = entries + pd.to_timedelta([1, 3, 5], unit="D")
    sides = np.array([1, -1, 1])
    risk = np.array([0.03, 0.05, 0.02])
    many = funding_r_many(f, entries, exits, sides, risk)
    for i in range(3):
        assert many[i] == pytest.approx(funding_r(f, entries[i], exits[i], sides[i], risk[i]))
    half = funding_r_many(f, entries, exits, sides, risk, held_fraction=0.5)
    assert np.allclose(half, many * 0.5)
    per = funding_r_many(f, entries, exits, sides, risk, held_fraction=np.array([1.0, 0.5, 0.25]))
    assert np.allclose(per, many * np.array([1.0, 0.5, 0.25]))


def test_empty_funding_is_zero():
    empty = pd.DataFrame({"time": pd.to_datetime([], utc=True), "rate": []})
    assert funding_r(empty, "2024-01-01", "2024-01-03", +1, 0.03) == 0.0


def test_slippage_tiers():
    assert slippage_bps(2e9) == 1.0
    assert slippage_bps(1e9) == 1.0
    assert slippage_bps(5e8) == 2.0
    assert slippage_bps(6e7) == 4.0
    assert slippage_bps(1e7) == 8.0
    assert slippage_bps(9e6) == 15.0
    assert slippage_bps(float("nan")) == 15.0
    assert slippage_r(4, 0.02) == pytest.approx(-0.0004 * 2 / 0.02)
    assert np.allclose(slippage_r(np.array([1.0, 8.0]), np.array([0.02, 0.04]), sides=1), [-0.0001 / 0.02, -0.0008 / 0.04])


def test_apply_costs_columns():
    f = make_funding(0.0001)
    trades = pd.DataFrame({
        "symbol": ["AAAUSDT", "AAAUSDT", "NOFUNDUSDT"],
        "side": [1, -1, 1],
        "time": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-01"], utc=True),
        "exit_t": pd.to_datetime(["2024-01-02", "2024-01-02", "2024-01-02"], utc=True),
        "r": [1.0, 1.0, 1.0],
        "risk_pct": [0.03, 0.03, 0.03],
        "adv_usdt": [2e9, 2e9, 5e6],
    })
    out = apply_costs(trades, funding={"AAAUSDT": f}, fetch=False)
    assert {"funding_r", "slip_r", "r_net", "funding_missing"} <= set(out.columns)
    assert "funding_r" not in trades.columns  # 不改原表
    assert out.loc[0, "funding_r"] < 0 < out.loc[1, "funding_r"]
    assert out["funding_missing"].tolist() == [False, False, True]
    assert out.loc[2, "funding_r"] == 0.0
    assert out.loc[0, "slip_r"] == pytest.approx(-(1 / 1e4) * 2 / 0.03)
    assert out.loc[2, "slip_r"] == pytest.approx(-(15 / 1e4) * 2 / 0.03)
    assert np.allclose(out["r_net"], out["r"] + out["funding_r"] + out["slip_r"])


def test_apply_costs_without_adv_uses_worst_tier():
    trades = pd.DataFrame({"symbol": ["X"], "side": [1], "time": pd.to_datetime(["2024-01-01"], utc=True),
                           "exit_t": pd.to_datetime(["2024-01-02"], utc=True), "r": [0.5], "risk_pct": [0.03]})
    out = apply_costs(trades, funding={}, fetch=False)
    assert out.loc[0, "slip_bps"] == costs.SLIP_DEFAULT_BPS
