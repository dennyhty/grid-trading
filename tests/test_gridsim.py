import numpy as np
import pandas as pd
import pytest

from grid_trading.gridsim import LONG, NEUTRAL, SHORT, simulate


def bars(prices):
    """每根 K 線只從上一個價走到下一個價（開=前收）。"""
    p = list(prices)
    rows = []
    for a, b in zip(p[:-1], p[1:]):
        rows.append({"open": a, "high": max(a, b), "low": min(a, b), "close": b})
    df = pd.DataFrame(rows)
    df["time"] = pd.date_range("2026-01-01", periods=len(df), freq="h", tz="UTC")
    return df


def test_neutral_grid_profits_from_oscillation_without_fees():
    # 區間 90~110 分 4 格（格距 5），價格在 100 和 90 之間來回 3 次
    r = simulate(bars([100, 90, 100, 90, 100, 90, 100]), 90, 110, 4, NEUTRAL, fee=0)
    # 下方兩格各完成 3 次低買高賣，每次獲利 5/價格 × 1/4 資金
    expected = 3 * (0.25 / 90 * 5 + 0.25 / 95 * 5)
    assert r.ret == pytest.approx(expected)
    assert r.trades == 6 and not r.stopped


def test_fees_make_idle_grid_lose_nothing_but_open_cost():
    r = simulate(bars([100, 100.1, 100]), 90, 110, 4, NEUTRAL, fee=0.0005)
    assert r.ret == 0 and r.trades == 0


def test_long_grid_stopped_on_crash():
    r = simulate(bars([100, 95, 90, 80, 70]), 90, 110, 4, LONG, stop_low=85, fee=0)
    assert r.stopped and r.ret < 0
    # 停損後就不再交易
    assert r.trades == 0


def test_short_grid_mirror_of_long():
    up = simulate(bars([100, 105, 100, 105, 100]), 90, 110, 4, LONG, fee=0)
    down = simulate(bars([100, 95, 100, 95, 100]), 90, 110, 4, SHORT, fee=0)
    assert up.trades == down.trades == 2
    assert up.ret > 0 and down.ret > 0


def test_long_grid_initial_position_gains_on_rally():
    r = simulate(bars([100, 110]), 90, 110, 4, LONG, fee=0)
    # 現價上方兩格開單時買進，漲到 110 全部賣出
    assert r.trades == 2 and r.ret > 0
