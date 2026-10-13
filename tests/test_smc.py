import numpy as np
import pandas as pd

from grid_trading import indicators as ind
from grid_trading import smc


def _bars(closes, wick=0.5) -> pd.DataFrame:
    c = np.asarray(closes, float)
    o = np.r_[c[0], c[:-1]]
    time = pd.date_range("2026-01-01", periods=len(c), freq="4h", tz="UTC")
    return pd.DataFrame({"time": time, "open": o, "high": np.maximum(o, c) + wick,
                         "low": np.minimum(o, c) - wick, "close": c, "volume": 1.0})


def _random_walk(n=1500) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    return _bars(100 * np.exp(np.cumsum(0.01 * rng.standard_normal(n))), wick=0.3)


def test_structure_does_not_peek_ahead():
    df = _random_walk()
    trend_full, breaks_full = smc.structure(df, 3)
    trend_cut, breaks_cut = smc.structure(df.iloc[:1000], 3)
    assert (trend_full[:1000] == trend_cut).all()
    assert [(b.bar, b.kind, b.pivot, b.sweep) for b in breaks_full if b.bar < 1000] == \
           [(b.bar, b.kind, b.pivot, b.sweep) for b in breaks_cut]


def test_sweep_then_break_up():
    # 下跌做出擺動低點 → 反彈做出擺動高點 → 影線跌破低點但收回（掃）→ 收盤突破高點
    closes = [110, 108, 106, 104, 102, 100, 102, 104, 106, 105, 104, 103, 102, 101.5, 102, 103, 105, 107, 108]
    df = _bars(closes)
    df.loc[13, "low"] = 98.0   # 掃掉 100 那個低點，收在 101.5
    trend, breaks = smc.structure(df, 2)
    up = [b for b in breaks if b.direction == 1]
    assert up, "應該要有向上突破"
    b = up[0]
    assert df["close"].iat[b.bar] > df["high"].iat[b.pivot]
    assert b.sweep == 13
    df["atr"] = ind.atr(df)
    setups = smc.find_setups(df, breaks, None, setup="sweep", entry="market", htf=False, discount=False,
                             min_risk_atr=0.0)
    s = [x for x in setups if x.bar == b.bar][0]
    assert s.sl < 98.0 < s.entry


def test_setups_have_stop_on_the_right_side():
    df = _random_walk()
    df["atr"] = ind.atr(df)
    _, breaks = smc.structure(df, 3)
    for entry in ("market", "fvg", "fvg_ce", "ob"):
        for s in smc.find_setups(df, breaks, None, setup="choch", entry=entry, htf=False, discount=False):
            assert s.direction * (s.entry - s.sl) > 0


def test_smc_sweep_strategy_signals_are_stable_and_ordered():
    from grid_trading import signals

    df = _random_walk(1500)
    strat = signals.get("smc_sweep")
    full = strat.scan(df, "X_USDT_PERP", "4H")
    cut = strat.scan(df.iloc[:1200], "X_USDT_PERP", "4H")
    assert cut, "隨機漫步上應該會有幾個訊號"
    # 只看前 1200 根時找到的訊號，在完整資料上也要一模一樣（不偷看未來）
    early = [(s.time, s.side, round(s.sl, 9)) for s in full if s.time <= cut[-1].time]
    assert early == [(s.time, s.side, round(s.sl, 9)) for s in cut]
    for s in full:
        assert s.direction * (s.price - s.sl) > 0
        assert s.direction * (s.tps[0] - s.price) > 0
    assert strat.intervals == ("4H",) and strat.markets == ("crypto",) and strat.tag
