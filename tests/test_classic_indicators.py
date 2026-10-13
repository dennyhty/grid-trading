import numpy as np
import pandas as pd
import pytest

from grid_trading import classic_indicators as ci
from grid_trading.signals.strategy import LONG, SHORT, Signal
from grid_trading.signals.tracker import track

numba = pytest.importorskip("numba")  # 研究程式才用到 numba
import classic_indicator_study as cs  # noqa: E402


def random_candles(n=1500, seed=0):
    rng = np.random.default_rng(seed)
    ret = rng.normal(0, 0.01, n) + 0.002 * np.sin(np.arange(n) / 40)
    close = 100 * np.exp(np.cumsum(ret))
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.008, n))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.008, n))
    vol = rng.lognormal(10, 0.5, n)
    t = pd.date_range("2022-01-01", periods=n, freq="4h", tz="UTC")
    return pd.DataFrame({"time": t, "open": open_, "high": high, "low": low, "close": close, "volume": vol})


def test_bollinger_matches_definition():
    df = random_candles(300)
    bb = ci.bollinger(df["close"], 20, 2)
    w = df["close"].iloc[100:120].to_numpy()
    assert bb["mid"].iloc[119] == pytest.approx(w.mean())
    assert bb["upper"].iloc[119] == pytest.approx(w.mean() + 2 * w.std(ddof=0))


def test_kd_and_williams_ranges():
    df = random_candles(500)
    kd = ci.kd(df).dropna()
    assert kd["k"].between(0, 100).all() and kd["d"].between(0, 100).all()
    wr = ci.williams_r(df).dropna()
    assert wr.between(-100, 0).all()
    m = ci.mfi(df).dropna()
    assert m.between(0, 100).all()


def test_kd_first_value_uses_one_third_smoothing():
    df = random_candles(50)
    kd = ci.kd(df, 9)
    rsv = kd["rsv"].iloc[8]
    assert kd["k"].iloc[8] == pytest.approx(50 + (rsv - 50) / 3)


def test_pivots_need_k_bars_on_both_sides():
    v = np.array([1, 2, 3, 2, 1, 2, 5, 2, 1, 0.5, 3.0])
    hi = ci.pivots(v, 2, "high")
    assert np.flatnonzero(hi).tolist() == [2, 6]
    lo = ci.pivots(v, 2, "low")
    assert np.flatnonzero(lo).tolist() == [4]


def test_bearish_divergence_fires_at_confirmation_bar():
    # 兩個高點：第二個價格更高、指標更低 → 在第二個高點 + k 根收盤時做空
    price = np.array([1, 2, 5, 2, 1, 1, 1, 2, 6, 2, 1, 1], float)
    osc = np.array([1, 2, 9, 2, 1, 1, 1, 2, 7, 2, 1, 1], float)
    bull, bear = ci.divergence(price, price, osc, k=2, min_gap=3, max_gap=20)
    assert np.flatnonzero(bear).tolist() == [10]
    assert not bull.any()


def test_no_lookahead_in_any_strategy():
    """用截斷的資料算出來的訊號，和用完整資料算出來的同一段完全一樣（沒有偷看未來）。"""
    df = random_candles(1500, seed=3)
    full = cs.features(df, "4H")
    cut = 1100
    part = cs.features(df.iloc[:cut].reset_index(drop=True), "4H")
    for st in cs.STRATEGIES:
        if st["fn"] is None:
            continue
        a = st["fn"](full)
        b = st["fn"](part)
        for x, y in zip(a[:2], b[:2]):
            assert np.array_equal(np.asarray(x)[:cut], np.asarray(y)), st["name"]


@pytest.mark.parametrize("tf", ["4H", "2H"])
def test_simulator_matches_live_tracker(tf):
    """研究用的 numba 模擬和網站用的 tracker 逐筆一樣（實盤出場、單一 TP 兩種）。"""
    df = random_candles(1200, seed=7)
    x = cs.features(df, "4H")
    arr = {k: x[k] for k in ("o", "h", "l", "c", "atr")}
    rng = np.random.default_rng(1)
    idx = np.sort(rng.choice(np.arange(250, 1100), 40, replace=False))
    ds = rng.choice([1, -1], len(idx)).astype(np.int64)
    for ex in ("live", "tp1", "tp3"):
        r, hit, j, kind = cs.run_exit(arr, idx, ds, tf, ex, 0.0)
        for s, (i, d) in enumerate(zip(idx, ds)):
            price = arr["c"][i]
            atr = arr["atr"][i]
            risk = max(cs.LIVE_ATR * atr, cs.MIN_STOP_PCT * price)
            if ex == "live":
                tp_r, portions, trail_k = cs.LIVE_PLANS[tf]
                exit_ = {"portions": list(portions), "trail": trail_k * atr}
                max_bars = cs.MAX_BARS
            else:
                tp_r, exit_ = (float(ex[2:]),), {"portions": [1.0]}
                max_bars = cs.FIXED_MAX_BARS
            sig = Signal("X", tf, LONG if d > 0 else SHORT, df["time"].iloc[i + 1], price, price - d * risk,
                         [price + d * k * risk for k in tp_r], "test", {}, exit_)
            res = track(sig, df, fee=cs.FEE, max_bars=max_bars)
            if res.status != "CLOSED":
                assert kind[s] == -1
                continue
            assert r[s] == pytest.approx(res.realized_r, abs=1e-9), (ex, s)
            assert hit[s] == res.tps_hit


def test_natural_exit_closes_on_signal_bar():
    df = random_candles(400, seed=5)
    x = cs.features(df, "4H")
    arr = {k: x[k] for k in ("o", "h", "l", "c", "atr")}
    xl = np.zeros(len(df), bool)
    xl[260] = True
    idx = np.array([250])
    r, hit, j, kind = cs.run_exit(arr, idx, np.array([1]), "4H", "natural", 0.0, xl, np.zeros(len(df), bool), "trend")
    if kind[0] == 4:
        assert j[0] == 260
        entry, risk = arr["c"][250], max(2.0 * arr["atr"][250], 0.001 * arr["c"][250])
        exp = (arr["c"][260] - entry - cs.FEE * (entry + arr["c"][260])) / risk
        assert r[0] == pytest.approx(exp)
    else:
        assert kind[0] == 1 and j[0] < 260  # 先碰到止損


def test_daily_cap_takes_at_most_k_per_day_and_one_per_coin():
    t = lambda s: np.datetime64(s, "ns")  # noqa: E731
    t_in = np.array([t("2024-01-01T04"), t("2024-01-01T04"), t("2024-01-01T08"), t("2024-01-01T12"), t("2024-01-02T00")])
    t_out = np.array([t("2024-01-03T00"), t("2024-01-01T08"), t("2024-01-01T20"), t("2024-01-01T16"), t("2024-01-02T04")])
    coin = np.array([0, 1, 1, 2, 0])
    rank = np.array([5, 1, 1, 9, 5])  # 同一根先做市值排名前面的（幣 1）
    r = np.ones(5)
    take = cs._cap_take(t_in, t_out, rank, coin, r, 2)
    # 第一天：04:00 那根先做幣 1 再做幣 0，2 張名額用完，08:00、12:00 的訊號都跳過；第二天幣 0 還在持倉
    assert take.tolist() == [True, True, False, False, False]
    assert cs._cap_take(t_in, t_out, rank, coin, r, 10 ** 6).tolist() == [True, True, True, True, False]


def test_stationary_bootstrap_keeps_blocks():
    idx = cs.stationary_bootstrap_idx(500, 200, 20, seed=0)
    assert idx.shape == (200, 500) and idx.min() >= 0 and idx.max() < 500
    cont = (idx[:, 1:] == (idx[:, :-1] + 1) % 500).mean()
    assert 0.93 < cont < 0.97  # 平均區塊 20 天 → 約 95% 的天接著前一天
