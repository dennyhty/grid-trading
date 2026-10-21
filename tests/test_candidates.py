"""階段 2 候選策略（research/candidates_*.py）核心邏輯的測試：全部用合成資料，不連網。"""
import math

import numpy as np
import pandas as pd
import pytest

from research import candidates_common as cc
from research import candidates_daily_trend as fa
from research import candidates_squeeze as fc
from research import candidates_xsec as fb

numba = pytest.importorskip("numba")  # 出場模擬重用 research.stop_research.sim_many


def daily_frame(close, start="2021-01-01", spread=0.01):
    close = np.asarray(close, float)
    t = pd.date_range(start, periods=len(close), freq="D", tz="UTC")
    return pd.DataFrame({"time": t, "open": close, "high": close * (1 + spread), "low": close * (1 - spread),
                         "close": close, "volume": 1000.0, "qv": 1000.0 * close, "n": 6})


# ---------------------------------------------------------------- 日線重取樣

def test_to_daily_aggregates_and_drops_partial_last_day():
    t = pd.date_range("2024-01-01", periods=6 * 2 + 3, freq="4h", tz="UTC")   # 兩天整 + 第三天 3 根
    px = np.arange(len(t), dtype=float) + 100
    df = pd.DataFrame({"time": t, "open": px, "high": px + 1, "low": px - 1, "close": px + 0.5, "volume": 2.0})
    d = cc.to_daily(df)
    assert len(d) == 2
    assert d["time"].iloc[0] == pd.Timestamp("2024-01-01", tz="UTC")
    first = d.iloc[0]
    assert first["open"] == 100 and first["close"] == 105.5 and first["high"] == 106 and first["low"] == 99
    assert first["volume"] == 12 and first["n"] == 6
    assert first["qv"] == pytest.approx(sum(2.0 * (p + 0.5) for p in px[:6]))


# ---------------------------------------------------------------- 唐奇安訊號

def test_breakout_first_cross_only_and_no_lookahead():
    c = np.r_[np.full(30, 100.0), 101, 102, 103, 100, 99, np.full(10, 99.0), 90]
    h, l = c + 0.5, c - 0.5
    sig = fa.breakout_signals(h, l, c, 20)
    assert sig[30] == 1                      # 第一次收盤突破前 20 天最高（100.5）
    assert sig[31] == 0 and sig[32] == 0     # 連續突破只算第一次
    assert sig[-1] == -1                     # 跌破前 20 天最低
    assert (sig[:30] == 0).all()
    # 改未來的價格不會改變過去的訊號
    c2 = c.copy()
    c2[35:] = 200
    sig2 = fa.breakout_signals(c2 + 0.5, c2 - 0.5, c2, 20)
    assert (sig2[:35] == sig[:35]).all()


def test_daily_trend_grid_is_preregistered_size_and_neighbors_in_grid():
    cf = fa.configs()
    assert len(cf) == 12
    for name in cf:
        assert set(fa.neighbors(name)) <= set(cf) and name not in fa.neighbors(name)
    assert fa.neighbors("N55|sl2|chan") == ["N20|sl2|chan", "N100|sl2|chan", "N55|sl3|chan"]
    assert len(fc.configs()) == 8 and len(fb.configs()) == 10
    for mod in (fc, fb):
        for name in mod.configs():
            assert set(mod.neighbors(name)) <= set(mod.configs())


# ---------------------------------------------------------------- 出場模擬

def test_channel_exit_and_stop():
    # 一路漲 40 天後回落：第 30 天進場（1R = 2 ATR），之後收盤跌破前 10 天最低時出場
    c = np.r_[np.linspace(100, 140, 41), np.linspace(139, 120, 20)]
    d = daily_frame(c, spread=0.002)
    arr = fa.arrays(d)
    plan = cc.ExitPlan(stop_atr=2.0, chan_exit=10)
    s = cc.simulate(arr, np.array([30]), np.array([1]), plan, fee=0.0)
    j = int(s["j"][0])
    lo10 = pd.Series(arr["low"]).rolling(10).min().shift(1).to_numpy()
    assert s["kind"][0] == 4
    assert arr["close"][j] < lo10[j] and (arr["close"][31:j] >= lo10[31:j]).all()
    risk = 2.0 * arr["atr"][30]
    assert s["r"][0] == pytest.approx((arr["close"][j] - arr["close"][30]) / risk)
    # 跳空跌破止損：用開盤價成交（gapfill），R 比 -1 更差
    c3 = np.r_[np.full(40, 100.0), 80.0, 80.0]
    d3 = daily_frame(c3, spread=0.01)
    arr3 = fa.arrays(d3)
    s3 = cc.simulate(arr3, np.array([39]), np.array([1]), cc.ExitPlan(stop_atr=2.0), fee=0.0)
    assert s3["kind"][0] == 1 and s3["r"][0] < -1


def test_trade_frame_times():
    d = daily_frame(np.linspace(100, 110, 20))
    sim = {"r": np.array([0.5, -1.0]), "hit": np.array([0, 0]), "j": np.array([10, 12]), "kind": np.array([4, 1]),
           "risk_pct": np.array([0.02, 0.02])}
    tr = cc.trade_frame("XUSDT", d, np.array([5, 11]), np.array([1, -1]), sim, cc.adv30(d["qv"]))
    assert tr["time"].iloc[0] == pd.Timestamp("2021-01-07", tz="UTC")          # 訊號日 01-06 收盤
    assert tr["exit_t"].iloc[0] == pd.Timestamp("2021-01-11 23:59:59", tz="UTC")  # 收盤出場
    assert tr["exit_t"].iloc[1] == pd.Timestamp("2021-01-13 12:00", tz="UTC")     # 盤中止損


def test_one_at_a_time_blocks_overlaps_but_allows_exit_bar():
    tr = pd.DataFrame({"symbol": ["A"] * 4 + ["B"], "i": [1, 3, 5, 9, 2], "j": [5, 6, 9, 12, 4],
                       "kind": [1, 1, 4, -1, 1]})
    out = cc.one_at_a_time(tr)
    assert list(zip(out["symbol"], out["i"])) == [("A", 1), ("A", 5), ("B", 2)]


def test_pick_headline_uses_in_sample_only():
    t = pd.DataFrame({"config": ["a", "b", "c"], "is_trades": [500, 500, 50], "is_t": [1.0, 2.0, 9.0],
                      "oos_t": [9.0, -1.0, 9.0]})
    assert cc.pick_headline(t, min_trades=100) == "b"


# ---------------------------------------------------------------- 壓縮突破

def test_trailing_pct_rank_matches_bruteforce_and_excludes_today():
    rng = np.random.default_rng(0)
    x = rng.normal(size=60)
    x[7] = np.nan
    got = fc.trailing_pct_rank(x, 20)
    for i in range(20, 60):
        past = x[i - 20:i]
        past = past[np.isfinite(past)]
        want = (past < x[i]).mean() if np.isfinite(x[i]) else np.nan
        assert (np.isnan(got[i]) and np.isnan(want)) or got[i] == pytest.approx(want)
    assert np.isnan(got[:20]).all()


def test_squeeze_signal_needs_prior_compression():
    rng = np.random.default_rng(1)
    wild = 100 * np.exp(np.cumsum(rng.normal(0, 0.05, 150)))      # 波動大
    calm = wild[-1] * (1 + rng.normal(0, 0.001, 30))               # 很窄的盤整
    c = np.r_[wild, calm, calm[-1] * 1.10]                          # 最後一天大漲突破上軌
    sig = fc.squeeze_signals(c, q=0.10)
    assert sig[-1] == 1
    # 沒有先壓縮（一直都大波動）就沒有訊號
    c_no = np.r_[wild, wild[-1] * 1.5]
    assert fc.squeeze_signals(c_no, q=0.10)[-1] == 0


# ---------------------------------------------------------------- 橫斷面組合

def test_select_and_target_notional_is_beta_neutral_and_vol_scaled():
    score = np.array([5, 4, 3, 2, 1, 0, -1, -2, -3, -4, -5, np.nan])
    elig = np.ones(12, bool)
    longs, shorts = fb.select(score, elig, 0.2)
    assert list(longs) == [0, 1, 2] and list(shorts) == [8, 9, 10]   # 11 個合格，每邊 max(3, 2) = 3
    vol = np.linspace(0.02, 0.08, 12)
    beta = np.linspace(0.8, 1.6, 12)
    w = fb.target_notional(longs, shorts, vol, beta, hedge_idx=11, n=12)
    assert w[0] == pytest.approx(cc.RISK / (vol[0] * math.sqrt(7)))
    assert w[8] == pytest.approx(-cc.RISK / (vol[8] * math.sqrt(7)))
    assert (w * beta).sum() == pytest.approx(0.0, abs=1e-12)
    # 合格不足就不做
    l2, s2 = fb.select(score, np.r_[np.ones(5, bool), np.zeros(7, bool)], 0.2)
    assert len(l2) == 0 and len(s2) == 0


def test_funding_daily_buckets():
    days = pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC")
    f = pd.DataFrame({"time": pd.to_datetime(["2024-01-01 08:00:00.000", "2024-01-02 00:00:00.003",
                                              "2024-01-02 08:00:00.000"], utc=True), "rate": [0.001, 0.002, 0.004]})
    got = fb.funding_daily(f, days)
    assert got == pytest.approx([0.003, 0.004, 0.0])   # 01-02 00:00 屬於 01-01 那根 K


def synthetic_panel(n_days=200, n_syms=12, funding_rate=0.0):
    """BTC 加 11 個幣，各自固定的日漲幅（不同斜率），沒有雜訊以外的東西；每天都在幣池。"""
    rng = np.random.default_rng(3)
    days = pd.date_range("2022-01-02", periods=n_days, freq="D", tz="UTC")
    syms = ["BTCUSDT"] + [f"C{k}USDT" for k in range(1, n_syms)]
    data = {}
    for k, s in enumerate(syms):
        drift = 0.0 if k == 0 else (k - n_syms / 2) * 0.002
        r = drift + rng.normal(0, 0.01 + 0.002 * k, n_days)
        c = 100 * np.exp(np.cumsum(r))
        data[s] = pd.DataFrame({"time": days, "open": c, "high": c, "low": c, "close": c, "volume": 1e7,
                                "qv": 1e7 * c, "n": 6})
    funding = {s: pd.DataFrame({"time": days + pd.Timedelta(hours=8), "rate": funding_rate}) for s in syms}
    panel = fb.build_panel(data, funding=funding, cache=False)
    pool = {(s, m) for s in syms for m in (days + pd.Timedelta(days=1)).strftime("%Y-%m").unique()}
    return panel, pool


def test_xsec_run_accounting_matches_positions():
    panel, pool = synthetic_panel()
    rows = fb.run(panel, pool, {"signal": "mom28", "q": 0.2}, start="2022-01-01")
    assert len(rows) and set(rows.columns) >= {"symbol", "side", "time", "exit_t", "r", "r_net", "hedge"}
    # 帳務：r_net = 價格損益 + 資金費率 + 成本（都以 0.25% 為 1）
    assert np.allclose(rows["r_net"], rows["r"] + rows["funding_r"] + rows["cost_r"])
    assert (rows["cost_r"] <= 0).all() and rows["cost_r"].sum() < 0
    assert (rows["funding_r"] == 0).all()
    # 每一列的出場時間是同一天，日報酬加總不跨日
    assert (rows["exit_t"].dt.floor("D") == rows["time"]).all()
    # 動能：斜率最大的幾個（C9~C11）做多、最小的做空
    pw = fb.position_weeks(rows)
    top = pw.groupby("symbol")["side"].mean()
    assert top["C11USDT"] > 0 and top["C1USDT"] < 0
    # 第一個再平衡後的第一天：每個部位的損益 = 名目 x 當天報酬
    first_day = rows["time"].min()
    sub = rows[rows["time"] == first_day]
    a = int(np.flatnonzero(panel.days == first_day)[0]) - 1
    legs = sub[~sub["hedge"]]
    assert len(legs) >= 6
    for _, row in legs.iterrows():
        k = panel.syms.index(row["symbol"])
        ret = panel.close[a + 1, k] / panel.close[a, k] - 1
        notional = cc.RISK / (panel.vol[a, k] * math.sqrt(7))      # 一週一個標準差 = 0.25%
        assert row["r"] * cc.RISK == pytest.approx(row["side"] * notional * ret)


def test_xsec_long_pays_positive_funding():
    panel, pool = synthetic_panel(funding_rate=0.001)
    rows = fb.run(panel, pool, {"signal": "lowvol", "q": 0.2}, start="2022-01-01")
    legs = rows[(rows["funding_r"] != 0)]
    assert (np.sign(legs["funding_r"]) == -np.sign(legs["side"])).all()


def test_xsec_random_selection_same_counts():
    score = np.arange(20, dtype=float)
    elig = np.ones(20, bool)
    l1, s1 = fb.select(score, elig, 0.3)
    l2, s2 = fb.select(score, elig, 0.3, rng=np.random.default_rng(0))
    assert len(l1) == len(l2) == 6 and len(s1) == len(s2) == 6
    assert not set(l2) & set(s2)


def test_compare_with_reports_correlation_and_combined_sharpe():
    idx = pd.date_range("2022-01-01", periods=400, freq="D", tz="UTC")
    rng = np.random.default_rng(5)
    a = pd.Series(rng.normal(0.001, 0.01, 400), index=idx)
    b = pd.Series(rng.normal(0.001, 0.01, 400), index=idx)
    out = cc.compare_with(a, b)
    assert abs(out["corr_daily"]) < 0.15
    assert out["sharpe_combined"] > min(out["sharpe_base"], out["sharpe_cand"])
    same = cc.compare_with(a, a)
    assert same["corr_daily"] == pytest.approx(1.0)
