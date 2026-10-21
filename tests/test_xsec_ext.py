"""候選 B 延伸版（現貨補長 + 三種對沖）的測試：grid_trading/spot_pit_universe.py 和 research/candidates_xsec_ext.py。
全部用合成資料，不連網。"""
import numpy as np
import pandas as pd
import pytest

from grid_trading import spot_pit_universe as spu
from research import candidates_xsec as fb
from research import candidates_xsec_ext as fx


# ---------------------------------------------------------------- 現貨幣池

def test_leveraged_and_stable_symbols_are_excluded():
    bases = {"BTC", "ETH", "BNB", "ADA", "JUP"}
    for s in ("BULLUSDT", "BEARUSDT", "ETHBULLUSDT", "EOSBEARUSDT", "BTCDOWNUSDT", "BTCUPUSDT",
              "USDCUSDT", "TUSDUSDT", "PAXUSDT", "BUSDUSDT", "EURUSDT", "USDSBUSDT"):
        assert not spu.keep_spot_symbol(s, bases), s
    for s in ("BTCUSDT", "XRPUSDT", "JUPUSDT", "SUPERUSDT", "BCHABCUSDT", "VENUSDT"):
        assert spu.keep_spot_symbol(s, bases), s
    assert not spu.keep_spot_symbol("ETHBTC", bases)          # 只要 USDT 計價
    assert spu.is_leveraged("ADAUP") and not spu.is_leveraged("ADAUP", {"BTC"})


def test_parse_klines_handles_header_and_microseconds():
    raw = ("open_time,open,high,low,close,volume,close_time,quote_volume,count,tb,tq,ignore\n"
           "1514764800000,1,2,0.5,1.5,10,1514851199999,15,1,1,1,0\n"
           "1735689600000000,3,4,2,3.5,20,1735775999999999,70,1,1,1,0\n")
    d = spu.parse_klines(raw, "XUSDT")
    assert list(d.columns) == spu.DAILY_COLUMNS and len(d) == 2
    assert d["time"].tolist() == [1514764800000, 1735689600000]
    assert d["quote_volume"].tolist() == [15.0, 70.0] and (d["symbol"] == "XUSDT").all()


def _spot_long(syms_vol: dict, start="2019-01-01", end="2019-04-30", dead: dict | None = None) -> pd.DataFrame:
    """合成現貨長表：每個代號每天成交額固定；dead[代號] = 從哪天起成交量為 0（下架）。"""
    days = pd.date_range(start, end, freq="D", tz="UTC")
    rows = []
    for s, qv in syms_vol.items():
        for t in days:
            v = 0.0 if dead and s in dead and t >= pd.Timestamp(dead[s], tz="UTC") else 1.0
            rows.append({"symbol": s, "time": t, "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0,
                         "volume": v, "quote_volume": qv * v})
    return pd.DataFrame(rows)


def test_spot_membership_ranks_prior_month_and_respects_min_days():
    d = _spot_long({"AUSDT": 300, "CUSDT": 200, "DUSDT": 100}, dead={"AUSDT": "2019-02-10"})
    table = spu.monthly_table(d)
    feb = table[(table["symbol"] == "AUSDT") & (table["month"] == "2019-02")].iloc[0]
    assert feb["days"] == 9 and feb["quote_volume"] == pytest.approx(2700)
    mem = spu.spot_membership(table, n=2, min_days=20, first="2019-01", last="2019-12")
    by = mem.groupby("month")["symbol"].apply(list).to_dict()
    assert by["2019-02"] == ["AUSDT", "CUSDT"]              # 一月成交額前 2 名（A 還活著）
    assert by["2019-03"] == ["CUSDT", "DUSDT"]              # 二月 A 只有 9 天有量 → 不算
    assert "2019-05" in by and "2019-01" not in by           # 第一個資料月的下個月才有池子
    mem_all = spu.spot_membership(table, n=100, min_days=20, first="2019-01", last="2019-12")
    assert len(mem_all[mem_all["month"] == "2019-02"]) == 3  # 不足 100 個就全取
    clipped = spu.spot_membership(table, n=2, first="2019-01", last="2019-03")
    assert clipped["month"].max() == "2019-03"


# ---------------------------------------------------------------- 指數與部位

def test_pool_mask_and_equal_weight_index():
    days = pd.date_range("2019-12-30", periods=4, freq="D", tz="UTC")
    close = pd.DataFrame({"A": [1.0, 1.1, 1.21, 1.21], "B": [1.0, 1.0, 0.9, 0.99], "C": [1.0, 2.0, 4.0, 8.0]},
                         index=days)
    pool = {("A", "2019-12"), ("B", "2019-12"), ("A", "2020-01"), ("C", "2020-01")}
    mask = fx.pool_mask(days, list(close.columns), pool)
    assert mask.tolist() == [[True, True, False], [True, True, False], [True, False, True], [True, False, True]]
    idx = fx.ew_index_returns(close, mask)
    assert np.isnan(idx.iloc[0])
    assert idx.iloc[1] == pytest.approx(np.log1p((0.1 + 0.0) / 2))
    assert idx.iloc[2] == pytest.approx(np.log1p((0.1 + 1.0) / 2))     # 2020-01：A、C
    assert idx.iloc[3] == pytest.approx(np.log1p((0.0 + 1.0) / 2))


def test_dollar_neutral_keeps_gross_and_zero_net():
    w = np.array([0.6, 0.4, 0.0, -0.2, -0.3])
    d = fx.dollar_neutral(w)
    assert d[d > 0].sum() == pytest.approx(0.75) and -d[d < 0].sum() == pytest.approx(0.75)
    assert np.abs(d).sum() == pytest.approx(np.abs(w).sum())
    assert d[0] / d[1] == pytest.approx(1.5) and d[2] == 0
    assert fx.dollar_neutral(np.array([1.0, 0.0])).tolist() == [1.0, 0.0]


def test_index_hedge_neutralizes_beta_to_index():
    w = np.array([0.5, 0.3, -0.2, 0.0, 0.0])
    beta = np.array([0.8, np.nan, 1.5, 1.0, 1.2])
    basket = np.array([0, 1, 2, 3])
    h = fx.index_hedge(w, beta, basket, 5)
    net = 0.5 * 0.8 + 0.3 * 1.0 - 0.2 * 1.5            # 缺值 beta 當 1
    assert h[:4] == pytest.approx([-net / 4] * 4) and h[4] == 0
    assert (w * np.nan_to_num(beta, nan=1.0)).sum() + h.sum() == pytest.approx(0)
    assert fx.index_hedge(w, beta, np.array([], int), 5).tolist() == [0.0] * 5


def test_preregistered_grid_has_eight_configs_and_neighbors_inside():
    cf = fx.configs()
    assert len(cf) == 8
    assert {c["hedge"] for c in cf.values()} == set(fx.HEDGES)
    for name in cf:
        assert set(fx.neighbors(name)) <= set(cf) and name not in fx.neighbors(name)


# ---------------------------------------------------------------- 接起來的組合模擬

def synthetic_ext(n_coins=12):
    """現貨 2019-06-01 ~ 2020-01-31、合約 2020-01-01 ~ 2020-07-18，同樣的代號；兩個時期的報酬不同。"""
    rng = np.random.default_rng(11)
    syms = ["BTCUSDT"] + [f"C{k}USDT" for k in range(1, n_coins)]
    fdays = pd.date_range("2020-01-01", periods=200, freq="D", tz="UTC")
    sdays = pd.date_range("2019-06-01", "2020-01-31", freq="D", tz="UTC")
    fdata, spot = {}, []
    for k, s in enumerate(syms):
        drift = 0.0 if k == 0 else (k - n_coins / 2) * 0.002
        c = 100 * np.exp(np.cumsum(drift + rng.normal(0, 0.01 + 0.002 * k, len(fdays))))
        fdata[s] = pd.DataFrame({"time": fdays, "open": c, "high": c, "low": c, "close": c, "volume": 1e7,
                                 "qv": 1e7 * c, "n": 6})
        cs = 50 * np.exp(np.cumsum(-drift + rng.normal(0, 0.01 + 0.002 * k, len(sdays))))
        spot.append(pd.DataFrame({"symbol": s, "time": sdays, "open": cs, "high": cs, "low": cs, "close": cs,
                                  "volume": 1e6, "quote_volume": 1e6 * cs}))
    funding = {s: pd.DataFrame({"time": fdays + pd.Timedelta(hours=8), "rate": 0.0001}) for s in syms}
    fut = fb.build_panel(fdata, funding=funding, cache=False)
    fut_months = (fdays + pd.Timedelta(days=1)).strftime("%Y-%m").unique()
    fut_pool = {(s, m) for s in syms for m in fut_months}
    spot_pool = {(fx.SPOT_PREFIX + s, m) for s in syms for m in ("2019-07", "2019-08", "2019-09", "2019-10",
                                                                 "2019-11", "2019-12")}
    trade_pool = spot_pool | {(s, m) for s, m in fut_pool if m >= fx.SPLICE_MONTH}
    index_pool = spot_pool | {(fx.SPOT_PREFIX + s, "2020-01") for s in syms} | {(s, m) for s, m in fut_pool
                                                                                if m > "2020-01"}
    p = fx.build_ext_panel(fut, pd.concat(spot, ignore_index=True), trade_pool, index_pool)
    return p, fut, fut_pool, trade_pool


def test_btc_hedge_futures_era_reproduces_pr53_run():
    p, fut, fut_pool, trade_pool = synthetic_ext()
    cfg = {"signal": "combo", "q": 0.3}
    orig = fb.run(fut, fut_pool, cfg, start="2020-01-01")
    ext = fx.run_ext(p, trade_pool, {**cfg, "hedge": "btc"}, start="2019-06-01")
    ext = ext[(ext["era"] == "fut") & (ext["time"] >= pd.Timestamp("2020-01-01", tz="UTC"))]
    assert len(orig) > 100
    m = orig.merge(ext, on=["symbol", "time"], how="outer", indicator=True, suffixes=("_o", "_e"))
    assert (m["_merge"] == "both").all()
    assert np.allclose(m["r_net_o"], m["r_net_e"], atol=1e-12)
    assert (m["side_o"] == m["side_e"]).all()


def test_splice_closes_spot_positions_and_labels_eras():
    p, fut, fut_pool, trade_pool = synthetic_ext()
    rows = fx.run_ext(p, trade_pool, {"signal": "combo", "q": 0.3, "hedge": "btc"}, start="2019-06-01")
    spot_rows = rows[rows["symbol"].str.startswith(fx.SPOT_PREFIX)]
    assert len(spot_rows) and (spot_rows["era"] == "spot").all()
    # 現貨部位最後一天 = 2020-01 第一個週日（2020-01-05），之後只剩合約
    assert spot_rows["time"].max() == pd.Timestamp("2020-01-05", tz="UTC")
    assert (spot_rows["funding_r"] == 0).all()
    last = spot_rows[spot_rows["time"] == spot_rows["time"].max()]
    assert (last["cost_r"] < 0).all()                    # 換約時平倉要付成本
    fut_rows = rows[~rows["symbol"].str.startswith(fx.SPOT_PREFIX)]
    assert (fut_rows["era"] == "fut").all() and fut_rows["time"].min() > pd.Timestamp("2020-01-05", tz="UTC")
    # 現貨期用現貨 BTC 對沖、合約期用合約 BTC
    assert rows.loc[rows["hedge"] & (rows["era"] == "spot"), "symbol"].eq(fx.SPOT_HEDGE).all()
    assert rows.loc[rows["hedge"] & (rows["era"] == "fut"), "symbol"].eq(fb.HEDGE).all()


def test_index_and_none_hedges():
    p, fut, fut_pool, trade_pool = synthetic_ext()
    idx = fx.run_ext(p, trade_pool, {"signal": "combo", "q": 0.3, "hedge": "index"}, start="2019-06-01")
    h = idx[idx["hedge"]]
    assert len(h) and (h["symbol"] == fx.IDX).all()
    assert h["cost_r"].sum() < 0                          # 籃子的換手也要付成本
    assert not idx.loc[~idx["hedge"], "symbol"].isin([fx.IDX]).any()
    assert np.allclose(idx["r_net"], idx["r"] + idx["funding_r"] + idx["cost_r"])
    none = fx.run_ext(p, trade_pool, {"signal": "combo", "q": 0.3, "hedge": "none"}, start="2019-06-01")
    assert not none["hedge"].any()
    # 金額中性：每個再平衡週第一天，多空名目（損益 / 報酬）相等 —— 用第一天 r / 當天報酬還原名目
    first = none[none["time"] == none["time"].min()]
    a = int(np.flatnonzero(p.days == first["time"].iloc[0])[0]) - 1
    notional = []
    for _, row in first.iterrows():
        k = p.syms.index(row["symbol"])
        ret = p.close[a + 1, k] / p.close[a, k] - 1
        notional.append(row["r"] * fx.RISK / ret)
    notional = np.array(notional)
    assert notional[notional > 0].sum() == pytest.approx(-notional[notional < 0].sum(), rel=1e-9)
    # 現貨手續費敏感度只改現貨期
    hi = fx.run_ext(p, trade_pool, {"signal": "combo", "q": 0.3, "hedge": "none"}, start="2019-06-01", spot_fee=0.001)
    assert hi.loc[hi["era"] == "spot", "cost_r"].sum() < none.loc[none["era"] == "spot", "cost_r"].sum()
    assert np.allclose(hi.loc[hi["era"] == "fut", "r_net"].to_numpy(), none.loc[none["era"] == "fut", "r_net"].to_numpy())


def test_perf_and_daily_ret():
    days = pd.date_range("2020-01-01", periods=4, freq="D", tz="UTC")
    rows = pd.DataFrame({"time": days[[0, 0, 2]], "r_net": [1.0, 1.0, -2.0]})
    d = fx.daily_ret(rows)
    assert d.tolist() == pytest.approx([2 * fx.RISK, 0.0, -2 * fx.RISK])
    d2 = fx.daily_ret(rows, lo="2019-12-31", hi="2020-01-05")
    assert len(d2) == 5 and d2.iloc[0] == 0
    pf = fx.perf(pd.Series([0.01, -0.02, 0.01], index=days[:3]))
    assert pf["max_dd_pct"] == pytest.approx(-2.0)
