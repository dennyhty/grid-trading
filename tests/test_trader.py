import pandas as pd
import pytest

from grid_trading.signals.strategy import LONG, SHORT, Signal
from grid_trading.signals.tracker import track
from grid_trading.trader.bot import Config, LiveBroker, Notifier, PaperBroker, Trader, bar_close
from grid_trading.trader.risk import RiskRules, can_open, risk_multiplier
from grid_trading.trader.sizing import Filters, fmt, parse_filters, plan_size, split

T0 = pd.Timestamp("2026-10-19 08:00", tz="UTC")
F = Filters(step=0.001, min_qty=0.001, max_qty=1e6, tick=0.01, min_notional=5.0)


# ---------- 倉位大小 ----------
def test_size_follows_risk():
    size, _ = plan_size(1000, 0.0025, 100.0, 96.0, F)
    assert size.steps * F.step == pytest.approx(0.625)  # 2.5U / 4
    assert size.risk_frac == pytest.approx(0.0025)
    assert sum(size.slices) == size.steps and not size.bumped


def test_min_notional_bumps_then_caps_risk():
    btc = Filters(step=0.001, min_qty=0.001, max_qty=1e6, tick=0.1, min_notional=100.0)
    size, why = plan_size(100, 0.0025, 70000.0, 68600.0, btc)  # 2% 止損，最少 100U → 風險 2%
    assert size is None and "超過上限" in why
    size, _ = plan_size(100, 0.0025, 70000.0, 68600.0, btc, max_risk=0.06)  # 要 4 個單位才分得成四段
    assert size.bumped and size.steps * btc.step * 70000 >= 100


def test_split_keeps_a_remainder_for_trailing():
    assert split(4) == [1, 1, 1, 1]
    assert split(10) == [2, 2, 2, 4]
    assert all(p >= 1 for p in split(5))


def test_fmt_rounds_to_unit():
    assert fmt(0.12345, 0.001) == "0.123"
    assert fmt(123.456, 0.1) == "123.5"
    assert fmt(5.0, 1.0) == "5"


def test_parse_filters_keeps_usdt_perps():
    info = {"symbols": [
        {"symbol": "XUSDT", "status": "TRADING", "contractType": "PERPETUAL", "quoteAsset": "USDT",
         "filters": [{"filterType": "LOT_SIZE", "stepSize": "0.01", "minQty": "0.01", "maxQty": "1000"},
                     {"filterType": "MARKET_LOT_SIZE", "stepSize": "0.01", "minQty": "0.01", "maxQty": "100"},
                     {"filterType": "PRICE_FILTER", "tickSize": "0.001"},
                     {"filterType": "MIN_NOTIONAL", "notional": "5"}]},
        {"symbol": "YUSDT", "status": "SETTLING", "contractType": "PERPETUAL", "quoteAsset": "USDT", "filters": []}]}
    f = parse_filters(info)
    assert list(f) == ["XUSDT"] and f["XUSDT"].max_qty == 100 and f["XUSDT"].min_notional == 5


# ---------- 風控 ----------
def test_half_risk_after_drawdown_until_new_high():
    rules = RiskRules(dd_half_r=20)
    led = [{"r": 10}] + [{"r": -1}] * 20
    assert risk_multiplier(led, rules) == 0.5
    assert risk_multiplier(led + [{"r": 5}], rules) == 0.5      # 回撤縮小但還沒新高，照樣減半
    assert risk_multiplier(led + [{"r": 20}], rules) == 1.0     # 回到新高
    assert risk_multiplier(led, RiskRules()) == 1.0
    assert can_open(14, RiskRules(max_open=15)) and not can_open(15, RiskRules(max_open=15))


# ---------- 測試用的假交易所 ----------
def candles(rows, start=T0, freq="2h"):
    df = pd.DataFrame(rows, columns=["high", "low", "close"])
    df["open"] = df["close"]
    df["volume"] = 1.0
    df["time"] = pd.date_range(start, periods=len(df), freq=freq, tz="UTC")
    return df


class FakeExchange:
    def __init__(self, price=100.0, df=None):
        self.px = price
        self.df = df
        self.pos = {}
        self.algo = {}
        self.next_id = 1
        self.pnl = 0.0
        self.hedge = False
        self.fail_stop = False
        self.done = {}

    # 行情
    def klines(self, symbol, interval, limit=500):
        return self.df

    def price(self, symbol):
        return self.px

    # 帳戶
    def balance_usdt(self):
        return 1000.0

    def positions(self):
        return {k: v for k, v in self.pos.items() if v > 1e-12}

    def hedge_mode(self):
        return self.hedge

    def set_hedge_mode(self):
        self.hedge = True

    def set_leverage(self, s, lev):
        pass

    def set_isolated(self, s):
        pass

    def market_order(self, symbol, side, ps, quantity, client_id):
        q = float(quantity)
        opening = (side == "BUY") == (ps == "LONG")
        self.pos[(symbol, ps)] = self.pos.get((symbol, ps), 0.0) + (q if opening else -q)
        return {"executedQty": quantity, "avgPrice": str(self.px)}

    def conditional_order(self, symbol, side, ps, kind, trigger, quantity=None, client_id=None):
        if kind == "STOP_MARKET" and self.fail_stop:
            from grid_trading.trader.binance import BinanceError
            raise BinanceError(400, -2021, "Order would immediately trigger.")
        oid = self.next_id
        self.next_id += 1
        self.algo[oid] = {"algoId": oid, "symbol": symbol, "ps": ps, "type": kind, "trigger": float(trigger),
                          "qty": None if quantity is None else float(quantity)}
        return {"algoId": oid}

    def cancel_conditional(self, oid):
        o = self.algo.pop(oid, None)
        if o:
            self.done[oid] = {**o, "algoStatus": "CANCELED"}

    def algo_order(self, oid):
        o = self.done[oid]
        return {"algoStatus": o["algoStatus"], "triggerPrice": str(o["trigger"]), "actualPrice": "0"}

    def open_conditionals(self, symbol):
        return [o for o in self.algo.values() if o["symbol"] == symbol]

    def realized_since(self, symbol, ps, start):
        return self.pnl

    def trigger(self, kind, n=0, oid=None):
        """讓第 n 張某類條件單（或指定 id）成交。"""
        o = self.algo[oid] if oid else [o for o in self.algo.values() if o["type"] == kind][n]
        key = (o["symbol"], o["ps"])
        self.pos[key] -= self.pos[key] if o["qty"] is None else o["qty"]
        del self.algo[o["algoId"]]
        self.done[o["algoId"]] = {**o, "algoStatus": "FINISHED"}


def make_signal(side=LONG, price=100.0, risk=4.0, interval="2H", t=T0):
    d = 1 if side == LONG else -1
    return Signal("XUSDT", interval, side, t, price, price - d * risk, [price + d * k * risk for k in (1, 3, 5)],
                  "vegas_trend", {"atr": risk / 1.5}, {"portions": [0.25, 0.25, 0.25], "trail": 8.0, "trail_atr": 3})


def make_trader(tmp_path, broker, ex, rules=RiskRules()):
    sink = []
    cfg = Config(mode="testnet", state_dir=tmp_path, coins=["X"], rules=rules)
    tr = Trader(cfg, ex, broker, Notifier("test", sink=sink), {"XUSDT": F})
    return tr, sink


# ---------- 實盤 / 測試網 ----------
def test_live_open_places_stop_and_three_tps(tmp_path):
    ex = FakeExchange(price=101.0)
    tr, sink = make_trader(tmp_path, LiveBroker(ex, ex, Config()), ex)
    tr.try_open(make_signal())
    pos = next(iter(tr.store.positions.values()))
    assert ex.hedge
    # 價位以成交價 101 重算：止損距離照訊號比例 4%
    assert pos["entry"] == 101.0 and pos["stop"] == pytest.approx(101 - 4.04)
    assert pos["tps"] == pytest.approx([101 + 4.04, 101 + 3 * 4.04, 101 + 5 * 4.04])
    kinds = sorted(o["type"] for o in ex.algo.values())
    assert kinds == ["STOP_MARKET"] + ["TAKE_PROFIT_MARKET"] * 3
    stop = next(o for o in ex.algo.values() if o["type"] == "STOP_MARKET")
    assert stop["qty"] == pytest.approx(pos["qty_f"])  # 止損用這張單自己的數量
    assert "已開倉" in sink[-1]


def test_live_moves_stop_after_each_tp_then_trails(tmp_path):
    rows = [(101, 99, 100)] * 3
    ex = FakeExchange(price=100.0, df=candles(rows))
    broker = LiveBroker(ex, ex, Config())
    tr, sink = make_trader(tmp_path, broker, ex)
    tr.try_open(make_signal())
    pid = next(iter(tr.store.positions))
    stop_of = lambda: next(o["trigger"] for o in ex.algo.values() if o["type"] == "STOP_MARKET")
    now = T0 + pd.Timedelta(minutes=30)

    ex.trigger("TAKE_PROFIT_MARKET", 0)
    tr.step_manage(now)
    assert stop_of() == pytest.approx(100 * 1.001)  # 保本 = 進場價 + 來回手續費
    ex.trigger("TAKE_PROFIT_MARKET", 0)
    tr.step_manage(now)
    assert stop_of() == pytest.approx(104.0)        # TP2 後移到 TP1
    assert len([o for o in ex.algo.values() if o["type"] == "STOP_MARKET"]) == 1

    ex.trigger("TAKE_PROFIT_MARKET", 0)
    tr.step_manage(now)
    assert tr.store.positions[pid]["tps_hit"] == 3
    # 下一根收盤後：TP3 那根起最高 130 → 追蹤止損 130 − 8 = 122
    ex.df = candles([(101, 99, 100), (130, 118, 125)], start=T0)
    tr.step_manage(T0 + pd.Timedelta(hours=4, seconds=20))
    assert stop_of() == pytest.approx(122.0)
    assert any("追蹤止損移到 122" in m for m in sink)

    stop = next(o for o in ex.algo.values() if o["type"] == "STOP_MARKET")
    assert stop["qty"] == pytest.approx(0.625 - 3 * 0.156)  # 最後剩下的那一份
    ex.trigger("STOP_MARKET")
    tr.step_manage(T0 + pd.Timedelta(hours=5))
    assert not tr.store.positions and not ex.algo and not ex.positions()
    q, rest = 0.156, 0.625 - 3 * 0.156
    gross = q * 4 + q * 12 + q * 20 + rest * 22
    fees = 0.0005 * (100 * 0.625 + q * (104 + 112 + 120) + rest * 122)
    assert tr.store.ledger[-1]["pnl"] == pytest.approx(gross - fees)
    assert tr.store.ledger[-1]["r"] == pytest.approx((gross - fees) / 2.5)


def test_live_stop_failure_closes_position(tmp_path):
    ex = FakeExchange()
    ex.fail_stop = True
    tr, sink = make_trader(tmp_path, LiveBroker(ex, ex, Config()), ex)
    tr.try_open(make_signal())
    assert not tr.store.positions and not ex.positions()
    assert "已市價平倉" in sink[-1]


def test_one_per_coin_and_interval(tmp_path):
    ex = FakeExchange()
    tr, sink = make_trader(tmp_path, LiveBroker(ex, ex, Config()), ex)
    tr.try_open(make_signal(interval="2H"))
    tr.try_open(make_signal(side=SHORT, interval="2H"))     # 同幣同週期：一次一張
    assert len(tr.store.positions) == 1 and "已有做多單" in sink[-1]
    tr.try_open(make_signal(interval="4H"))                 # 另一個週期同方向：可以（2026-10-20 起）
    assert len(tr.store.positions) == 2


def test_two_same_side_positions_are_managed_separately(tmp_path):
    ex = FakeExchange(df=candles([(101, 99, 100)] * 3))
    tr, sink = make_trader(tmp_path, LiveBroker(ex, ex, Config()), ex)
    tr.try_open(make_signal(interval="2H"))
    tr.try_open(make_signal(interval="4H", t=T0 + pd.Timedelta(hours=4)))
    p2h, p4h = sorted(tr.store.positions.values(), key=lambda p: p["interval"])
    assert ex.positions()[("XUSDT", "LONG")] == pytest.approx(1.25)  # 幣安上合成一個倉位
    ex.trigger("STOP_MARKET", oid=p2h["stop_id"])           # 只有 2H 那張止損
    tr.step_manage(T0 + pd.Timedelta(hours=1))
    assert list(tr.store.positions) == [p4h["id"]]
    assert ex.positions()[("XUSDT", "LONG")] == pytest.approx(0.625)
    assert tr.store.ledger[-1]["interval"] == "2H" and tr.store.ledger[-1]["pnl"] < 0
    assert sum(o["type"] == "TAKE_PROFIT_MARKET" for o in ex.algo.values()) == 3  # 2H 的止盈已取消


def test_max_open_and_pause(tmp_path):
    ex = FakeExchange()
    tr, sink = make_trader(tmp_path, LiveBroker(ex, ex, Config()), ex, RiskRules(max_open=0))
    tr.try_open(make_signal())
    assert not tr.store.positions and "上限" in sink[-1]
    tr.cfg.rules = RiskRules()
    (tmp_path / "pause").touch()
    tr.try_open(make_signal())
    assert not tr.store.positions and "暫停" in sink[-1]


def test_reconcile_replaces_missing_stop(tmp_path):
    ex = FakeExchange()
    broker = LiveBroker(ex, ex, Config())
    tr, sink = make_trader(tmp_path, broker, ex)
    tr.try_open(make_signal())
    stop_id = next(o["algoId"] for o in ex.algo.values() if o["type"] == "STOP_MARKET")
    ex.cancel_conditional(stop_id)
    broker.reconcile(tr.store, tr.notify)
    assert any(o["type"] == "STOP_MARKET" for o in ex.algo.values())
    assert "補掛" in sink[-1]


# ---------- 紙上交易 = 回測 ----------
def test_paper_matches_tracker(tmp_path):
    rows = [(101, 99, 100), (105, 99, 104), (113, 103, 112), (121, 111, 120), (121, 110, 112)]
    ex = FakeExchange(df=candles(rows))
    cfg = Config(mode="paper", state_dir=tmp_path, coins=["X"])
    broker = PaperBroker(ex, cfg)
    sink = []
    tr = Trader(cfg, ex, broker, Notifier("paper", sink=sink), {"XUSDT": F})
    sig = make_signal()
    tr.try_open(sig)
    tr.step_manage(T0 + pd.Timedelta(hours=10))
    expect = track(sig, candles(rows), 0.0005, 240)
    assert expect.status == "CLOSED"
    row = tr.store.ledger[-1]
    pos_risk = 0.062 * 4.0  # 權益 100U × 0.25% ÷ 4，捨去到 0.001
    assert row["pnl"] == pytest.approx(expect.realized_r * pos_risk)
    assert row["r"] == pytest.approx(row["pnl"] / 0.25)
    assert broker.equity(tr.store) == pytest.approx(100 + row["pnl"])


# ---------- 進場時間 ----------
def test_first_run_does_not_trade_old_bars(tmp_path):
    ex = FakeExchange()
    tr, _ = make_trader(tmp_path, LiveBroker(ex, ex, Config()), ex)
    now = T0 + pd.Timedelta(minutes=1)
    assert tr.due(now) == []                               # 第一次只記時間
    assert tr.due(now + pd.Timedelta(hours=2)) == ["2H"]   # 下一根 2H 收盤
    assert tr.due(now + pd.Timedelta(hours=4)) == ["2H", "4H"]
    assert bar_close(T0 + pd.Timedelta(hours=3), "4H") == pd.Timestamp("2026-10-19 08:00", tz="UTC")


def test_leverage_lowered_for_wide_stops(tmp_path):
    ex = FakeExchange()
    calls = []
    ex.set_leverage = lambda s, lev: calls.append(lev)
    tr, _ = make_trader(tmp_path, LiveBroker(ex, ex, Config()), ex)
    tr.try_open(make_signal(risk=4.0))                       # 止損 4% → 5 倍
    tr.try_open(make_signal(risk=16.0, interval="4H"))       # 止損 16% → 3 倍（強平在約 33% 外）
    assert calls == [5, 3]
    assert tr.store.leverage == {"XUSDT": 3}
