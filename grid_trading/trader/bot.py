"""自動交易主程式：每根 2H / 4H 收盤後找新訊號、開倉；平時管理已開的倉位。

進場（和網站的實盤維加斯一樣）：
- 只看剛收盤那一根的訊號；程式停機期間錯過的舊訊號不補做。
- 同一個幣、同一個週期一次只做一張；2H 和 4H 可以同時各有一張，同方向也可以（Denny 2026-10-20 決定，和網站一致）。
- 每筆風險 = 權益 × 0.25%（回撤減半時 × 0.5），最小下單量造成的實際風險超過 1% 就不做。

出場（和回測的 tracker 一樣）：止損 1.5 ATR（1R）；TP 1R / 3R / 5R 各平 1/4；TP1 後止損移到保本
（進場價 + 來回手續費）、TP2 後移到 TP1；TP3 後最後 1/4 用 3 ATR 追蹤止損（每根 K 線收盤時更新）；
240 根 K 線還沒結束就市價平掉。

- paper：PaperBroker 直接用 tracker 算結果，和回測完全一致，只記帳。
- testnet / live：LiveBroker 開倉後馬上在交易所掛止損（這張單的數量）和三個止盈（各 1/4），
  之後每分鐘查各條件單的狀態判斷哪個 TP 成交了，再移動止損。程式停了，交易所上的止損還在。
  價位以實際成交價為準（止損距離照訊號的比例），所以 1R 的大小和訊號一致。
"""

from __future__ import annotations

import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from ..binance_history import binance_symbols
from ..signals import get
from ..signals.strategy import LONG, Signal
from ..signals.tracker import breakeven, track
from ..universe import TOP100
from .binance import BinanceError, BinanceFutures
from .risk import RiskRules, can_open, risk_multiplier
from .sizing import Filters, fmt, plan_size

BAR = {"2H": pd.Timedelta(hours=2), "4H": pd.Timedelta(hours=4)}
ORDER = ("2H", "4H")  # 同一個時間點兩個週期都有訊號時，2H 先做
MAX_BARS = 240
LABEL = {"paper": "幣安紙上交易", "testnet": "幣安測試網", "live": "幣安實盤"}


@dataclass
class Config:
    mode: str = "paper"
    risk: float = 0.0025
    max_risk: float = 0.01
    leverage: int = 5
    fee: float = 0.0005
    paper_equity: float = 100.0
    rules: RiskRules = field(default_factory=RiskRules)
    state_dir: Path = Path("trader_state")
    coins: list[str] = field(default_factory=lambda: list(TOP100))
    bars: int = 1000


def bar_close(now: pd.Timestamp, interval: str) -> pd.Timestamp:
    """now 之前最近一次 interval 收盤的時間（從 00:00 UTC 起算，和幣安 K 線一致）。"""
    return now.floor(BAR[interval])


def signal_from(d: dict) -> Signal:
    return Signal(d["symbol"], d["interval"], d["side"], pd.Timestamp(d["time"], unit="ms", tz="UTC"),
                  d["price"], d["sl"], list(d["tps"]), d["strategy"], d.get("info") or {}, d.get("exit") or {})


class Store:
    """狀態檔：持倉、已平倉紀錄、每個週期處理到哪根 K 線。每次改完立刻寫檔（先寫暫存檔再改名）。"""

    def __init__(self, path: Path):
        self.path = path
        try:
            data = json.loads(path.read_text())
        except FileNotFoundError:
            data = {}
        self.positions: dict[str, dict] = data.get("positions", {})
        self.ledger: list[dict] = data.get("ledger", [])
        self.processed: dict[str, int] = data.get("processed", {})
        self.setup: list[str] = data.get("setup", [])
        self.leverage: dict[str, int] = data.get("leverage", {})

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"positions": self.positions, "ledger": self.ledger,
                                   "processed": self.processed, "setup": self.setup,
                                   "leverage": self.leverage}, ensure_ascii=False, indent=1))
        tmp.replace(self.path)


class Notifier:
    def __init__(self, prefix: str, token: str = "", chat_id: str = "", sink=None):
        self.prefix, self.token, self.chat_id = prefix, token, chat_id
        self.sink = sink  # 測試用：收集訊息

    def __call__(self, text: str) -> None:
        msg = f"［{self.prefix}］{text}"
        print(time.strftime("%Y-%m-%d %H:%M:%S"), msg, flush=True)
        if self.sink is not None:
            self.sink.append(msg)
        elif self.token and self.chat_id:
            try:
                from signal_telegram import send
                send([msg], self.token, self.chat_id)
            except Exception as e:  # 推播失敗不影響交易
                print("Telegram 失敗:", e, flush=True)


def _pair(sym: str) -> str:
    return sym.replace("USDT", "/USDT")


def _side_text(side: str) -> str:
    return "做多" if side == LONG else "做空"


class Trader:
    def __init__(self, cfg: Config, data: BinanceFutures, broker, notify: Notifier, filters: dict[str, Filters]):
        self.cfg = cfg
        self.data = data          # 行情（正式站）
        self.broker = broker
        self.notify = notify
        self.filters = filters    # 下單的那個交易所（測試網 / 正式站）有的合約
        self.store = Store(cfg.state_dir / f"{cfg.mode}.json")
        self.strategy = get("vegas_trend")
        self.symbols = {}
        for coin in cfg.coins:
            sym = binance_symbols(coin)[-1]
            if sym in filters:
                self.symbols[sym] = coin

    # ---------- 進場 ----------
    def paused(self) -> bool:
        return (self.cfg.state_dir / "pause").exists()

    def due(self, now: pd.Timestamp) -> list[str]:
        """這次要處理的週期：收盤時間比上次處理的新。第一次啟動只記下時間、不開單（避免開到舊訊號）。"""
        out = []
        for iv in ORDER:
            t = int(bar_close(now, iv).timestamp() * 1000)
            last = self.store.processed.get(iv)
            if last is None:
                self.store.processed[iv] = t
            elif t > last:
                out.append(iv)
        self.store.save()
        return out

    def find_signals(self, interval: str, close: pd.Timestamp) -> list[Signal]:
        def one(sym):
            try:
                df = self.data.klines(sym, interval, self.cfg.bars)
            except Exception as e:
                print(f"{sym} {interval} K 線失敗: {e}", flush=True)
                return []
            closed = df[df["time"] + BAR[interval] <= close].reset_index(drop=True)
            if len(closed) < 250 or closed["time"].iloc[-1] + BAR[interval] != close:
                return []
            return [s for s in self.strategy.scan(closed, sym, interval, start=len(closed) - 1) if s.time == close]
        with ThreadPoolExecutor(4) as ex:
            found = [s for batch in ex.map(one, list(self.symbols)) for s in batch]
        return sorted(found, key=lambda s: s.symbol)

    def blocked(self, sig: Signal) -> str:
        for p in self.store.positions.values():
            if p["symbol"] != sig.symbol:
                continue
            if p["interval"] == sig.interval:
                return f"{_pair(sig.symbol)} {sig.interval} 已有{_side_text(p['side'])}單"
        return ""

    def step_entries(self, now: pd.Timestamp) -> None:
        for iv in self.due(now):
            close = bar_close(now, iv)
            sigs = self.find_signals(iv, close)
            # 沒訊號也印一行，看紀錄就知道每根收盤都有掃（不推 Telegram）
            print(time.strftime("%Y-%m-%d %H:%M:%S"),
                  f"{iv} 收盤 {close:%m-%d %H:%M} UTC 掃描完成：{len(sigs)} 個訊號，持倉 {len(self.store.positions)} 張",
                  flush=True)
            for sig in sigs:
                self.try_open(sig)
            self.store.processed[iv] = int(close.timestamp() * 1000)
            self.store.save()

    def try_open(self, sig: Signal) -> None:
        head = f"{_pair(sig.symbol)} {sig.interval} {_side_text(sig.side)}訊號 @ {sig.price:.6g}"
        reason = self.blocked(sig)
        if not reason and self.paused():
            reason = "已暫停開新倉（pause 檔存在）"
        if not reason and not can_open(len(self.store.positions), self.cfg.rules):
            reason = f"同時持倉已達上限 {self.cfg.rules.max_open} 張"
        if reason:
            self.notify(f"{head}，不做：{reason}")
            return
        mult = risk_multiplier(self.store.ledger, self.cfg.rules)
        equity = self.broker.equity(self.store)
        size, why = plan_size(equity, self.cfg.risk * mult, sig.price, sig.sl, self.filters[sig.symbol],
                              tuple(sig.exit.get("portions") or (0.25, 0.25, 0.25)), self.cfg.max_risk)
        if size is None:
            self.notify(f"{head}，不做：{why}")
            return
        try:
            pos = self.broker.open(sig, size, self.filters[sig.symbol], self.store)
        except Exception as e:
            self.notify(f"{head}，開倉失敗：{e}")
            return
        if pos is None:
            return
        pos["base_usdt"] = equity * self.cfg.risk
        pos["risk_mult"] = mult
        self.store.positions[pos["id"]] = pos
        self.store.save()
        extra = "（風險減半）" if mult < 1 else ""
        extra += f"（最小下單量，實際風險 {size.risk_frac:.2%}）" if size.bumped else ""
        self.notify(f"{head}，已開倉 {pos['qty']} @ {pos['entry']:.6g}，止損 {pos['stop']:.6g}，"
                    f"TP {' / '.join(f'{t:.6g}' for t in pos['tps'])}{extra}")

    # ---------- 持倉管理 ----------
    def step_manage(self, now: pd.Timestamp) -> None:
        if not self.store.positions:
            return
        self.broker.begin_step()
        for pid in list(self.store.positions):
            pos = self.store.positions[pid]
            try:
                result = self.broker.manage(pos, now, self.notify)
            except Exception as e:
                msg = f"{_pair(pos['symbol'])} {pos['interval']} 管理失敗：{e}"
                if pos.get("last_error") != msg:  # 同樣的錯誤只推一次
                    pos["last_error"] = msg
                    self.notify(msg)
                self.store.save()
                continue
            pos.pop("last_error", None)  # 這輪沒出錯，之後同樣的錯誤再出現要再通知一次
            if result is not None:  # 已平倉：result 是實現損益（USDT）
                r = result / pos["base_usdt"] if pos.get("base_usdt") else 0.0
                self.store.ledger.append({"id": pid, "symbol": pos["symbol"], "interval": pos["interval"],
                                          "side": pos["side"], "opened": pos["opened"],
                                          "closed": int(now.timestamp() * 1000), "pnl": result, "r": r})
                del self.store.positions[pid]
                cum = sum(x["r"] for x in self.store.ledger)
                self.notify(f"{_pair(pos['symbol'])} {pos['interval']} {_side_text(pos['side'])}已平倉："
                            f"{result:+.4f} USDT（{r:+.2f}R），累計 {cum:+.2f}R")
            self.store.save()

    def run(self, poll: float = 30, once: bool = False) -> None:
        self.notify(f"啟動：{len(self.symbols)} 個幣，持倉 {len(self.store.positions)} 張")
        self.broker.reconcile(self.store, self.notify)
        self.store.save()
        while True:
            now = pd.Timestamp.now(tz="UTC")
            try:
                # 收盤後等 15 秒再抓 K 線，幣安的最後一根才會是完整的
                if (now - bar_close(now, "2H")) >= pd.Timedelta(seconds=15):
                    self.step_entries(now)
                self.step_manage(now)
            except Exception as e:  # 網路斷線之類，下一輪再試
                print("本輪失敗:", e, flush=True)
            if once:
                return
            time.sleep(poll)


# ---------- 紙上交易 ----------
class PaperBroker:
    """不下單：開倉價 = 訊號收盤價，出場用 tracker 逐根 K 線算（和回測一樣）。"""

    def __init__(self, data: BinanceFutures, cfg: Config):
        self.data, self.cfg = data, cfg

    def equity(self, store: Store) -> float:
        return self.cfg.paper_equity + sum(x["pnl"] for x in store.ledger)

    def reconcile(self, store: Store, notify) -> None:
        pass

    def begin_step(self) -> None:
        pass

    def open(self, sig: Signal, size, f: Filters, store: Store) -> dict:
        qty = size.steps * f.step
        return {"id": f"{sig.symbol}|{sig.interval}|{sig.side}|{int(sig.time.timestamp())}",
                "symbol": sig.symbol, "interval": sig.interval, "side": sig.side,
                "signal": sig.to_dict(), "qty": fmt(qty, f.step), "qty_f": qty,
                "entry": sig.price, "stop": sig.sl, "tps": list(sig.tps),
                "risk_usdt": qty * abs(sig.price - sig.sl), "events": 0,
                "opened": int(sig.time.timestamp() * 1000)}

    def manage(self, pos: dict, now: pd.Timestamp, notify) -> float | None:
        sig = signal_from(pos["signal"])
        df = self.data.klines(sig.symbol, sig.interval, MAX_BARS + 20)
        res = track(sig, df[df["time"] <= now], self.cfg.fee, MAX_BARS)
        for e in res.events[pos["events"]:]:
            if e.kind in ("TP1", "TP2", "TP3", "STOP_MOVED", "TRAIL_MOVED"):
                notify(f"{_pair(sig.symbol)} {sig.interval} {_side_text(sig.side)}：{e.text}")
        pos["events"] = len(res.events)
        pos["stop"] = res.stop
        if res.status == "CLOSED":
            return res.realized_r * pos["risk_usdt"]
        return None


# ---------- 測試網 / 實盤 ----------
def client_tag(sig: Signal) -> str:
    """掛單用的 client id 前綴。幣安要求同一帳戶內不能重複，所以要帶幣名（同一根 K 線常有好幾個幣同時出訊號）；
    id 最長 36 字元，後面還要接 e / s12 / t1 等尾巴，太長就把幣名縮成雜湊。"""
    name = "".join(c for c in sig.symbol if c.isalnum())
    tag = f"v{sig.interval}{int(sig.time.timestamp())}{sig.side[0]}{name}"
    if len(tag) > 30:
        tag = f"v{sig.interval}{int(sig.time.timestamp())}{sig.side[0]}{hashlib.md5(sig.symbol.encode()).hexdigest()[:10]}"
    return tag


class LiveBroker:
    """同一個幣同方向可能同時有 2H 和 4H 兩張單，在幣安上會合成同一個倉位（雙向持倉只有一個 LONG / SHORT），
    所以每張單的止損、止盈都用自己的數量掛，成交與否看各自條件單的狀態，不看合併後的持倉數量。
    損益也用自己的成交紀錄算（成交價 × 數量，扣來回手續費）。"""

    FILLED = ("TRIGGERED", "FINISHED")

    def __init__(self, client: BinanceFutures, data: BinanceFutures, cfg: Config):
        self.client, self.data, self.cfg = client, data, cfg
        self._open: dict[str, set] = {}

    def equity(self, store: Store) -> float:
        return self.client.balance_usdt()

    def begin_step(self) -> None:
        self._open = {}  # 每輪重抓各幣的掛單（同一個幣只抓一次）

    def open_ids(self, symbol: str) -> set:
        if symbol not in self._open:
            self._open[symbol] = {o.get("algoId") for o in self.client.open_conditionals(symbol)}
        return self._open[symbol]

    def status(self, pos: dict, oid) -> tuple[str, float]:
        """條件單的狀態：("open" / "filled" / "pending" / "gone", 成交價)。
        pending 是剛觸發、幣安還在送出市價單（TRIGGERING）：這時掛單清單裡已經沒有它，但還不算成交也不是不見，下一輪再看。"""
        if oid is None:
            return "gone", 0.0
        if oid in self.open_ids(pos["symbol"]):
            return "open", 0.0
        o = self.client.algo_order(oid)
        st = str(o.get("algoStatus") or "").upper()
        price = float(o.get("actualPrice") or 0) or float(o.get("triggerPrice") or 0)
        pos["last_algo_status"] = st  # 補掛時寫進推播，之後才查得出止損單是怎麼不見的
        if st in self.FILLED:
            return "filled", price
        return ("pending" if st == "TRIGGERING" else "gone"), price

    def ensure_setup(self, symbol: str, store: Store, leverage: int) -> None:
        """雙向持倉、逐倉；槓桿用 leverage（每個幣記住目前設定，要更低時才改）。"""
        if "*hedge" not in store.setup:
            if not self.client.hedge_mode():
                self.client.set_hedge_mode()
            store.setup.append("*hedge")
        if symbol not in store.setup:
            self.client.set_isolated(symbol)
            store.setup.append(symbol)
        if store.leverage.get(symbol) != leverage:
            self.client.set_leverage(symbol, leverage)
            store.leverage[symbol] = leverage
        store.save()

    def leverage_for(self, sig: Signal) -> int:
        """止損越遠槓桿越低，讓強平價離進場至少是止損距離的 2 倍（槓桿 ≤ 1 / (2 × 止損%)），最多 cfg.leverage。"""
        stop_pct = abs(sig.price - sig.sl) / sig.price
        return max(1, min(self.cfg.leverage, int(1 / (2 * stop_pct)) if stop_pct > 0 else self.cfg.leverage))

    def reconcile(self, store: Store, notify) -> None:
        """啟動時比對：交易所有、狀態檔沒有的倉位只提醒，不碰；狀態檔的單止損不見（不是成交）就補掛。"""
        self.begin_step()
        migrated = [p for p in store.positions.values() if "remaining" not in p]
        for p in migrated:
            # 2026-10-20 以前開的單：沒有 remaining / fills，止損是「全部平倉」。補上欄位，止損改成用自己的數量
            p["remaining"] = max(0.0, p["qty_f"] - sum(p["slices"][:p["tps_hit"]]))
            p["fills"] = [[p["slices"][n], p["tps"][n]] for n in range(p["tps_hit"])]
        live = self.client.positions()
        expected: dict[tuple, float] = {}
        for p in store.positions.values():
            key = (p["symbol"], p["side"])
            expected[key] = expected.get(key, 0.0) + p["remaining"]
        for key, amt in live.items():
            if amt > expected.get(key, 0.0) + 1e-9:
                notify(f"注意：{_pair(key[0])} {key[1]} 有 {amt}，比這個程式記錄的多，多出來的不會管理")
        for pos in migrated:
            if self.status(pos, pos.get("stop_id"))[0] == "open" and pos["remaining"] > pos["step"] / 2:
                self.move_stop(pos, pos["stop"])
        for pos in store.positions.values():
            if self.status(pos, pos.get("stop_id"))[0] == "gone":
                pos["stop_version"] = pos.get("stop_version", 0) + 1
                pos["stop_id"] = self._place_stop(pos, pos["stop"], pos["stop_version"])
                notify(f"{_pair(pos['symbol'])} 止損單不見了，已補掛在 {pos['stop']:.6g}")

    def open(self, sig: Signal, size, f: Filters, store: Store) -> dict | None:
        self.ensure_setup(sig.symbol, store, self.leverage_for(sig))
        d = sig.direction
        ps = sig.side  # 雙向持倉：LONG / SHORT
        buy, sell = ("BUY", "SELL") if d > 0 else ("SELL", "BUY")
        tag = client_tag(sig)
        order = self.client.market_order(sig.symbol, buy, ps, fmt(size.steps * f.step, f.step), f"{tag}e")
        qty = float(order.get("executedQty") or 0)
        entry = float(order.get("avgPrice") or 0) or self.client.price(sig.symbol)
        if qty <= 0:
            raise RuntimeError(f"市價單沒有成交：{order}")
        # 價位以實際成交價重算，止損距離照訊號的比例
        risk = abs(sig.price - sig.sl) / sig.price * entry
        r_mult = [abs(t - sig.price) / abs(sig.price - sig.sl) for t in sig.tps]
        stop = entry - d * risk
        tps = [entry + d * k * risk for k in r_mult]
        pos = {"id": f"{sig.symbol}|{sig.interval}|{sig.side}|{int(sig.time.timestamp())}",
               "symbol": sig.symbol, "interval": sig.interval, "side": sig.side, "signal": sig.to_dict(),
               "qty": fmt(qty, f.step), "qty_f": qty, "remaining": qty, "fills": [],
               "step": f.step, "tick": f.tick,
               "slices": [s * f.step for s in size.slices], "entry": entry, "stop": stop, "tps": tps,
               "be": breakeven(Signal(sig.symbol, sig.interval, sig.side, sig.time, entry, stop, tps, ""),
                               self.cfg.fee),
               "trail": float(sig.exit.get("trail") or 0) / sig.price * entry,
               "risk_usdt": qty * risk, "tps_hit": 0, "trail_from": None,
               "opened": int(time.time() * 1000), "close_side": sell, "tag": tag}
        try:
            pos["stop_id"] = self._place_stop(pos, stop, 0)
        except Exception as e:
            # 掛不上止損就馬上平掉這張單，絕不留沒有止損的倉位
            self.client.market_order(sig.symbol, sell, ps, pos["qty"], f"{tag}x")
            raise RuntimeError(f"止損掛單失敗，已市價平倉：{e}")
        pos["tp_ids"] = []
        for n, (tp, q) in enumerate(zip(tps, pos["slices"][:3])):
            try:
                r = self.client.conditional_order(sig.symbol, sell, ps, "TAKE_PROFIT_MARKET", fmt(tp, f.tick),
                                                  quantity=fmt(q, f.step), client_id=f"{tag}t{n + 1}")
                pos["tp_ids"].append(r.get("algoId"))
            except Exception as e:
                pos["tp_ids"].append(None)
                print(f"{sig.symbol} TP{n + 1} 掛單失敗: {e}", flush=True)
        return pos

    def _place_stop(self, pos: dict, price: float, version: int):
        """止損用這張單剩下的數量掛（雙向持倉下 SELL+LONG / BUY+SHORT 只會減倉，不會反向開倉）。"""
        r = self.client.conditional_order(pos["symbol"], pos["close_side"], pos["side"], "STOP_MARKET",
                                          fmt(price, pos["tick"]), quantity=fmt(pos["remaining"], pos["step"]),
                                          client_id=f"{pos['tag']}s{version}")
        if pos["symbol"] in getattr(self, "_open", {}):
            self._open[pos["symbol"]].add(r.get("algoId"))  # 這一輪的掛單快取也要有新止損
        return r.get("algoId")

    def _close_rest(self, pos: dict, reason: str) -> None:
        """市價平掉這張單剩下的數量，並記下成交。"""
        if pos["remaining"] <= pos["step"] / 2:
            return
        try:
            r = self.client.market_order(pos["symbol"], pos["close_side"], pos["side"],
                                         fmt(pos["remaining"], pos["step"]), f"{pos['tag']}{reason}")
        except BinanceError as e:
            if e.code != -2022 or not self._flat(pos):  # -2022：倉位已經不在，沒東西可平
                raise
            r = {}  # 被別的單平掉了（止損或止盈已成交）：照目前價格記帳
        price = float(r.get("avgPrice") or 0) or self.client.price(pos["symbol"])
        pos["fills"].append([pos["remaining"], price])
        pos["remaining"] = 0.0

    def _flat(self, pos: dict) -> bool:
        """交易所上這個幣這一邊已經沒有倉位。"""
        return self.client.positions().get((pos["symbol"], pos["side"]), 0.0) <= pos["step"] / 2

    def move_stop(self, pos: dict, price: float) -> None:
        """先掛新止損再取消舊的；掛不上時先取消再掛；還是不行（價格已越過）就市價平掉。"""
        version = pos.get("stop_version", 0) + 1
        old = pos.get("stop_id")
        try:
            new = self._place_stop(pos, price, version)
            if old:
                self.client.cancel_conditional(old)
        except BinanceError:
            if old:
                self.client.cancel_conditional(old)
            try:
                new = self._place_stop(pos, price, version)
            except BinanceError as e:
                pos["stop_id"], pos["stop"], pos["stop_version"] = None, price, version
                self._close_rest(pos, f"m{version}")
                raise RuntimeError(f"止損移不上去，已市價平倉：{e}")
        pos["stop_id"], pos["stop"], pos["stop_version"] = new, price, version

    def pnl(self, pos: dict) -> float:
        d = 1 if pos["side"] == LONG else -1
        gross = sum(d * (p - pos["entry"]) * q for q, p in pos["fills"])
        fees = self.cfg.fee * (pos["entry"] * pos["qty_f"] + sum(p * q for q, p in pos["fills"]))
        return gross - fees

    def finish(self, pos: dict) -> float:
        for oid in [pos.get("stop_id")] + list(pos.get("tp_ids") or []):
            if oid and oid in self.open_ids(pos["symbol"]):
                self.client.cancel_conditional(oid)
        return self.pnl(pos)

    def manage(self, pos: dict, now: pd.Timestamp, notify) -> float | None:
        name = f"{_pair(pos['symbol'])} {pos['interval']} {_side_text(pos['side'])}"
        if pos["remaining"] <= pos["step"] / 2:  # 上一輪已經平掉（例如移止損失敗時市價平倉）
            return self.finish(pos)
        st, price = self.status(pos, pos.get("stop_id"))
        if st == "filled":
            pos["fills"].append([pos["remaining"], price or pos["stop"]])
            pos["remaining"] = 0.0
            return self.finish(pos)
        if st == "pending":  # 止損剛觸發，等下一輪看成交結果
            return None
        # 止損單不見了（不是成交）：先處理止盈成交再補掛。止盈成交後倉位變小，
        # 用舊的剩餘數量補掛會被幣安以 -2022 拒絕；止盈的處理會順便用新的剩餘數量重掛止損。
        stop_missing, stop_version = st == "gone", pos.get("stop_version", 0)
        d = 1 if pos["side"] == LONG else -1
        while pos["tps_hit"] < 3:
            n = pos["tps_hit"]
            st, price = self.status(pos, pos["tp_ids"][n])
            if st != "filled":
                break
            q = pos["slices"][n]
            pos["fills"].append([q, price or pos["tps"][n]])
            pos["remaining"] = max(0.0, pos["remaining"] - q)
            pos["tps_hit"] += 1
            notify(f"{name}：TP{n + 1} 觸及 {pos['tps'][n]:.6g}")
            if n == 0:
                self.move_stop(pos, pos["be"])
                notify(f"{name}：保護止損設於 {pos['be']:.6g}")
            elif n == 1:
                self.move_stop(pos, pos["tps"][0])
                notify(f"{name}：保護止損設於 {pos['tps'][0]:.6g}")
            else:
                pos["trail_from"] = int(now.timestamp() * 1000)
                self.move_stop(pos, pos["stop"])  # 數量改成最後剩下的那一份
        if pos["remaining"] <= pos["step"] / 2:
            return self.finish(pos)
        if stop_missing and pos.get("stop_version", 0) == stop_version:  # 止盈沒有順便重掛
            why = f"（交易所狀態 {pos.get('last_algo_status') or '查無'}）"
            pos["stop_version"] = stop_version + 1
            try:
                pos["stop_id"] = self._place_stop(pos, pos["stop"], pos["stop_version"])
            except BinanceError as e:
                if e.code == -2021:  # 止損價已經被價格越過，掛不上去
                    pos["stop_id"] = None
                    self._close_rest(pos, f"m{pos['stop_version']}")
                    notify(f"{name}：止損單不見了{why}，而且價格已越過 {pos['stop']:.6g}，已市價平倉")
                    return self.finish(pos)
                if e.code == -2022 and self._flat(pos):  # 倉位已經不在交易所
                    pos["stop_id"] = None
                    self._close_rest(pos, f"m{pos['stop_version']}")
                    notify(f"{name}：止損單不見了{why}，交易所上已經沒有這個倉位，照目前價格記帳")
                    return self.finish(pos)
                raise
            notify(f"{name}：止損單不見了{why}，已補掛在 {pos['stop']:.6g}")
        bar = BAR[pos["interval"]]
        if pos["tps_hit"] >= 3 and pos["trail"] > 0:
            # 追蹤止損：TP3 那根 K 線起的最高價（做多）/ 最低價（做空）往回 3 ATR，每根收盤後更新
            last_close = int(bar_close(now, pos["interval"]).timestamp() * 1000)
            if pos.get("trail_checked") != last_close:
                df = self.client.klines(pos["symbol"], pos["interval"], MAX_BARS + 20)
                start = pd.Timestamp(pos["trail_from"], unit="ms", tz="UTC").floor(bar)
                seg = df[(df["time"] >= start) & (df["time"] + bar <= now)]
                if len(seg):
                    ext = float(seg["high"].max() if d > 0 else seg["low"].min())
                    ts = ext - d * pos["trail"]
                    if d * (ts - pos["stop"]) > pos["tick"]:
                        self.move_stop(pos, ts)
                        notify(f"{name}：追蹤止損移到 {ts:.6g}")
                pos["trail_checked"] = last_close
        opened = pd.Timestamp(pos["signal"]["time"], unit="ms", tz="UTC")
        if now - opened >= MAX_BARS * bar:
            self._close_rest(pos, "x")
            notify(f"{name}：超過 {MAX_BARS} 根未結束，市價平倉")
            return self.finish(pos)
        return None
