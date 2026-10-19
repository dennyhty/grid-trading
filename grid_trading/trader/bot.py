"""自動交易主程式：每根 2H / 4H 收盤後找新訊號、開倉；平時管理已開的倉位。

進場（和網站的實盤維加斯一樣）：
- 只看剛收盤那一根的訊號；程式停機期間錯過的舊訊號不補做。
- 同一個幣同方向只做一張（2H、4H 共用，同時出現 2H 優先）；同一個幣同週期有單時也不做新單（和網站一樣）。
- 每筆風險 = 權益 × 0.25%（回撤減半時 × 0.5），最小下單量造成的實際風險超過 1% 就不做。

出場（和回測的 tracker 一樣）：止損 1.5 ATR（1R）；TP 1R / 3R / 5R 各平 1/4；TP1 後止損移到保本
（進場價 + 來回手續費）、TP2 後移到 TP1；TP3 後最後 1/4 用 3 ATR 追蹤止損（每根 K 線收盤時更新）；
240 根 K 線還沒結束就市價平掉。

- paper：PaperBroker 直接用 tracker 算結果，和回測完全一致，只記帳。
- testnet / live：LiveBroker 開倉後馬上在交易所掛止損（全部平倉）和三個止盈（各 1/4），
  之後每分鐘看持倉數量判斷哪個 TP 成交了，再移動止損。程式停了，交易所上的止損還在。
  價位以實際成交價為準（止損距離照訊號的比例），所以 1R 的大小和訊號一致。
"""

from __future__ import annotations

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

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"positions": self.positions, "ledger": self.ledger,
                                   "processed": self.processed, "setup": self.setup}, ensure_ascii=False, indent=1))
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
            if p["side"] == sig.side:
                return f"{_pair(sig.symbol)} 已有{_side_text(sig.side)}單（{p['interval']}）"
            if p["interval"] == sig.interval:
                return f"{_pair(sig.symbol)} {sig.interval} 已有反向單"
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
class LiveBroker:
    def __init__(self, client: BinanceFutures, data: BinanceFutures, cfg: Config):
        self.client, self.data, self.cfg = client, data, cfg

    def equity(self, store: Store) -> float:
        return self.client.balance_usdt()

    def begin_step(self) -> None:
        """每輪管理開始時抓一次全部持倉，不用每張單各抓一次。"""
        self._live = self.client.positions()

    def ensure_setup(self, symbol: str, store: Store) -> None:
        if "*hedge" not in store.setup:
            if not self.client.hedge_mode():
                self.client.set_hedge_mode()
            store.setup.append("*hedge")
        if symbol not in store.setup:
            self.client.set_isolated(symbol)
            self.client.set_leverage(symbol, self.cfg.leverage)
            store.setup.append(symbol)
            store.save()

    def reconcile(self, store: Store, notify) -> None:
        """啟動時比對：狀態檔有、交易所沒有的倉位交給 manage() 結算；交易所有、狀態檔沒有的只提醒，不碰。"""
        live = self.client.positions()
        mine = {(p["symbol"], p["side"]) for p in store.positions.values()}
        for (sym, side), amt in live.items():
            if (sym, side) not in mine:
                notify(f"注意：{_pair(sym)} {side} 有 {amt} 不是這個程式開的倉，不會管理它")
        for pos in store.positions.values():
            if (pos["symbol"], pos["side"]) in live:
                self.ensure_stop(pos, notify)

    def open(self, sig: Signal, size, f: Filters, store: Store) -> dict | None:
        self.ensure_setup(sig.symbol, store)
        d = sig.direction
        ps = sig.side  # 雙向持倉：LONG / SHORT
        buy, sell = ("BUY", "SELL") if d > 0 else ("SELL", "BUY")
        tag = f"vt{sig.interval}{int(sig.time.timestamp())}"
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
               "qty": fmt(qty, f.step), "qty_f": qty, "step": f.step, "tick": f.tick,
               "slices": [s * f.step for s in size.slices], "entry": entry, "stop": stop, "tps": tps,
               "be": breakeven(Signal(sig.symbol, sig.interval, sig.side, sig.time, entry, stop, tps, ""),
                               self.cfg.fee),
               "trail": float(sig.exit.get("trail") or 0) / sig.price * entry,
               "risk_usdt": qty * risk, "tps_hit": 0, "extreme": None, "trail_from": None,
               "opened": int(time.time() * 1000), "close_side": sell, "tag": tag}
        try:
            pos["stop_id"] = self._place_stop(pos, stop, 0)
        except Exception as e:
            # 掛不上止損就馬上平倉，絕不留沒有止損的倉位
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
        r = self.client.conditional_order(pos["symbol"], pos["close_side"], pos["side"], "STOP_MARKET",
                                          fmt(price, pos["tick"]), client_id=f"{pos['tag']}s{version}")
        return r.get("algoId")

    def move_stop(self, pos: dict, price: float) -> None:
        """先掛新止損再取消舊的；幣安不讓同時掛兩張時，先取消再掛。"""
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
                # 價格已經越過新止損（會立即觸發）之類：直接市價平掉剩下的倉位，不留沒有止損的倉位
                amt = self.client.positions().get((pos["symbol"], pos["side"]), 0.0)
                if amt > 0:
                    self.client.market_order(pos["symbol"], pos["close_side"], pos["side"],
                                             fmt(amt, pos["step"]), f"{pos['tag']}m{version}")
                pos["stop_id"], pos["stop"], pos["stop_version"] = None, price, version
                raise RuntimeError(f"止損移不上去，已市價平倉：{e}")
        pos["stop_id"], pos["stop"], pos["stop_version"] = new, price, version

    def ensure_stop(self, pos: dict, notify) -> None:
        ids = {o.get("algoId") for o in self.client.open_conditionals(pos["symbol"])}
        if pos.get("stop_id") not in ids:
            pos["stop_version"] = pos.get("stop_version", 0) + 1
            pos["stop_id"] = self._place_stop(pos, pos["stop"], pos["stop_version"])
            notify(f"{_pair(pos['symbol'])} 止損單不見了，已補掛在 {pos['stop']:.6g}")

    def manage(self, pos: dict, now: pd.Timestamp, notify) -> float | None:
        name = f"{_pair(pos['symbol'])} {pos['interval']} {_side_text(pos['side'])}"
        live = getattr(self, "_live", None)
        amt = (live if live is not None else self.client.positions()).get((pos["symbol"], pos["side"]), 0.0)
        if amt <= 0:
            for oid in [pos.get("stop_id")] + list(pos.get("tp_ids") or []):
                if oid:
                    self.client.cancel_conditional(oid)
            return self.client.realized_since(pos["symbol"], pos["side"], pos["opened"] - 5000)
        d = 1 if pos["side"] == LONG else -1
        filled = pos["qty_f"] - amt
        hit, cum = 0, 0.0
        for q in pos["slices"][:3]:
            if filled + pos["step"] / 2 >= cum + q:
                hit += 1
                cum += q
        while pos["tps_hit"] < hit:
            n = pos["tps_hit"]
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
            self.client.market_order(pos["symbol"], pos["close_side"], pos["side"], fmt(amt, pos["step"]),
                                     f"{pos['tag']}x")
            notify(f"{name}：超過 {MAX_BARS} 根未結束，市價平倉")
        return None
