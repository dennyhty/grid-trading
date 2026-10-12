"""即時掃描：定時抓最新 K 線跑策略，追蹤每個訊號的 TP / 止損，並把新事件寫進通知中心。"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pandas as pd

from .. import indicators as ind
from ..pionex import PionexPublic
from .strategy import Strategy
from .tracker import track

INTERVAL_MS = {"15M": 900_000, "30M": 1_800_000, "60M": 3_600_000, "4H": 14_400_000, "8H": 28_800_000,
               "12H": 43_200_000, "1D": 86_400_000}
LABEL = {"60M": "1H"}


def label(interval: str) -> str:
    return LABEL.get(interval, interval)


def fetch_klines(api: PionexPublic, symbol: str, interval: str, bars: int = 1000) -> pd.DataFrame:
    """抓最近 bars 根 K 線（每次最多 500，往回翻頁）。最後一根可能還沒收盤。"""
    frames = []
    end = None
    while bars > 0:
        params = {"symbol": symbol, "interval": interval, "limit": min(500, bars)}
        if end is not None:
            params["endTime"] = end
        try:
            data = api._get("/api/v1/market/klines", params)
        except Exception:
            break
        batch = pd.DataFrame(data.get("klines", []))
        if batch.empty:
            break
        frames.append(batch)
        bars -= len(batch)
        if len(batch) < 500:
            break
        end = int(batch["time"].min()) - 1
    if not frames:
        return pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])
    df = pd.concat(frames).drop_duplicates("time")
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True)
    return df.sort_values("time").reset_index(drop=True)


class Engine:
    def __init__(self, strategies: list[Strategy], symbols: list[str], intervals: list[str],
                 keep_days: float = 7, fee: float = 0.0005, state_dir: Path = Path("output"),
                 stats_path: Path = Path("stats/signal_stats.json")):
        self.api = PionexPublic(pause=0.1)
        self.strategies = {s.name: s for s in strategies}
        self.symbols = symbols
        self.intervals = intervals
        self.keep = pd.Timedelta(days=keep_days)
        self.fee = fee
        self.state_dir = state_dir
        self.stats_path = stats_path
        self.lock = threading.Lock()
        self.signals: dict[str, dict] = {}
        self.notifications: list[dict] = self._load_notifications()
        self.updated: int | None = None
        self._prices: dict[str, float] = {}
        self._prices_at = 0.0

    # ---------- 行情 ----------
    def prices(self) -> dict[str, float]:
        """所有永續合約最新價，快取 10 秒。"""
        if time.time() - self._prices_at > 10:
            try:
                t = self.api.perp_tickers()
                self._prices = dict(zip(t["symbol"], t["close"].astype(float)))
                self._prices_at = time.time()
            except Exception:
                pass
        return self._prices

    def chart(self, symbol: str, interval: str, strategy: str | None = None, bars: int = 1000) -> dict:
        df = fetch_klines(self.api, symbol, interval, bars)
        strat = self.strategies.get(strategy or "") or next(iter(self.strategies.values()))
        lines = {}
        for name, n in strat.chart_emas.items():
            e = ind.ema(df["close"], n)
            lines[name] = [{"time": int(t.timestamp()), "value": float(v)}
                           for t, v in zip(df["time"], e) if pd.notna(v)]
        rsi = ind.rsi(df["close"])
        macd = ind.macd(df["close"])
        candles = [{"time": int(t.timestamp()), "open": o, "high": h, "low": lo, "close": c}
                   for t, o, h, lo, c in zip(df["time"], df["open"], df["high"], df["low"], df["close"])]
        last = {}
        if len(df):
            last = {"time": int(df["time"].iloc[-1].timestamp() * 1000), "open": float(df["open"].iloc[-1]),
                    "high": float(df["high"].iloc[-1]), "low": float(df["low"].iloc[-1]),
                    "close": float(df["close"].iloc[-1]), "rsi": float(rsi.iloc[-1]),
                    "macd": float(macd["macd"].iloc[-1]), "macd_signal": float(macd["signal"].iloc[-1]),
                    "hist": float(macd["hist"].iloc[-1])}
        return {"symbol": symbol, "interval": interval, "label": label(interval), "candles": candles,
                "emas": lines, "last": last}

    # ---------- 掃描 ----------
    def scan_once(self) -> None:
        now = pd.Timestamp.now(tz="UTC")
        found: dict[str, dict] = {}
        for interval in self.intervals:
            bar = pd.Timedelta(milliseconds=INTERVAL_MS[interval])
            for sym in self.symbols:
                df = fetch_klines(self.api, sym, interval, 1000)
                if len(df) < 250:
                    continue
                closed = df[df["time"] + bar <= now]
                start = max(0, len(closed) - int(self.keep / bar) - 1)
                for strat in self.strategies.values():
                    busy_until = None
                    for sig in strat.scan(closed, sym, interval, start=start):
                        if busy_until is not None and sig.time < busy_until:
                            continue
                        res = track(sig, df, self.fee)  # 含尚未收盤的那根，盤中觸及也算
                        if res.status == "CLOSED":
                            busy_until = res.events[-1].time
                        else:
                            busy_until = now + bar
                        sid = f"{sym}|{interval}|{strat.name}|{int(sig.time.timestamp())}"
                        found[sid] = {"id": sid, "label": label(interval), **sig.to_dict(),
                                      "track": res.to_dict()}
        with self.lock:
            self._notify(found)
            self.signals = found
            self.updated = int(time.time() * 1000)

    def _notify(self, found: dict[str, dict]) -> None:
        first_run = not self.signals and not self.notifications
        for sid, s in found.items():
            old = self.signals.get(sid)
            pair = s["symbol"].replace("_USDT_PERP", "/USDT")
            side = "做多" if s["side"] == "LONG" else "做空"
            if old is None and not any(n["signal_id"] == sid and n["kind"] == "NEW" for n in self.notifications):
                self._push(sid, "NEW", s["time"], f"{pair} {s['label']} {side}訊號 @ {s['price']:.6g}", first_run)
            seen = len(old["track"]["events"]) if old else self._seen_events(sid)
            for e in s["track"]["events"][seen:]:
                self._push(sid, e["kind"], e["time"], f"{pair} {s['label']} {side}：{e['text']}", first_run)
        self.notifications = self.notifications[-300:]
        self._save_notifications()

    def _seen_events(self, sid: str) -> int:
        return sum(1 for n in self.notifications if n["signal_id"] == sid and n["kind"] != "NEW")

    def _push(self, sid: str, kind: str, t: int, text: str, quiet: bool) -> None:
        self.notifications.append({"id": f"{sid}|{kind}|{t}", "signal_id": sid, "kind": kind, "time": t,
                                   "created": int(time.time() * 1000), "text": text, "initial": quiet})

    def _load_notifications(self) -> list[dict]:
        p = self.state_dir / "notifications.json"
        try:
            return json.loads(p.read_text())
        except Exception:
            return []

    def _save_notifications(self) -> None:
        self.state_dir.mkdir(exist_ok=True)
        (self.state_dir / "notifications.json").write_text(json.dumps(self.notifications, ensure_ascii=False))

    def run_forever(self, every: float = 120) -> None:
        while True:
            time.sleep(every)
            try:
                self.scan_once()
            except Exception as e:  # 網路問題不讓背景執行緒死掉
                print("掃描失敗:", e)

    # ---------- 給網頁的資料 ----------
    def state(self) -> dict:
        prices = self.prices()
        try:
            stats = json.loads(self.stats_path.read_text())
        except Exception:
            stats = {}
        with self.lock:
            sigs = sorted(self.signals.values(), key=lambda s: s["time"], reverse=True)
            notes = list(reversed(self.notifications[-100:]))
        for s in sigs:
            s["live"] = prices.get(s["symbol"])
        strategies = {n: {"title": s.title, "description": s.description} for n, s in self.strategies.items()}
        return {"signals": sigs, "notifications": notes, "stats": stats, "strategies": strategies,
                "updated": self.updated, "intervals": [label(i) for i in self.intervals],
                "symbols": len(self.symbols), "fee": self.fee}
