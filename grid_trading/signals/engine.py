"""即時掃描：定時抓最新 K 線跑策略，追蹤每個訊號的 TP / 止損，並把新事件寫進通知中心。"""

from __future__ import annotations

import json
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from .. import indicators as ind
from ..history import resample
from ..markets import LABELS as MARKET_LABELS, classify, display_name
from ..pionex import PionexPublic
from .strategy import Signal, Strategy
from .tracker import track

INTERVAL_MS = {"15M": 900_000, "30M": 1_800_000, "60M": 3_600_000, "2H": 7_200_000, "4H": 14_400_000,
               "8H": 28_800_000, "12H": 43_200_000, "1D": 86_400_000}
LABEL = {"60M": "1H"}
# 派網沒有的週期，用較小的週期合成：週期 → (來源週期, 幾根合一根)。分組從 00:00 UTC 起算，和幣安的 2h K 線一致。
SYNTHETIC = {"2H": ("60M", 2)}
# 掃描時每個週期抓幾根。派網對同一個 IP 有請求量限制（超過回 429），2H 由 1H 合成、每根要 2 根 1H，
# 所以只抓 500 根（1000 根 1H，2 次請求）；EMA169 用 500 根暖身已經夠準
SCAN_BARS = {"2H": 500}


def label(interval: str) -> str:
    return LABEL.get(interval, interval)


def _get_retry(api: PionexPublic, path: str, params: dict, tries: int = 4) -> tuple[dict | None, str]:
    """派網偶爾回錯（例如請求太密被限流），等一下再試；試完還是失敗就回傳 (None, 錯誤訊息)。"""
    err = ""
    for i in range(tries):
        try:
            return api._get(path, params), ""
        except Exception as e:
            err = f"{type(e).__name__}: {str(e)[:120]}"
            if i < tries - 1:
                time.sleep(1.5 * 2 ** i)
    return None, err


def fetch_klines(api: PionexPublic, symbol: str, interval: str, bars: int = 1000) -> pd.DataFrame:
    """抓最近 bars 根 K 線（每次最多 500，往回翻頁）。最後一根可能還沒收盤。

    重試後還是抓不到時，回傳已抓到的部分，並在 df.attrs["incomplete"] 標記 True，
    讓呼叫的人分得出「這個幣歷史本來就短」和「這次沒抓到」。
    """
    if interval in SYNTHETIC:
        base, k = SYNTHETIC[interval]
        raw = fetch_klines(api, symbol, base, bars * k)
        df = resample(raw, f"{INTERVAL_MS[interval] // 3_600_000}h", origin="epoch") if len(raw) else raw
        df.attrs["incomplete"] = raw.attrs.get("incomplete", False)
        df.attrs["error"] = raw.attrs.get("error", "")
        return df
    frames = []
    end = None
    incomplete, error = False, ""
    while bars > 0:
        params = {"symbol": symbol, "interval": interval, "limit": min(500, bars)}
        if end is not None:
            params["endTime"] = end
        data, error = _get_retry(api, "/api/v1/market/klines", params)
        if data is None:
            incomplete = True
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
        df = pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])
        df.attrs["incomplete"], df.attrs["error"] = incomplete, error
        return df
    df = pd.concat(frames).drop_duplicates("time")
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True)
    df = df.sort_values("time").reset_index(drop=True)
    df.attrs["incomplete"], df.attrs["error"] = incomplete, error
    return df


class Engine:
    def __init__(self, strategies: list[Strategy], symbols: list[str], intervals: list[str],
                 keep_days: float = 7, fee: float = 0.0005, state_dir: Path = Path("output"),
                 stats_path: Path = Path("stats/signal_stats.json"), workers: int = 2):
        self.api = PionexPublic(pause=0.1)
        self.workers = workers  # 同時抓幾個 K 線（每條執行緒各用自己的連線）
        self._local = threading.local()
        self.failed: list[tuple[str, str]] = []
        self.errors: dict[tuple[str, str], str] = {}
        # 上一次掃描的訊號（網頁版從目前網站的 state.json 讀進來）。某個（幣, 週期）這次 K 線沒抓到時，
        # 沿用上一次的訊號，網站上的單才不會整批消失
        self.previous: dict[str, dict] = {}
        self.carried = 0
        self.frames: dict[tuple[str, str], pd.DataFrame] = {}  # 這次掃描抓到的 K 線，畫圖和實盤紀錄直接沿用
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

    def client(self) -> PionexPublic:
        """這條執行緒自己的 API 連線。"""
        if not hasattr(self._local, "api"):
            self._local.api = PionexPublic(pause=0.1)
        return self._local.api

    def klines_many(self, jobs: list[tuple[str, str]], bars: int = 1000) -> dict[tuple[str, str], pd.DataFrame]:
        """同時抓多個（symbol, interval）的 K 線；SCAN_BARS 裡有的週期改用那個根數。"""
        with ThreadPoolExecutor(self.workers) as ex:
            return dict(zip(jobs, ex.map(
                lambda j: fetch_klines(self.client(), j[0], j[1], SCAN_BARS.get(j[1], bars)), jobs)))

    def klines(self, symbol: str, interval: str, bars: int = 1000) -> pd.DataFrame:
        """先用這次掃描抓到的 K 線（省一次請求），沒有才去抓。"""
        df = self.frames.get((symbol, interval))
        if df is not None and len(df) and not df.attrs.get("incomplete"):
            return df.tail(bars).reset_index(drop=True)
        return fetch_klines(self.client(), symbol, interval, bars)

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
        df = self.klines(symbol, interval, bars)
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
        frames = self.klines_many([(sym, iv) for iv in self.intervals for sym in self.symbols])
        self.frames = frames
        # 重試後還是沒抓完整的（幣, 週期）：這些沿用上一次的訊號
        self.failed = sorted(k for k, df in frames.items() if df.attrs.get("incomplete"))
        self.errors = {k: frames[k].attrs.get("error", "") for k in self.failed}
        failed = set(self.failed)
        self.carried = 0
        # 還沒結束的單（含已經超過 keep_days、離開掃描範圍的）：這次照樣追蹤到結束為止，
        # 網站上的持倉才不會在第 7 天消失，之後的 TP / 止損也照樣推播
        lingering: dict[tuple, list[dict]] = {}
        for sid, old in (self.previous or self.signals).items():
            is_open = (old.get("track") or {}).get("status") == "OPEN"
            if (old["symbol"], old["interval"]) in failed:
                if is_open or now - pd.Timestamp(old["time"], unit="ms", tz="UTC") <= self.keep:
                    found[sid] = old
                    self.carried += 1
            elif is_open:
                lingering.setdefault((old["symbol"], old["interval"], old.get("strategy")), []).append(old)
        for interval in self.intervals:
            bar = pd.Timedelta(milliseconds=INTERVAL_MS[interval])
            for sym in self.symbols:
                df = frames[(sym, interval)]
                if len(df) < 250 or (sym, interval) in failed:
                    continue
                closed = df[df["time"] + bar <= now]
                start = max(0, len(closed) - int(self.keep / bar) - 1)
                # 這次掃描最早能找到的訊號時間（訊號時間 = 訊號 K 線的收盤時間）
                window = closed["time"].iloc[start] + bar if len(closed) else now
                for strat in self.strategies.values():
                    if (strat.intervals and interval not in strat.intervals) or \
                            (strat.markets and classify(sym) not in strat.markets):
                        continue
                    scanned = {f"{sym}|{interval}|{strat.name}|{int(sig.time.timestamp())}": sig
                               for sig in strat.scan(closed, sym, interval, start=start)}
                    # 已經發出、還沒結束的單優先：照樣追蹤到結束，期間同一個幣不出新訊號。
                    # 進場規則改了以後（例如 2H 加上 ADX 過濾），重新掃描不會再出現的舊單也照樣追蹤，網站上的持倉不會消失
                    held: list[tuple[pd.Timestamp, pd.Timestamp]] = []
                    for old in sorted(lingering.get((sym, interval, strat.name), []), key=lambda o: o["time"]):
                        t0 = pd.Timestamp(old["time"], unit="ms", tz="UTC")
                        sig = scanned.pop(old["id"], None)
                        if sig is not None:  # 還在掃描範圍內，用這次掃到的（規則若改了，用新的出場方式）
                            res = track(sig, df, self.fee)
                            found[old["id"]] = self._entry(sig, res)
                        elif df["time"].iloc[0] > t0:
                            found[old["id"]] = old  # K 線不夠長，沒辦法重新追蹤，先沿用
                            held.append((t0, now + bar))
                            continue
                        else:
                            sig = Signal(sym, interval, old["side"], t0, old["price"], old["sl"], list(old["tps"]),
                                         strat.name, old.get("info") or {}, old.get("exit") or {})
                            res = track(sig, df, self.fee)
                            found[old["id"]] = {**old, "track": res.to_dict()}
                        held.append((t0, res.events[-1].time if res.status == "CLOSED" else now + bar))
                    busy_until = None
                    for sid, sig in scanned.items():
                        if (busy_until is not None and sig.time < busy_until) or \
                                any(t0 <= sig.time < t1 for t0, t1 in held):
                            continue
                        res = track(sig, df, self.fee)  # 含尚未收盤的那根，盤中觸及也算
                        end = res.events[-1].time if res.status == "CLOSED" else now + bar
                        if any(sig.time < t0 < end for t0, _ in held):
                            continue  # 會和已經發出的單重疊
                        busy_until = end
                        found[sid] = self._entry(sig, res)
        with self.lock:
            self._notify(found)
            self.signals = found
            self.updated = int(time.time() * 1000)

    @staticmethod
    def _entry(sig: Signal, res) -> dict:
        sid = f"{sig.symbol}|{sig.interval}|{sig.strategy}|{int(sig.time.timestamp())}"
        return {"id": sid, "label": label(sig.interval), "market": classify(sig.symbol),
                "name": display_name(sig.symbol), **sig.to_dict(), "track": res.to_dict()}

    def _notify(self, found: dict[str, dict]) -> None:
        first_run = not self.signals and not self.notifications
        now = time.time() * 1000
        # 已經寫過的通知（同一個 id 可能出現兩次：TP1、TP2 在同一根 K 線觸及時，會有兩個同時間的 STOP_MOVED）
        have = Counter(n["id"] for n in self.notifications)
        latest: dict[str, int] = {}  # 每個訊號已通知過的最新事件時間
        for n in self.notifications:
            if n["kind"] != "NEW":
                latest[n["signal_id"]] = max(latest.get(n["signal_id"], 0), n["time"])
        for sid, s in found.items():
            old = self.signals.get(sid)
            pair = s["name"] if s["market"] != "crypto" else s["symbol"].replace("_USDT_PERP", "/USDT")
            side = "做多" if s["side"] == "LONG" else "做空"
            strat = self.strategies.get(s.get("strategy", ""))
            if strat is not None and strat.tag:
                pair = f"［{strat.tag}］{pair}"
            quiet = first_run
            if old is None and not any(n["signal_id"] == sid and n["kind"] == "NEW" for n in self.notifications):
                # 第一次出現時已經是 3 根 K 線以前的訊號（例如剛加上的新週期補掃到的舊訊號）：記下來但不推播
                quiet = quiet or now - s["time"] > 3 * INTERVAL_MS[s["interval"]]
                self._push(sid, "NEW", s["time"], f"{pair} {s['label']} {side}訊號 @ {s['price']:.6g}", quiet)
            # 用通知 id 比對哪些事件已經通知過，不用「第幾個事件」：出場規則改了、事件清單跟著變時，
            # 用數量比對會漏推或推錯
            count: Counter = Counter()
            for e in s["track"]["events"]:
                nid = f"{sid}|{e['kind']}|{e['time']}"
                count[nid] += 1
                if count[nid] > have[nid]:
                    have[nid] += 1
                    # 比已通知的事件還早：是出場規則改了、重新追蹤補出來的舊事件，只記錄不推播
                    late = e["time"] < latest.get(sid, 0)
                    self._push(sid, e["kind"], e["time"], f"{pair} {s['label']} {side}：{e['text']}", quiet or late)
        # 只保留最近 300 則，但網站上還在的訊號，它的通知全部留著（不然下次會當成沒通知過）
        cut = len(self.notifications) - 300
        self.notifications = [n for i, n in enumerate(self.notifications) if i >= cut or n["signal_id"] in found]
        self._save_notifications()

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
        strategies = {n: {"title": s.title, "description": s.description, "tag": s.tag,
                          "intervals": [label(i) for i in s.intervals] if s.intervals else None,
                          "markets": list(s.markets) if s.markets else None}
                      for n, s in self.strategies.items()}
        return {"signals": sigs, "notifications": notes, "stats": stats, "strategies": strategies,
                "updated": self.updated, "intervals": [label(i) for i in self.intervals],
                "symbols": len(self.symbols), "fee": self.fee,
                "markets": {m: {"label": lbl, "symbols": sum(classify(s) == m for s in self.symbols)}
                            for m, lbl in MARKET_LABELS.items()}}
