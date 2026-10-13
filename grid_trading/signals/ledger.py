"""實盤訊號紀錄：每個訊號從出現到結束的結果，用來累積實盤成績（例如 RSI 收緊觀察組）。

網頁只顯示最近幾天的訊號，但一張單可能拖到 40 天才結束，所以紀錄另外存一份
（GitHub Pages 版存在網站的 data/ledger.json，每次執行接著用）。還沒結束的單，
離開掃描範圍後會用最新 K 線重新追蹤，直到結束為止。
"""

from __future__ import annotations

from typing import Callable

import pandas as pd

from .strategy import Signal
from .tracker import track

FIELDS = ("id", "symbol", "interval", "strategy", "market", "side", "time", "price", "sl", "tps")


def entry_from(sig: dict) -> dict:
    t = sig["track"]
    e = {k: sig[k] for k in FIELDS}
    if sig.get("exit"):
        e["exit_plan"] = sig["exit"]  # 出場方式（2026-10-13 起 4H/2H 不同）；舊紀錄沒有就是各平 1/3
    e["strict_rsi"] = (sig.get("info") or {}).get("strict_rsi")
    e.update(_result(t))
    return e


def _result(t: dict) -> dict:
    events = t.get("events") or []
    closed = t["status"] == "CLOSED"
    return {"status": t["status"], "r": t["realized_r"] if closed else None, "tps_hit": t["tps_hit"],
            "exit": events[-1]["kind"] if closed and events else None,
            "exit_time": events[-1]["time"] if closed and events else None}


def update(ledger: list[dict], signals: list[dict], candles: Callable[[str, str], pd.DataFrame],
           fee: float = 0.0005) -> list[dict]:
    """把這次掃到的訊號寫進紀錄；不在這次掃描裡、還沒結束的單用 candles(symbol, interval) 重新追蹤。"""
    book = {e["id"]: e for e in ledger}
    seen = set()
    for s in signals:
        old = book.get(s["id"])
        if old and old["status"] == "CLOSED":
            continue  # 已結束的不再改
        book[s["id"]] = entry_from(s)
        seen.add(s["id"])
    for e in book.values():
        if e["status"] == "CLOSED" or e["id"] in seen:
            continue
        df = candles(e["symbol"], e["interval"])
        if df is None or df.empty:
            continue
        sig = Signal(e["symbol"], e["interval"], e["side"], pd.Timestamp(e["time"], unit="ms", tz="UTC"),
                     e["price"], e["sl"], list(e["tps"]), e["strategy"], exit=e.get("exit_plan") or {})
        e.update(_result(track(sig, df, fee).to_dict()))
    return sorted(book.values(), key=lambda e: e["time"])


def summary(ledger: list[dict]) -> dict:
    """groups[「策略|類別|週期」]：全部、收緊組也會進、收緊組會剔除，各自已結束的筆數、勝率、平均 R，和還沒結束的筆數。"""
    out: dict[str, dict] = {}
    for e in ledger:
        key = f"{e['strategy']}|{e['market']}|{e['interval']}"
        g = out.setdefault(key, {})
        groups = ["all"]
        if e.get("strict_rsi") is not None:
            groups.append("strict" if e["strict_rsi"] else "excluded")
        for name in groups:
            b = g.setdefault(name, {"trades": 0, "wins": 0, "sum_r": 0.0, "open": 0})
            if e["status"] == "CLOSED":
                b["trades"] += 1
                b["wins"] += e["r"] > 0
                b["sum_r"] += e["r"]
            else:
                b["open"] += 1
    for g in out.values():
        for b in g.values():
            n = b["trades"]
            b["win_rate"] = b["wins"] / n if n else None
            b["expectancy_r"] = b["sum_r"] / n if n else None
    return {"since": min((e["time"] for e in ledger), default=None), "groups": out}
