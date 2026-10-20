"""自動交易（VPS 上的 binance-trader）的日報／週報／月報資料：從網站公開的 trader.json 算出統計（只做分析）。

python -m research.trader_report --window day      # 過去 24 小時
python -m research.trader_report --window week
python -m research.trader_report --window month
python -m research.trader_report --window all --file /path/to/trader.json   # 用本機檔案

固定程式算數字，模型（日報 Haiku、週報 Sonnet、月分析 Opus）只負責讀這份輸出、寫評語，數字不讓模型自己算。
對照基準是誠實版回測（reports/signals/report_honest_backtest_20261020.md）：時點幣池、扣滑價與資金費率後，
2H+4H 合併每筆 +0.094R、勝率約 54%、多單約 +0.01R、空單約 +0.18R。
降級提醒（只通知，不自動減碼；2026-10-20 Denny 決定維加斯不設持倉上限和回撤煞車）：
實盤滿 100 筆後每筆平均比回測低 2 個標準誤以上。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import urllib.request

FEED = "https://dennyhty.github.io/grid-trading/data/trader.json"
BASELINE_R = 0.094  # 誠實版回測，扣成本後每筆
BASELINE_WIN = 0.54
STALE_MIN = 45  # 紀錄超過這麼久沒更新就提醒（trader 每個 K 線收盤才更新）
WINDOWS = {"day": 1, "week": 7, "month": 30, "all": None}


def load(path: str | None) -> dict:
    if path:
        with open(path) as f:
            return json.load(f)
    with urllib.request.urlopen(FEED, timeout=30) as r:
        return json.load(r)


def stats(rows: list[dict]) -> dict:
    """一批已平倉單的統計。r 是以風險為單位的損益。"""
    n = len(rows)
    if not n:
        return {"trades": 0}
    r = [x["r"] for x in rows]
    mean = sum(r) / n
    sd = math.sqrt(sum((v - mean) ** 2 for v in r) / (n - 1)) if n > 1 else float("nan")
    out = {"trades": n, "win_rate": sum(v > 0 for v in r) / n, "mean_r": mean, "sum_r": sum(r),
           "pnl_usdt": sum(x.get("pnl", 0.0) for x in rows), "se": sd / math.sqrt(n) if n > 1 else float("nan")}
    for side in ("LONG", "SHORT"):
        sub = [x["r"] for x in rows if x["side"] == side]
        out[side.lower()] = {"trades": len(sub), "mean_r": sum(sub) / len(sub) if sub else float("nan")}
    return out


def window_rows(ledger: list[dict], days: int | None, now_ms: float) -> list[dict]:
    if days is None:
        return list(ledger)
    return [x for x in ledger if x["closed"] >= now_ms - days * 86_400_000]


def build(feed: dict, window: str, now_ms: float | None = None) -> dict:
    now_ms = now_ms or time.time() * 1000
    days = WINDOWS[window]
    books = {}
    for name, b in feed["books"].items():
        ledger = b.get("ledger", [])
        rows = window_rows(ledger, days, now_ms)
        allst = stats(ledger)
        notes = []
        age_min = (now_ms - b.get("updated", 0)) / 60_000
        if age_min > STALE_MIN:
            notes.append(f"紀錄已 {age_min:.0f} 分鐘沒更新（trader 可能停了）")
        if b.get("paused"):
            notes.append("開新倉已暫停（pause 檔存在）")
        if allst["trades"] >= 100 and not math.isnan(allst["se"]) and allst["mean_r"] < BASELINE_R - 2 * allst["se"]:
            notes.append(f"累積 {allst['trades']} 筆每筆 {allst['mean_r']:+.3f}R，比回測 {BASELINE_R:+.3f}R 低超過 2 個標準誤（降級提醒，只通知）")
        books[name] = {"window": stats(rows), "all_time": allst, "open_positions": len(b.get("positions", [])),
                       "open": [{k: p.get(k) for k in ("symbol", "interval", "side", "entry", "stop", "tps_hit", "unrealized_r", "unrealized_pnl") if k in p}
                                for p in b.get("positions", [])],
                       "notes": notes, "processed": b.get("processed")}
    return {"window": window, "generated": feed.get("generated"), "baseline": {"mean_r": BASELINE_R, "win_rate": BASELINE_WIN}, "books": books}


def to_md(rep: dict) -> str:
    label = {"day": "過去 24 小時", "week": "過去 7 天", "month": "過去 30 天", "all": "全部"}[rep["window"]]
    lines = [f"# 自動交易報告（{label}）", "", f"回測基準：每筆 {rep['baseline']['mean_r']:+.3f}R、勝率約 {rep['baseline']['win_rate']:.0%}（誠實版回測，扣成本）", ""]
    for name, b in rep["books"].items():
        w, a = b["window"], b["all_time"]
        lines.append(f"## {name}")
        if w["trades"]:
            lines.append(f"- 期間：{w['trades']} 筆已平倉，勝率 {w['win_rate']:.0%}，每筆 {w['mean_r']:+.3f}R，合計 {w['sum_r']:+.2f}R（{w['pnl_usdt']:+.2f} USDT）；"
                         f"多 {w['long']['trades']} 筆、空 {w['short']['trades']} 筆")
        else:
            lines.append("- 期間：沒有已平倉的單")
        if a["trades"]:
            se = f"，標準誤 {a['se']:.3f}" if not math.isnan(a["se"]) else ""
            lines.append(f"- 累積：{a['trades']} 筆，勝率 {a['win_rate']:.0%}，每筆 {a['mean_r']:+.3f}R{se}，合計 {a['sum_r']:+.2f}R")
        lines.append(f"- 目前持倉 {b['open_positions']} 張")
        for n in b["notes"]:
            lines.append(f"- ⚠ {n}")
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--window", choices=list(WINDOWS), default="day")
    ap.add_argument("--file", help="本機的 trader.json（不給就抓網站上的）")
    ap.add_argument("--json", action="store_true", help="輸出 JSON 而不是文字")
    a = ap.parse_args(argv)
    rep = build(load(a.file), a.window)
    print(json.dumps(rep, ensure_ascii=False, indent=1) if a.json else to_md(rep))
    return 0


if __name__ == "__main__":
    sys.exit(main())
