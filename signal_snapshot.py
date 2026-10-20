"""產生靜態網站用的資料快照：python signal_snapshot.py --out site

給 GitHub Pages 用。GitHub Actions 每 15 分鐘跑一次：用幣安 USDT 永續合約的 K 線掃描實盤維加斯 4H / 2H、
寫出 JSON，網頁（web/）直接讀這些 JSON，不需要一直開著的伺服器。2026-10-20 起只用幣安（和 VPS 上的自動交易同一份合約名單），
派網、觀察組、美股/商品都不再掃描。

輸出：
  <out>/data/state.json                    訊號、通知、回測數字（同 /api/state）
  <out>/data/notifications.json            通知紀錄，下一次執行接著用
  <out>/data/ledger.json                   每個訊號的結果（實盤紀錄），下一次執行接著用
  <out>/data/chart/<symbol>_<interval>.json  圖表資料（同 /api/chart）
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from grid_trading import signals
import signal_telegram as telegram
from grid_trading.binance_history import binance_symbols
from grid_trading.binance_public import from_binance, universe
from grid_trading.markets import CRYPTO, classify
from grid_trading.signals import ledger as ledger_mod
from grid_trading.signals.engine import INTERVAL_MS, Engine
from signal_server import _sanitize

CHART_INTERVALS = ["15M", "60M", "4H", "1D"]


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_sanitize(obj), ensure_ascii=False, separators=(",", ":"), allow_nan=False))


def closing_intervals(intervals: list[str], now_ms: int, within_ms: int = 10 * 60_000) -> list[str]:
    """最近 within_ms 內剛收盤的週期（例如 UTC 08:00 是 4H 和 2H，10:00 只有 2H）。"""
    return [iv for iv in intervals if now_ms % INTERVAL_MS[iv] < within_ms]


# 2026-10-18 以前沒有「上一次掃到哪根 K 線」的紀錄，EMA 微幅漂移讓幾天前的訊號事後冒出來、
# 被當成新單記下（只記不推播）。第一次用新紀錄時，把這段期間這樣冒出來的實盤訊號拿掉
LATE_SINCE_MS = 1_792_108_800_000  # 2026-10-16 00:00 UTC：在這之後沒有新加的週期或策略要補掃


def retract_late(prev_state: dict, notifications: list[dict], ledger: list[dict]) -> set[str]:
    """上一次的 state.json 還沒有 scanned（這個功能上線後第一次）時，找出事後才冒出來的實盤訊號：
    第一次出現就已經晚了 3 根 K 線以上、只記不推播的 NEW。回傳它們的 id，並從通知和實盤紀錄裡拿掉。"""
    if "scanned" in prev_state or not prev_state.get("signals"):
        return set()
    late = set()
    for n in notifications:
        iv = n["signal_id"].split("|")[1] if n["signal_id"].count("|") >= 3 else ""
        if (n["kind"] == "NEW" and n.get("initial") and not n.get("observe") and iv in INTERVAL_MS
                and n["created"] >= LATE_SINCE_MS and n["created"] - n["time"] > 3 * INTERVAL_MS[iv]):
            late.add(n["signal_id"])
    notifications[:] = [n for n in notifications if n["signal_id"] not in late]
    ledger[:] = [e for e in ledger if e["id"] not in late]
    return late


def early_push(args, symbols: list[str], data: Path, previous: dict, blocked: dict, sent_path: Path,
               scanned: dict) -> dict:
    """2H/4H 剛收盤時先只掃加密貨幣的實盤策略（沒有 tag 的），新通知馬上送 Telegram，不等整個網站產生和部署完。

    抓到的 K 線回傳給完整掃描沿用，不多打幣安。通知 id 和完整掃描算出來的一樣，所以送過的寫到 sent_path，
    之後的 outbox 跳過。萬一這一輪網站沒部署成功，下一輪會再送一次（寧可重複，不要漏）。"""
    sent_path.write_text("[]")
    live = [n for n in args.strategies if not signals.get(n).tag]
    # 只看實盤策略會用到的週期：1H 只有觀察組，整點不用搶先掃
    used = {iv for n in live for iv in (signals.get(n).intervals or args.intervals)}
    closing = closing_intervals([iv for iv in args.intervals if iv in used], int(time.time() * 1000))
    crypto = [s for s in symbols if classify(s) == CRYPTO]
    if not closing or not live or not crypto:
        return {}
    t0 = time.time()
    tmp = Path(tempfile.mkdtemp())
    if (data / "notifications.json").exists():
        shutil.copy(data / "notifications.json", tmp / "notifications.json")
    quick = Engine([signals.get(n) for n in live], crypto, closing, args.keep_days, state_dir=tmp)
    if not quick.notifications:
        return {}  # 沒有上一次的通知紀錄：交給完整掃描（它會當成第一次啟動、不推播）
    seen = {n["id"] for n in quick.notifications}
    # 沒收盤的週期也一起帶進去：「同幣同方向只做一張」要看另一個週期還開著的單
    quick.previous = {k: s for k, s in previous.items()
                      if s.get("symbol") in set(crypto) and s.get("strategy") in live}
    quick.blocked = dict(blocked)
    quick.scanned = dict(scanned)
    quick.scan_once()
    state = quick.state()
    msgs = telegram.messages(quick.notifications, seen, state["signals"], state["strategies"])
    token, chat_id = os.environ.get("TELEGRAM_BOT_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID")
    if msgs and token and chat_id:
        sent, failed = telegram.send([m for _, m in msgs], token, chat_id)
        if not failed:  # 有失敗就全部交給之後的 outbox 重送
            sent_path.write_text(json.dumps([i for i, _ in msgs]))
        print(f"收盤搶先推播（{'/'.join(closing)}）：{sent} 則，失敗 {failed} 則，"
              f"收盤後 {int(time.time() * 1000) % INTERVAL_MS[closing[-1]] / 1000:.0f} 秒", file=sys.stderr)
    print(f"收盤搶先掃描 {len(crypto)} 個幣 {closing}，{time.time() - t0:.0f} 秒", file=sys.stderr)
    return quick.frames


def main() -> int:
    p = argparse.ArgumentParser(description="產生訊號網頁的靜態資料")
    p.add_argument("--out", default="site")
    p.add_argument("--web", default="web", help="網頁原始檔，會複製到 --out")
    p.add_argument("--previous", help="上一次的 notifications.json，用來接續通知紀錄")
    p.add_argument("--ledger", help="上一次的 ledger.json（實盤紀錄），沒有就從頭開始")
    p.add_argument("--strategies", nargs="+", default=["vegas_trend"], choices=sorted(signals.REGISTRY))
    p.add_argument("--intervals", nargs="+", default=["4H", "2H"])
    p.add_argument("--keep-days", type=float, default=7)
    p.add_argument("--previous-state", help="上一次的 state.json；這次 K 線沒抓到的（幣, 週期）沿用裡面的訊號")
    p.add_argument("--early-push", metavar="SENT_JSON",
                   help="2H/4H 剛收盤時，先只掃加密貨幣的實盤策略、馬上把新通知送到 Telegram（要有 TELEGRAM_BOT_TOKEN / "
                        "TELEGRAM_CHAT_ID），送過的通知 id 寫到這個檔，之後的 outbox 會跳過")
    p.add_argument("--max-failed", type=float, default=0.5,
                   help="K 線抓不到的（幣, 週期）超過這個比例（大概是幣安整個連不上）就不輸出，網站保留上一版")
    args = p.parse_args()

    out = Path(args.out)
    data = out / "data"
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(args.web, out)
    # 策略研究頁讀的報告原文（reports/*.md）
    if Path("reports").is_dir():
        shutil.copytree("reports", out / "reports")
    # 每次部署給 app.js、style.css 加上版本號，瀏覽器才不會拿舊的頁面配新的程式（或反過來）
    # 版本號用網頁原始檔的內容算，網頁沒改就不變（一直開著的分頁只在真的改版時才重新載入）
    digest = hashlib.sha256()
    for f in sorted(Path(args.web).rglob("*")):
        if f.is_file():
            digest.update(str(f.relative_to(args.web)).encode() + f.read_bytes())
    version = digest.hexdigest()[:12]
    index = out / "index.html"
    index.write_text(index.read_text().replace('src="app.js"', f'src="app.js?v={version}"')
                     .replace('src="perf.js"', f'src="perf.js?v={version}"')
                     .replace('href="style.css"', f'href="style.css?v={version}"'))
    data.mkdir(parents=True)
    # 一直開著的分頁會用 data/version.json 發現網站改版，自己重新載入，不會繼續跑舊程式
    (data / "version.json").write_text(json.dumps({"version": version}))
    (out / ".nojekyll").write_text("")
    try:
        prev_state = json.loads(Path(args.previous_state).read_text()) if args.previous_state else {}
    except Exception:
        prev_state = {}
    previous = json.loads(Path(args.ledger).read_text()) if args.ledger and Path(args.ledger).exists() else []
    late: set[str] = set()
    if args.previous and Path(args.previous).exists():
        notes = json.loads(Path(args.previous).read_text())
        late = retract_late(prev_state, notes, previous)
        if late:
            print(f"拿掉事後才冒出來的訊號：{sorted(late)}", file=sys.stderr)
        (data / "notifications.json").write_text(json.dumps(notes, ensure_ascii=False))

    symbols = universe()
    engine = Engine([signals.get(n) for n in args.strategies], symbols, args.intervals, args.keep_days, state_dir=data)
    engine.previous = {s["id"]: s for s in prev_state.get("signals", []) if "id" in s and s["id"] not in late}
    engine.blocked = dict(prev_state.get("blocked") or {})
    engine.scanned = dict(prev_state.get("scanned") or {})
    # 派網時代的代號換成幣安合約名稱的（PEPE → 1000PEPE、SHIB → 1000SHIB）：沿用掃到哪根 K 線的紀錄，
    # 舊 K 線上的訊號才不會被當成新訊號
    for key, t in list(engine.scanned.items()):
        sym, iv = key.split("|", 1)
        new = from_binance(binance_symbols(sym.split("_")[0])[-1])
        if new != sym and new in symbols:
            engine.scanned.setdefault(f"{new}|{iv}", t)
    if not engine.scanned and prev_state.get("updated"):
        # 上一次的 state.json 還沒有 scanned：當成上一次每個幣、週期都掃到它更新的時間
        engine.scanned = {f"{s}|{iv}": prev_state["updated"] for s in symbols for iv in args.intervals}
    if args.early_push:
        engine.prefetched = early_push(args, symbols, data, engine.previous, engine.blocked, Path(args.early_push),
                                       engine.scanned)
    engine.scan_once()
    if engine.failed:
        sample = sorted(set(engine.errors.values()))[:3]
        print(f"注意：{len(engine.failed)} 組 K 線重試後仍抓不到，沿用上一次的 {engine.carried} 個訊號。"
              f"例如 {engine.failed[:5]}；錯誤 {sample}", file=sys.stderr)
    if len(engine.failed) > args.max_failed * len(symbols) * len(args.intervals):
        print("抓不到的太多（大概是幣安連不上），這次不更新網站", file=sys.stderr)
        return 1
    state = engine.state()
    state["static"] = True
    state["source"] = "binance"

    book = ledger_mod.update(previous, state["signals"], lambda sym, iv: engine.klines(sym, iv, 1000),
                             engine.fee)
    write_json(data / "ledger.json", book)
    state["ledger"] = ledger_mod.summary(book)
    write_json(data / "state.json", state)

    pairs = {(s["symbol"], s["interval"], s["strategy"]) for s in state["signals"]}
    charts: dict[tuple[str, str], tuple[str, int]] = {}  # (幣, 週期) → (策略, 根數)；訊號本身的週期多抓一些
    for sym, sig_iv, strat in sorted(pairs):
        for iv in dict.fromkeys([sig_iv, *CHART_INTERVALS]):
            bars = 1000 if iv == sig_iv else 500
            if charts.get((sym, iv), ("", 0))[1] < bars:
                charts[(sym, iv)] = (strat, bars)

    def write_chart(job):
        (sym, iv), (strat, bars) = job
        write_json(data / "chart" / f"{sym}_{iv}.json", engine.chart(sym, iv, strat, bars))

    with ThreadPoolExecutor(engine.workers) as ex:
        list(ex.map(write_chart, charts.items()))
    print(f"{len(state['signals'])} 個訊號，{len(pairs)} 個幣的圖表，輸出到 {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
