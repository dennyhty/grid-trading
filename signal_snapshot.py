"""產生靜態網站用的資料快照：python signal_snapshot.py --out site

給 GitHub Pages 用。GitHub Actions 每隔幾分鐘跑一次：掃描訊號、寫出 JSON，
網頁（web/）直接讀這些 JSON，不需要一直開著的伺服器。

輸出：
  <out>/data/state.json                    訊號、通知、回測數字（同 /api/state）
  <out>/data/notifications.json            通知紀錄，下一次執行接著用
  <out>/data/ledger.json                   每個訊號的結果（實盤紀錄），下一次執行接著用
  <out>/data/chart/<symbol>_<interval>.json  圖表資料（同 /api/chart）
"""

from __future__ import annotations

import argparse
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
from grid_trading.markets import COMMODITY, CRYPTO, STOCK, classify
from grid_trading.pionex import PionexPublic
from grid_trading.signals import ledger as ledger_mod
from grid_trading.signals.engine import INTERVAL_MS, Engine
from signal_backtest import liquid_symbols
from signal_server import _sanitize

CHART_INTERVALS = ["15M", "60M", "4H", "1D"]


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_sanitize(obj), ensure_ascii=False, separators=(",", ":"), allow_nan=False))


def order_specs(symbols: set[str], previous: dict) -> dict:
    """網頁「下單計算」用的派網最小下單量；抓不到就沿用上一次的。"""
    try:
        specs = PionexPublic().perp_specs()
    except Exception as e:
        print(f"注意：派網下單限制抓不到（{e}），沿用上一次的", file=sys.stderr)
        specs = previous
    return {s: specs[s] for s in sorted(symbols) if s in specs}


def closing_intervals(intervals: list[str], now_ms: int, within_ms: int = 10 * 60_000) -> list[str]:
    """最近 within_ms 內剛收盤的週期（例如 UTC 08:00 是 4H 和 2H，10:00 只有 2H）。"""
    return [iv for iv in intervals if now_ms % INTERVAL_MS[iv] < within_ms]


def early_push(args, symbols: list[str], data: Path, previous: dict, sent_path: Path) -> dict:
    """2H/4H 剛收盤時先只掃加密貨幣的實盤策略（沒有 tag 的），新通知馬上送 Telegram，不等整個網站產生和部署完。

    抓到的 K 線回傳給完整掃描沿用，不多打派網。通知 id 和完整掃描算出來的一樣，所以送過的寫到 sent_path，
    之後的 outbox 跳過。萬一這一輪網站沒部署成功，下一輪會再送一次（寧可重複，不要漏）。"""
    sent_path.write_text("[]")
    closing = closing_intervals(args.intervals, int(time.time() * 1000))
    live = [n for n in args.strategies if not signals.get(n).tag]
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
    quick.previous = {k: s for k, s in previous.items()
                      if s.get("symbol") in set(crypto) and s.get("interval") in closing and s.get("strategy") in live}
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
    p.add_argument("--strategies", nargs="+", default=["vegas_trend", "smc_sweep", "smc_sweep_2h", "oi_build", "oi_build_2h"], choices=sorted(signals.REGISTRY))
    p.add_argument("--intervals", nargs="+", default=["4H", "2H"])
    p.add_argument("--crypto-list", choices=["top100", "volume"], default="top100",
                   help="加密貨幣：市值前 100 名（預設），或 24h 成交額 >= --min-volume")
    p.add_argument("--min-volume", type=float, default=5e6, help="加密貨幣 24h 成交額下限 (USDT)")
    p.add_argument("--min-volume-rwa", type=float, default=3e5, help="美股/商品 24h 成交額下限 (USDT)")
    p.add_argument("--markets", nargs="+", default=[CRYPTO, STOCK, COMMODITY], choices=[CRYPTO, STOCK, COMMODITY])
    p.add_argument("--keep-days", type=float, default=7)
    p.add_argument("--previous-state", help="上一次的 state.json；這次 K 線沒抓到的（幣, 週期）沿用裡面的訊號")
    p.add_argument("--early-push", metavar="SENT_JSON",
                   help="2H/4H 剛收盤時，先只掃加密貨幣的實盤策略、馬上把新通知送到 Telegram（要有 TELEGRAM_BOT_TOKEN / "
                        "TELEGRAM_CHAT_ID），送過的通知 id 寫到這個檔，之後的 outbox 會跳過")
    p.add_argument("--max-failed", type=float, default=0.5,
                   help="K 線抓不到的（幣, 週期）超過這個比例（大概是派網整個連不上）就不輸出，網站保留上一版")
    args = p.parse_args()

    out = Path(args.out)
    data = out / "data"
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(args.web, out)
    # 每次部署給 app.js、style.css 加上版本號，瀏覽器才不會拿舊的頁面配新的程式（或反過來）
    version = str(int(time.time()))
    index = out / "index.html"
    index.write_text(index.read_text().replace('src="app.js"', f'src="app.js?v={version}"')
                     .replace('href="style.css"', f'href="style.css?v={version}"'))
    data.mkdir(parents=True)
    (out / ".nojekyll").write_text("")
    if args.previous and Path(args.previous).exists():
        shutil.copy(args.previous, data / "notifications.json")

    symbols = liquid_symbols(args.min_volume, tuple(args.markets), args.min_volume_rwa, args.crypto_list)
    engine = Engine([signals.get(n) for n in args.strategies], symbols, args.intervals, args.keep_days, state_dir=data)
    try:
        prev_state = json.loads(Path(args.previous_state).read_text()) if args.previous_state else {}
    except Exception:
        prev_state = {}
    engine.previous = {s["id"]: s for s in prev_state.get("signals", []) if "id" in s}
    if args.early_push:
        engine.prefetched = early_push(args, symbols, data, engine.previous, Path(args.early_push))
    engine.scan_once()
    if engine.failed:
        sample = sorted(set(engine.errors.values()))[:3]
        print(f"注意：{len(engine.failed)} 組 K 線重試後仍抓不到，沿用上一次的 {engine.carried} 個訊號。"
              f"例如 {engine.failed[:5]}；錯誤 {sample}", file=sys.stderr)
    if len(engine.failed) > args.max_failed * len(symbols) * len(args.intervals):
        print("抓不到的太多（大概是派網連不上），這次不更新網站", file=sys.stderr)
        return 1
    state = engine.state()
    state["static"] = True

    previous = json.loads(Path(args.ledger).read_text()) if args.ledger and Path(args.ledger).exists() else []
    book = ledger_mod.update(previous, state["signals"], lambda sym, iv: engine.klines(sym, iv, 1000),
                             engine.fee)
    write_json(data / "ledger.json", book)
    state["ledger"] = ledger_mod.summary(book)
    state["specs"] = order_specs({s["symbol"] for s in state["signals"]}, prev_state.get("specs") or {})
    write_json(data / "state.json", state)

    pairs = {(s["symbol"], s["interval"], s["strategy"]) for s in state["signals"]}
    charts: dict[tuple[str, str], tuple[str, int]] = {}  # (幣, 週期) → (策略, 根數)；訊號本身的週期多抓一些
    for sym, sig_iv, strat in sorted(pairs):
        for iv in dict.fromkeys([sig_iv, *CHART_INTERVALS]):
            bars = 1000 if iv == sig_iv else 500  # 其他週期只抓一頁（500 根），減少對派網的請求
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
