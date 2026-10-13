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
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from grid_trading import signals
from grid_trading.markets import COMMODITY, CRYPTO, STOCK
from grid_trading.signals import ledger as ledger_mod
from grid_trading.signals.engine import Engine, fetch_klines
from signal_backtest import liquid_symbols
from signal_server import _sanitize

CHART_INTERVALS = ["15M", "60M", "4H", "1D"]


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_sanitize(obj), ensure_ascii=False, separators=(",", ":"), allow_nan=False))


def main() -> int:
    p = argparse.ArgumentParser(description="產生訊號網頁的靜態資料")
    p.add_argument("--out", default="site")
    p.add_argument("--web", default="web", help="網頁原始檔，會複製到 --out")
    p.add_argument("--previous", help="上一次的 notifications.json，用來接續通知紀錄")
    p.add_argument("--ledger", help="上一次的 ledger.json（實盤紀錄），沒有就從頭開始")
    p.add_argument("--strategies", nargs="+", default=["vegas_trend", "smc_sweep"], choices=sorted(signals.REGISTRY))
    p.add_argument("--intervals", nargs="+", default=["4H", "2H"])
    p.add_argument("--crypto-list", choices=["top100", "volume"], default="top100",
                   help="加密貨幣：市值前 100 名（預設），或 24h 成交額 >= --min-volume")
    p.add_argument("--min-volume", type=float, default=5e6, help="加密貨幣 24h 成交額下限 (USDT)")
    p.add_argument("--min-volume-rwa", type=float, default=3e5, help="美股/商品 24h 成交額下限 (USDT)")
    p.add_argument("--markets", nargs="+", default=[CRYPTO, STOCK, COMMODITY], choices=[CRYPTO, STOCK, COMMODITY])
    p.add_argument("--keep-days", type=float, default=7)
    p.add_argument("--max-failed", type=float, default=0.02,
                   help="K 線抓不到的（幣, 週期）超過這個比例就不輸出，避免網站少掉一批訊號")
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
    engine.scan_once()
    if len(engine.failed) > args.max_failed * len(symbols) * len(args.intervals):
        # 一部分幣的 K 線沒抓到時，那些幣的訊號會整批從網站消失（例如 2H 全不見、持倉變空白）。
        # 這種時候寧可這次不更新，網站繼續顯示上一版，等下一輪再掃。
        print(f"{len(engine.failed)} 組 K 線重試後仍抓不到，這次不更新網站：{engine.failed[:10]}", file=sys.stderr)
        return 1
    if engine.failed:
        print(f"注意：{len(engine.failed)} 組 K 線沒抓到：{engine.failed}", file=sys.stderr)
    state = engine.state()
    state["static"] = True

    previous = json.loads(Path(args.ledger).read_text()) if args.ledger and Path(args.ledger).exists() else []
    book = ledger_mod.update(previous, state["signals"], lambda sym, iv: fetch_klines(engine.client(), sym, iv, 1000),
                             engine.fee)
    write_json(data / "ledger.json", book)
    state["ledger"] = ledger_mod.summary(book)
    write_json(data / "state.json", state)

    pairs = {(s["symbol"], s["interval"], s["strategy"]) for s in state["signals"]}
    charts: dict[tuple[str, str], tuple[str, int]] = {}  # (幣, 週期) → (策略, 根數)；訊號本身的週期多抓一些
    for sym, sig_iv, strat in sorted(pairs):
        for iv in dict.fromkeys([sig_iv, *CHART_INTERVALS]):
            bars = 1000 if iv == sig_iv else 700
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
