"""產生靜態網站用的資料快照：python signal_snapshot.py --out site

給 GitHub Pages 用。GitHub Actions 每隔幾分鐘跑一次：掃描訊號、寫出 JSON，
網頁（web/）直接讀這些 JSON，不需要一直開著的伺服器。

輸出：
  <out>/data/state.json                    訊號、通知、回測數字（同 /api/state）
  <out>/data/notifications.json            通知紀錄，下一次執行接著用
  <out>/data/chart/<symbol>_<interval>.json  圖表資料（同 /api/chart）
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from grid_trading import signals
from grid_trading.markets import COMMODITY, CRYPTO, STOCK
from grid_trading.signals.engine import Engine
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
    p.add_argument("--strategies", nargs="+", default=["vegas_trend"], choices=sorted(signals.REGISTRY))
    p.add_argument("--intervals", nargs="+", default=["4H"])
    p.add_argument("--min-volume", type=float, default=5e6, help="加密貨幣 24h 成交額下限 (USDT)")
    p.add_argument("--min-volume-rwa", type=float, default=3e5, help="美股/商品 24h 成交額下限 (USDT)")
    p.add_argument("--markets", nargs="+", default=[CRYPTO, STOCK, COMMODITY], choices=[CRYPTO, STOCK, COMMODITY])
    p.add_argument("--keep-days", type=float, default=7)
    args = p.parse_args()

    out = Path(args.out)
    data = out / "data"
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(args.web, out)
    data.mkdir(parents=True)
    (out / ".nojekyll").write_text("")
    if args.previous and Path(args.previous).exists():
        shutil.copy(args.previous, data / "notifications.json")

    symbols = liquid_symbols(args.min_volume, tuple(args.markets), args.min_volume_rwa)
    engine = Engine([signals.get(n) for n in args.strategies], symbols, args.intervals, args.keep_days, state_dir=data)
    engine.scan_once()
    state = engine.state()
    state["static"] = True
    write_json(data / "state.json", state)

    pairs = {(s["symbol"], s["interval"], s["strategy"]) for s in state["signals"]}
    for sym, sig_iv, strat in sorted(pairs):
        for iv in dict.fromkeys([sig_iv, *CHART_INTERVALS]):
            write_json(data / "chart" / f"{sym}_{iv}.json", engine.chart(sym, iv, strat, 1000 if iv == sig_iv else 700))
    print(f"{len(state['signals'])} 個訊號，{len(pairs)} 個幣的圖表，輸出到 {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
