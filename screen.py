"""選幣篩選器：python screen.py

抓派網所有 USDT 永續合約的公開 K 線，排出目前最適合開網格的幣種。
結果印在畫面上，並存一份 CSV 到 output/。不需要 API Key。
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from grid_trading.pionex import PionexPublic
from grid_trading.screener import ScreenConfig, analyze, rank

COLUMNS = {
    "score": "分數",
    "direction": "方向",
    "price": "現價",
    "lower": "下限",
    "upper": "上限",
    "grids": "格數",
    "grid_spacing_pct": "格距%",
    "stop_low": "下方停損",
    "stop_high": "上方停損",
    "adx": "ADX",
    "efficiency": "效率比",
    "crossings": "穿越次數",
    "atr_pct": "ATR%",
    "amount": "24h成交額",
}


def main() -> int:
    p = argparse.ArgumentParser(description="派網合約網格選幣篩選器")
    p.add_argument("--interval", default="4H", help="K 線週期，預設 4H")
    p.add_argument("--limit", type=int, default=500, help="每個幣抓幾根 K 線，預設 500")
    p.add_argument("--min-volume", type=float, default=5e6, help="24h 成交額下限 (USDT)，預設 500 萬")
    p.add_argument("--top", type=int, default=15, help="顯示前幾名，預設 15")
    p.add_argument("--symbols", nargs="*", help="只看指定幣種，例如 BTC_USDT_PERP ETH_USDT_PERP")
    args = p.parse_args()

    api = PionexPublic()
    tickers = api.perp_tickers()
    symbols = args.symbols or api.perp_symbols()
    if not args.symbols and "amount" in tickers:
        liquid = set(tickers.loc[tickers["amount"] >= args.min_volume, "symbol"])
        symbols = [s for s in symbols if s in liquid]
    print(f"篩選 {len(symbols)} 個幣種（{args.interval} K 線）...", file=sys.stderr)

    cfg = ScreenConfig()
    results = {}
    for i, sym in enumerate(symbols, 1):
        try:
            df = api.klines(sym, args.interval, args.limit)
        except Exception as e:  # 單一幣種失敗不影響其他
            print(f"  跳過 {sym}: {e}", file=sys.stderr)
            continue
        r = analyze(df, cfg)
        if r is not None:
            results[sym] = r
        if i % 20 == 0:
            print(f"  已處理 {i}/{len(symbols)}", file=sys.stderr)

    if not results:
        print("沒有足夠資料的幣種。", file=sys.stderr)
        return 1

    table = rank(results, tickers)
    out_dir = Path("output")
    out_dir.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
    csv_path = out_dir / f"screen_{args.interval}_{stamp}.csv"
    table.to_csv(csv_path, encoding="utf-8-sig")

    view = table[[c for c in COLUMNS if c in table]].head(args.top).rename(columns=COLUMNS)
    with pd.option_context("display.width", 200, "display.max_columns", None, "display.float_format", "{:.4g}".format):
        print(view.to_string())
    print(f"\n完整結果：{csv_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
