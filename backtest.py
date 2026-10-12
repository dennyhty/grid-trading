"""網格回測：python backtest.py

用一年多的 1H K 線，滾動檢驗篩選器的開單規則，比較不同參數組合。
前半段時間當「訓練期」挑參數，後半段當「驗證期」確認不是巧合。
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import pandas as pd

from grid_trading.backtest import Variant, summarize, windows
from grid_trading.history import fetch_history
from grid_trading.pionex import PionexPublic
from grid_trading.screener import ScreenConfig


def main() -> int:
    p = argparse.ArgumentParser(description="派網合約網格回測")
    p.add_argument("--days", type=int, default=420, help="抓多少天的 1H 歷史，預設 420")
    p.add_argument("--min-volume", type=float, default=5e6, help="24h 成交額下限 (USDT)")
    p.add_argument("--symbols", nargs="*", help="只測指定幣種")
    p.add_argument("--fee", type=float, default=0.0005, help="每次成交手續費率，預設 0.05%%")
    p.add_argument("--refresh", action="store_true", help="重新下載歷史資料")
    p.add_argument("--interval", choices=["60M", "4H"], default="60M",
                   help="模擬用的 K 線。60M 約可回溯 14 個月；4H 可回溯到 2023 年初，但同一根 K 線內的來回會被忽略")
    p.add_argument("--suite", choices=["base", "exits", "cycle"], default="exits",
                   help="base：第一輪 24 組；exits：加入提早收網與更緊停損的改良版")
    p.add_argument("--tag", default="", help="輸出檔名後綴")
    args = p.parse_args()

    if args.symbols:
        symbols = args.symbols
    else:
        api = PionexPublic()
        tickers = api.perp_tickers()
        liquid = set(tickers.loc[tickers["amount"] >= args.min_volume, "symbol"])
        symbols = sorted(s for s in api.perp_symbols() if s in liquid)

    data = {}
    for sym in symbols:
        df = fetch_history(sym, args.interval, args.days, args.refresh)
        if (df["time"].iloc[-1] - df["time"].iloc[0]).days > 60 if len(df) else False:
            data[sym] = df
        print(f"  {sym}: {len(df)} 根 {args.interval} K 線", file=sys.stderr)

    if args.suite == "base":
        variants = [
            Variant(direction=d, filtered=f, hold_days=h, stop_atr_mult=s)
            for d, f, h, s in itertools.product(["vegas", "neutral"], [True, False], [3, 7, 14], [1.0, 2.0])
        ]
    elif args.suite == "cycle":
        # 長週期：比較順勢、中性、只做多，都只在震盪時開、不加提早收網
        variants = [
            Variant(direction=d, filtered=True, hold_days=h, stop_atr_mult=s)
            for d, h, s in itertools.product(["vegas", "neutral", "long"], [7, 14], [0.5, 1.0])
        ]
    else:
        # 第一輪發現：過濾一定要開、持有久一點較好，所以只在這個基礎上改出場規則
        variants = [
            Variant(direction=d, filtered=True, hold_days=h, stop_atr_mult=s, exit_adx=e)
            for d, h, s, e in itertools.product(["vegas", "neutral"], [7, 14], [0.5, 1.0], [None, 30.0, 25.0])
        ]
    cfg = ScreenConfig()
    all_rows, summary = [], []
    for v in variants:
        rows = []
        for sym, df in data.items():
            rows += windows(df, v, cfg, sym, args.fee)
        res = pd.DataFrame(rows)
        if res.empty:
            continue
        res["variant"] = v.label()
        all_rows.append(res)
        mid = res["start"].min() + (res["start"].max() - res["start"].min()) / 2
        for period, part in (("全部", res), ("前半(訓練)", res[res["start"] < mid]), ("後半(驗證)", res[res["start"] >= mid])):
            summary.append({"組合": v.label(), "期間": period, **summarize(part, v.hold_days)})
        print(f"  完成 {v.label()}", file=sys.stderr)

    out = Path("output")
    out.mkdir(exist_ok=True)
    pd.concat(all_rows).to_csv(out / f"backtest_windows{args.tag}.csv", index=False, encoding="utf-8-sig")
    table = pd.DataFrame(summary)
    table.to_csv(out / f"backtest_summary{args.tag}.csv", index=False, encoding="utf-8-sig")

    with pd.option_context("display.width", 250, "display.max_columns", None, "display.max_rows", 200,
                           "display.float_format", "{:.2f}".format):
        print(table[table["期間"] == "全部"].sort_values("組合年化%", ascending=False).to_string(index=False))
    print(f"\n完整結果：{out / f'backtest_summary{args.tag}.csv'}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
