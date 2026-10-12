"""訊號策略回測：python signal_backtest.py [--source binance]

把策略在歷史 K 線上逐根掃描，統計勝率、平均 R、期望值，結果存到 stats/signal_stats.json，
訊號網頁會把這些數字顯示在訊號旁邊。

--source binance 用幣安 USDT 永續的月檔 K 線（2020 年起，只到上個月底），只適用加密貨幣；
結果存在「週期@binance」這種鍵，網頁以幣安的數字為主、派網的為輔。掃描和下單仍然只用派網。
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from grid_trading import signals
from grid_trading.binance_history import fetch_binance
from grid_trading.history import fetch_history
from grid_trading.markets import COMMODITY, CRYPTO, STOCK, classify
from grid_trading.pionex import PionexPublic
from grid_trading.universe import TOP100, perp_symbols
from grid_trading.signals.evaluate import group_summary, split_summary, trades_for

STATS_PATH = Path("stats/signal_stats.json")
BINANCE_INTERVALS = {"60M": "1h", "4H": "4h"}


def stats_key(market: str, interval: str, source: str = "pionex") -> str:
    """stats 裡的鍵：加密貨幣「4H」、美股「stock|4H」，幣安資料再加「@binance」。"""
    key = interval if market == CRYPTO else f"{market}|{interval}"
    return key if source == "pionex" else f"{key}@{source}"


def liquid_symbols(min_volume: float, markets: tuple[str, ...] = (CRYPTO,),
                   min_volume_rwa: float | None = None, crypto_list: str = "top100") -> list[str]:
    """要掃描的永續合約。markets 選類別。
    加密貨幣：crypto_list="top100" 用 CoinMarketCap 市值前 100 名（派網有合約的），"volume" 用 24h 成交額 >= min_volume。
    美股/商品：24h 成交額 >= min_volume_rwa（成交量普遍比幣小）。"""
    api = PionexPublic()
    tickers = api.perp_tickers()
    amount = dict(zip(tickers["symbol"], tickers["amount"]))
    top = set(perp_symbols(TOP100))
    out = []
    for s in api.perp_symbols():
        m = classify(s)
        if m not in markets:
            continue
        if m == CRYPTO and crypto_list == "top100":
            if s in top:
                out.append(s)
            continue
        floor = min_volume if m == CRYPTO or min_volume_rwa is None else min_volume_rwa
        if amount.get(s, 0) >= floor:
            out.append(s)
    return sorted(out)


def main() -> int:
    p = argparse.ArgumentParser(description="訊號策略回測")
    p.add_argument("--strategy", default="vegas_trend", choices=sorted(signals.REGISTRY))
    p.add_argument("--intervals", nargs="+", default=["60M", "4H"])
    p.add_argument("--days", type=int, default=420, help="60M 最多約 420 天；4H 可到 1300 天以上")
    p.add_argument("--days-4h", type=int, default=1300)
    p.add_argument("--crypto-list", choices=["top100", "volume"], default="top100",
                   help="加密貨幣用市值前 100 名，或用 --min-volume 成交額門檻")
    p.add_argument("--min-volume", type=float, default=5e6)
    p.add_argument("--market", choices=[CRYPTO, STOCK, COMMODITY], default=CRYPTO,
                   help="回測哪一類標的；美股/商品的結果存成 stats 裡的「類別|週期」")
    p.add_argument("--min-volume-rwa", type=float, default=3e5)
    p.add_argument("--symbols", nargs="*")
    p.add_argument("--source", choices=["pionex", "binance"], default="pionex",
                   help="K 線來源；binance 只能用在加密貨幣（2020 年起的月檔）")
    p.add_argument("--fee", type=float, default=0.0005)
    p.add_argument("--risk", type=float, default=0.0025,
                   help="每張單虧損上限佔總資金比例（1R），預設 0.25%%（1%% 時回撤超過 -70%%）")
    p.add_argument("--param", action="append", default=[], help="策略參數，例如 --param sl_atr=2")
    args = p.parse_args()

    if args.source == "binance" and args.market != CRYPTO:
        p.error("幣安資料只有加密貨幣")
    params = {k: json.loads(v) for k, v in (x.split("=", 1) for x in args.param)}
    strat = signals.get(args.strategy, **params)
    symbols = args.symbols or liquid_symbols(args.min_volume, (args.market,), args.min_volume_rwa, args.crypto_list)

    stats = json.loads(STATS_PATH.read_text()) if STATS_PATH.exists() else {}
    for interval in args.intervals:
        days = args.days_4h if interval == "4H" else args.days
        rows = []
        for sym in symbols:
            if args.source == "binance":
                df = fetch_binance(sym.split("_")[0], BINANCE_INTERVALS[interval])
            else:
                df = fetch_history(sym, interval, days)
            if len(df) < strat.min_bars + 50:
                continue
            t = trades_for(strat, df, sym, interval, args.fee)
            rows.extend(t)
            print(f"  {interval} {sym}: {len(df)} 根，{len(t)} 筆交易", file=sys.stderr)
        trades = pd.DataFrame(rows)
        summary = split_summary(trades, args.risk)
        groups = group_summary(trades)
        if groups:
            summary["rsi_groups"] = groups
        summary.update({"market": args.market, "source": args.source, "strategy": strat.name, "title": strat.title, "interval": interval, "params": params,
                        "fee": args.fee, "generated": datetime.now(timezone.utc).isoformat(timespec="seconds")})
        stats.setdefault(strat.name, {})[stats_key(args.market, interval, args.source)] = summary
        Path("output").mkdir(exist_ok=True)
        STATS_PATH.parent.mkdir(exist_ok=True)
        if not trades.empty:
            src = "" if args.source == "pionex" else f"_{args.source}"
            trades.to_csv(f"output/signal_trades_{strat.name}_{args.market}_{interval}{src}.csv", index=False)
        a = summary["all"]
        if a.get("trades"):
            print(f"{interval}: {a['trades']} 筆，勝率 {a['win_rate']:.1%}，期望值 {a['expectancy_r']:+.3f}R，"
                  f"獲利因子 {a['profit_factor']:.2f}，每筆風險 {args.risk:.2%} 時總報酬 {a['total_return_pct']:+.1f}%，"
                  f"最大回撤 {a['max_drawdown_pct']:.1f}%")
            for half in ("first_half", "second_half"):
                h = summary[half]
                if h.get("trades"):
                    print(f"   {half}: {h['trades']} 筆，勝率 {h['win_rate']:.1%}，期望值 {h['expectancy_r']:+.3f}R")
    STATS_PATH.write_text(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"已寫入 {STATS_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
