"""幣安自動交易（實盤維加斯 4H + 2H）。要在美國以外、固定 IP 的主機上跑，安裝方法見 docs/binance_bot.md。

  python binance_trader.py --mode paper              # 紙上交易：不用金鑰，只記帳
  python binance_trader.py --mode testnet            # 測試網：要測試網的 BINANCE_API_KEY / BINANCE_API_SECRET
  TRADER_ALLOW_LIVE=1 python binance_trader.py --mode live   # 正式帳戶（測試網跑過再開）

選項：--risk 0.0025（每筆風險）、--max-risk 0.01（最小下單量造成的風險上限）、--leverage 5（逐倉）、
--max-open 15（同時最多幾張）、--dd-half-r 20（回撤 20R 後風險減半）、--state-dir（狀態檔位置）。
暫停開新倉：在狀態資料夾建立 pause 檔（touch trader_state/pause）；刪掉就恢復。已開的倉位照常管理。
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from grid_trading.trader.binance import MAINNET, TESTNET, BinanceFutures
from grid_trading.trader.bot import LABEL, Config, LiveBroker, Notifier, PaperBroker, Trader
from grid_trading.trader.risk import RiskRules
from grid_trading.trader.sizing import parse_filters


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["paper", "testnet", "live"], default="paper")
    ap.add_argument("--risk", type=float, default=0.0025)
    ap.add_argument("--max-risk", type=float, default=0.01)
    ap.add_argument("--leverage", type=int, default=5)
    ap.add_argument("--max-open", type=int, default=None)
    ap.add_argument("--dd-half-r", type=float, default=None)
    ap.add_argument("--paper-equity", type=float, default=100.0)
    ap.add_argument("--state-dir", type=Path, default=Path("trader_state"))
    ap.add_argument("--once", action="store_true", help="只跑一輪（測試用）")
    a = ap.parse_args(argv)

    if a.mode == "live" and os.environ.get("TRADER_ALLOW_LIVE") != "1":
        print("正式帳戶要先設 TRADER_ALLOW_LIVE=1（請先在測試網跑過）", file=sys.stderr)
        return 2
    cfg = Config(mode=a.mode, risk=a.risk, max_risk=a.max_risk, leverage=a.leverage, paper_equity=a.paper_equity,
                 rules=RiskRules(max_open=a.max_open, dd_half_r=a.dd_half_r), state_dir=a.state_dir)
    data = BinanceFutures(MAINNET)
    if a.mode == "paper":
        venue = data
        broker = PaperBroker(data, cfg)
    else:
        key, secret = os.environ.get("BINANCE_API_KEY", ""), os.environ.get("BINANCE_API_SECRET", "")
        if not (key and secret):
            print("沒有設定 BINANCE_API_KEY / BINANCE_API_SECRET", file=sys.stderr)
            return 2
        venue = BinanceFutures(TESTNET if a.mode == "testnet" else MAINNET, key, secret)
        venue.sync_time()
        broker = LiveBroker(venue, data, cfg)
    filters = parse_filters(venue.exchange_info())
    notify = Notifier(LABEL[a.mode], os.environ.get("TELEGRAM_BOT_TOKEN", ""), os.environ.get("TELEGRAM_CHAT_ID", ""))
    Trader(cfg, data, broker, notify, filters).run(once=a.once)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
