"""唯讀檢查：某個幣在程式狀態檔和幣安上的倉位、掛單對不對得上。不下單、不改狀態，輸出沒有金鑰。

  cd ~/grid-trading && set -a && . /home/trader/trader.env && set +a && \
    .venv/bin/python deploy/trader_diag.py APTUSDT            # 測試網（預設）
    .venv/bin/python deploy/trader_diag.py APTUSDT --mode live
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from grid_trading.trader.binance import MAINNET, TESTNET, BinanceFutures  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("symbol", help="例如 APTUSDT")
    ap.add_argument("--mode", choices=["testnet", "live"], default="testnet")
    ap.add_argument("--state-dir", type=Path, default=Path("trader_state"))
    a = ap.parse_args()
    state = json.loads((a.state_dir / f"{a.mode}.json").read_text())
    print(f"== 程式狀態檔裡 {a.symbol} 的單 ==")
    for pid, p in state.get("positions", {}).items():
        if p["symbol"] != a.symbol:
            continue
        keys = ("interval", "side", "qty_f", "remaining", "tps_hit", "entry", "stop", "tps", "slices",
                "stop_id", "tp_ids", "stop_version", "last_algo_status", "last_error", "opened")
        print(pid, {k: p.get(k) for k in keys})
    venue = BinanceFutures(TESTNET if a.mode == "testnet" else MAINNET,
                           os.environ["BINANCE_API_KEY"], os.environ["BINANCE_API_SECRET"])
    venue.sync_time()
    print("== 幣安上的倉位 ==")
    for k, v in venue.positions().items():
        if k[0] == a.symbol:
            print(k, v)
    print("== 幣安上的條件單 ==")
    for o in venue.open_conditionals(a.symbol):
        print({k: o.get(k) for k in ("algoId", "positionSide", "side", "orderType", "type", "triggerPrice",
                                     "quantity", "algoStatus", "clientAlgoId", "closePosition")})
    return 0


if __name__ == "__main__":
    sys.exit(main())
