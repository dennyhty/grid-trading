"""把自動交易的紀錄（trader_state/*.json）整理成網站要的 JSON，用 HTTP 提供給 GitHub Actions 抓。

在 VPS 上和 binance_trader.py 一起跑（只用 Python 標準庫，不需要金鑰，也不會讀 trader.env）：

  FEED_TOKEN=一串隨機字 python3 deploy/trader_feed.py --state-dir trader_state --port 8787

網址是 http://VPS的IP:8787/<FEED_TOKEN>/trader.json；token 不對一律回 404。
把這個網址存成 GitHub 的 Actions secret「TRADER_FEED_URL」，網站每 15 分鐘更新時會抓一次。

輸出只有交易紀錄：持倉（幣、方向、進場價、止損、TP、數量）、已平倉（損益、R）、每個週期掃到哪根 K 線。
不含金鑰、訂單編號、帳戶設定。未平倉的浮動損益用幣安公開的標記價格估算。
"""

from __future__ import annotations

import argparse
import hmac
import json
import os
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

MODES = ("paper", "testnet", "live")
PRICE_URL = "https://fapi.binance.com/fapi/v1/premiumIndex"
_prices: dict = {"at": 0.0, "map": {}}


def mark_prices() -> dict[str, float]:
    """幣安合約的標記價格，最多 60 秒更新一次。抓不到就沿用上一次的。"""
    if time.time() - _prices["at"] > 60:
        try:
            with urllib.request.urlopen(PRICE_URL, timeout=10) as r:
                rows = json.load(r)
            _prices["map"] = {x["symbol"]: float(x["markPrice"]) for x in rows}
            _prices["at"] = time.time()
        except Exception as e:  # 斷線時網站只是少了浮動損益
            print("標記價格失敗:", e, flush=True)
    return _prices["map"]


def tps_hit(p: dict) -> int:
    """測試網/實盤有 tps_hit；紙上交易沒有，從止損移到哪裡推回去（保本 → 1，TP1 → 2，再往上追蹤 → 3）。"""
    if "tps_hit" in p:
        return int(p["tps_hit"])
    d = 1 if p["side"] == "LONG" else -1
    tps, stop, entry = p.get("tps") or [], p["stop"], p["entry"]
    if not tps or d * (stop - entry) <= 0:
        return 0
    tiny = abs(entry) * 1e-6
    if d * (stop - tps[0]) < -tiny:
        return 1
    return 2 if d * (stop - tps[0]) <= tiny else 3


def position(p: dict, prices: dict) -> dict:
    d = 1 if p["side"] == "LONG" else -1
    qty = float(p.get("qty_f") or 0)
    hits = tps_hit(p)
    out = {k: p.get(k) for k in ("id", "symbol", "interval", "side", "entry", "stop", "tps", "opened",
                                 "risk_usdt", "base_usdt", "risk_mult")}
    out["qty"] = qty
    out["tps_hit"] = hits
    out["signal_time"] = (p.get("signal") or {}).get("time")
    mark = prices.get(p["symbol"])
    if mark and qty and p.get("base_usdt"):
        # 已經到的 TP 各平 1/4（照價位估），剩下的照標記價格算
        part = sum(0.25 * qty * d * (t - p["entry"]) for t in (p.get("tps") or [])[:hits])
        rest = qty * max(0.0, 1 - 0.25 * hits) * d * (mark - p["entry"])
        out["mark"] = mark
        out["pnl_est"] = part + rest
        out["r_est"] = (part + rest) / p["base_usdt"]
    return out


def build(state_dir: Path) -> dict:
    prices = None
    books = {}
    for mode in MODES:
        path = state_dir / f"{mode}.json"
        try:
            data = json.loads(path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            continue
        if data.get("positions") and prices is None:
            prices = mark_prices()
        books[mode] = {
            "updated": int(path.stat().st_mtime * 1000),
            "processed": data.get("processed", {}),
            "paused": (state_dir / "pause").exists(),
            "positions": [position(p, prices or {}) for p in data.get("positions", {}).values()],
            "ledger": [{k: x.get(k) for k in ("id", "symbol", "interval", "side", "opened", "closed", "pnl", "r")}
                       for x in data.get("ledger", [])],
        }
    return {"generated": int(time.time() * 1000), "books": books}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--state-dir", type=Path, default=Path("trader_state"))
    ap.add_argument("--port", type=int, default=8787)
    ap.add_argument("--print", action="store_true", help="只印出 JSON（檢查用）")
    a = ap.parse_args()
    if a.print:
        print(json.dumps(build(a.state_dir), ensure_ascii=False, indent=1))
        return
    token = os.environ.get("FEED_TOKEN", "")
    if len(token) < 16:
        raise SystemExit("FEED_TOKEN 要設一串至少 16 個字的隨機字（例如 openssl rand -hex 16）")
    want = f"/{token}/trader.json"

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if not hmac.compare_digest(self.path.split("?")[0], want):
                self.send_error(404)
                return
            body = json.dumps(build(a.state_dir), ensure_ascii=False).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt, *args):  # 不印每個請求（含 token 的網址）
            pass

    print(f"trader feed 在 port {a.port}，狀態資料夾 {a.state_dir.resolve()}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
