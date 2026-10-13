"""訊號網頁：python signal_server.py，然後用瀏覽器開 http://localhost:8000

手機要看的話，手機和電腦連同一個 Wi-Fi，開 http://<電腦的區網 IP>:8000。
只用派網公開行情，不需要 API Key，也不會下單。
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from grid_trading import signals
from grid_trading.markets import COMMODITY, CRYPTO, STOCK
from grid_trading.signals.engine import INTERVAL_MS, Engine

WEB = Path(__file__).parent / "web"


def make_handler(engine: Engine):
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=str(WEB), **kw)

        def log_message(self, fmt, *args):  # 不要每個請求都印一行
            pass

        def _json(self, obj, code: int = 200):
            body = json.dumps(obj, ensure_ascii=False, allow_nan=False, default=_clean).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            url = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(url.query).items()}
            try:
                if url.path == "/api/state":
                    return self._json(_sanitize(engine.state()))
                if url.path == "/api/price":
                    return self._json({"symbol": q.get("symbol"), "price": engine.prices().get(q.get("symbol", ""))})
                if url.path == "/api/chart":
                    interval = q.get("interval", "4H")
                    if interval not in INTERVAL_MS or not q.get("symbol", "").endswith("_USDT_PERP"):
                        return self._json({"error": "參數錯誤"}, 400)
                    return self._json(_sanitize(engine.chart(q["symbol"], interval, q.get("strategy"))))
            except Exception as e:
                return self._json({"error": str(e)}, 500)
            return super().do_GET()

    return Handler


def _clean(o):
    return str(o)


def _sanitize(o):
    """JSON 不接受 NaN / inf，換成 None。"""
    if isinstance(o, float):
        return o if o == o and abs(o) != float("inf") else None
    if isinstance(o, dict):
        return {k: _sanitize(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_sanitize(v) for v in o]
    return o


def main() -> int:
    p = argparse.ArgumentParser(description="交易訊號網頁")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--strategies", nargs="+", default=["vegas_trend", "smc_sweep", "smc_sweep_2h", "oi_build"], choices=sorted(signals.REGISTRY))
    p.add_argument("--intervals", nargs="+", default=["4H", "2H"],
                   help="預設看 4H 和 2H（2H 由 1H 合成）；60M（1H）回測是負期望值，要看可加上")
    p.add_argument("--symbols", nargs="*", help="預設：市值前 100 名的幣，加上成交額夠的美股/商品")
    p.add_argument("--crypto-list", choices=["top100", "volume"], default="top100",
                   help="加密貨幣：市值前 100 名（預設），或 24h 成交額 >= --min-volume")
    p.add_argument("--min-volume", type=float, default=5e6, help="加密貨幣 24h 成交額下限 (USDT)")
    p.add_argument("--min-volume-rwa", type=float, default=3e5, help="美股/商品 24h 成交額下限 (USDT)")
    p.add_argument("--markets", nargs="+", default=[CRYPTO, STOCK, COMMODITY], choices=[CRYPTO, STOCK, COMMODITY])
    p.add_argument("--keep-days", type=float, default=7, help="顯示最近幾天的訊號")
    p.add_argument("--every", type=float, default=120, help="每幾秒重新掃描一次")
    args = p.parse_args()

    if args.symbols:
        symbols = args.symbols
    else:
        from signal_backtest import liquid_symbols
        symbols = liquid_symbols(args.min_volume, tuple(args.markets), args.min_volume_rwa, args.crypto_list)
    engine = Engine([signals.get(n) for n in args.strategies], symbols, args.intervals, args.keep_days)
    print(f"監看 {len(symbols)} 個幣、週期 {args.intervals}，第一次掃描中…", file=sys.stderr)
    engine.scan_once()
    threading.Thread(target=engine.run_forever, args=(args.every,), daemon=True).start()
    server = ThreadingHTTPServer((args.host, args.port), make_handler(engine))
    print(f"開啟 http://localhost:{args.port}", file=sys.stderr)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
