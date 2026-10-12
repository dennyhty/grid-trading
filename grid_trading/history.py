"""抓長期歷史 K 線（往回翻頁）並快取到 data/。

派網公開 API 每次最多 500 根，可往回查的深度依週期不同：
60M 約一年多、15M 不到 200 天，4H 和 1D 可以回到 2023-01。
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pandas as pd
import requests

from .pionex import BASE_URL

CACHE_DIR = Path("data")
INTERVAL_MS = {"15M": 15 * 60_000, "30M": 30 * 60_000, "60M": 3_600_000, "4H": 4 * 3_600_000,
               "8H": 8 * 3_600_000, "12H": 12 * 3_600_000, "1D": 86_400_000}


def fetch_history(symbol: str, interval: str = "60M", days: int = 420, refresh: bool = False) -> pd.DataFrame:
    """回傳最近 days 天的 K 線，依時間由舊到新排序。結果快取在 data/{symbol}_{interval}.csv.gz。"""
    CACHE_DIR.mkdir(exist_ok=True)
    path = CACHE_DIR / f"{symbol}_{interval}.csv.gz"
    now_ms = int(time.time() * 1000)
    start_ms = now_ms - days * 86_400_000

    cached = pd.DataFrame()
    if path.exists() and not refresh:
        cached = pd.read_csv(path)
        if not cached.empty and cached["time"].min() <= start_ms + INTERVAL_MS[interval] \
                and cached["time"].max() >= now_ms - 2 * INTERVAL_MS[interval]:
            return _finish(cached, start_ms)

    rows: list[dict] = []
    end = None
    session = requests.Session()
    while True:
        params = {"symbol": symbol, "interval": interval, "limit": 500}
        if end is not None:
            params["endTime"] = end
        body = session.get(BASE_URL + "/api/v1/market/klines", params=params, timeout=15).json()
        if not body.get("result"):
            break  # 超過可查詢深度時 API 回 endTime param error
        batch = body["data"].get("klines", [])
        if not batch:
            break
        rows.extend(batch)
        oldest = min(k["time"] for k in batch)
        if oldest <= start_ms or len(batch) < 500:
            break
        end = oldest - 1
        time.sleep(0.12)

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = pd.concat([cached, df]).drop_duplicates("time")
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.sort_values("time").reset_index(drop=True)
    # 最後一根可能還沒收盤，不存
    df = df[df["time"] <= now_ms - INTERVAL_MS[interval]]
    # 先寫暫存檔再改名，多個程序同時讀寫快取時才不會讀到寫一半的檔案
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp.gz")
    df[["time", "open", "high", "low", "close", "volume"]].to_csv(tmp, index=False)
    os.replace(tmp, path)
    return _finish(df, start_ms)


def _finish(df: pd.DataFrame, start_ms: int) -> pd.DataFrame:
    df = df[df["time"] >= start_ms].copy()
    df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True)
    return df.reset_index(drop=True)


def resample(df: pd.DataFrame, rule: str = "4h", origin="start_day") -> pd.DataFrame:
    """把 1H K 線合成 4H，讓回測時的選幣訊號和實盤一致，又不會偷看未來資料。

    origin 決定分組從哪裡起算，預設是第一天的 00:00 UTC。
    """
    out = df.set_index("time").resample(rule, label="left", closed="left", origin=origin).agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna()
    return out.reset_index()
