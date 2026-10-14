"""幣安 USDT 永續的每日 metrics 檔（data.binance.vision），訂單流研究用。

每天一個檔，每 5 分鐘一列：
- sum_open_interest / sum_open_interest_value：持倉量（幣）/ 持倉價值（USDT）
- count_toptrader_long_short_ratio：大戶帳戶多空比；sum_toptrader_long_short_ratio：大戶持倉多空比
- count_long_short_ratio：全體帳戶多空比
- sum_taker_long_short_vol_ratio：主動買量 / 主動賣量

BTC 從 2020-09 起，大部分幣從 2021 年底起；中間偶爾缺幾天或幾列。
這裡取每小時最後一筆（時間對齊成該小時的開盤時間，和 K 線一樣），快取在 data/binance/{幣}_metrics_1h.csv.gz。
"""

from __future__ import annotations

import io
import os
import re
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

from .binance_history import FILES, LISTING, binance_symbols

CACHE_DIR = Path("data/binance")
FIELDS = {
    "sum_open_interest": "oi",
    "sum_open_interest_value": "oi_value",
    "count_toptrader_long_short_ratio": "top_account_ratio",
    "sum_toptrader_long_short_ratio": "top_position_ratio",
    "count_long_short_ratio": "account_ratio",
    "sum_taker_long_short_vol_ratio": "taker_ratio",
}


def _first_day(symbol: str, session: requests.Session) -> pd.Timestamp | None:
    prefix = f"data/futures/um/daily/metrics/{symbol}/"
    body = session.get(LISTING, params={"delimiter": "/", "prefix": prefix}, timeout=30).text
    keys = re.findall(r"<Key>[^<]+-(\d{4}-\d{2}-\d{2})\.zip</Key>", body)
    return pd.Timestamp(min(keys)) if keys else None


def _read_day(symbol: str, day: pd.Timestamp, session: requests.Session) -> pd.DataFrame | None:
    url = f"{FILES}data/futures/um/daily/metrics/{symbol}/{symbol}-metrics-{day:%Y-%m-%d}.zip"
    for _ in range(3):
        try:
            resp = session.get(url, timeout=30)
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            break
        except requests.RequestException:
            continue
    else:
        return None
    with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
        raw = z.read(z.namelist()[0]).decode()
    df = pd.read_csv(io.StringIO(raw))
    if "create_time" not in df:
        return None
    df = df.rename(columns=FIELDS)
    df["time"] = pd.to_datetime(df["create_time"], utc=True)
    return df[["time", *FIELDS.values()]]


def fetch_metrics(coin: str, end: str, refresh: bool = False, workers: int = 16) -> pd.DataFrame:
    """回傳每小時的 metrics（time 為該小時開盤時間，值為該小時最後一筆），抓到 end（含）為止。"""
    path = CACHE_DIR / f"{coin}_metrics_1h.csv.gz"
    if path.exists() and not refresh:
        return load_metrics(coin)
    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(pool_connections=workers, pool_maxsize=workers)
    session.mount("https://", adapter)
    parts = []
    for sym in binance_symbols(coin):
        first = _first_day(sym, session)
        if first is None:
            continue
        days = pd.date_range(first, pd.Timestamp(end), freq="D")
        with ThreadPoolExecutor(workers) as ex:
            parts += [d for d in ex.map(lambda d: _read_day(sym, d, session), days) if d is not None]
    if not parts:
        return pd.DataFrame()
    df = pd.concat(parts).drop_duplicates("time", keep="last").sort_values("time")
    for col in FIELDS.values():
        df[col] = pd.to_numeric(df[col], errors="coerce")
    # 每小時最後一筆：xx:55 那筆代表這一小時結束時的狀態；對齊到開盤時間，和 K 線的 time 一致
    hourly = df.set_index("time").resample("1h", label="left", closed="left").last().dropna(how="all")
    hourly = hourly.reset_index()
    hourly["time"] = (hourly["time"] - pd.Timestamp(0, tz="UTC")) // pd.Timedelta(milliseconds=1)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp.gz")
    hourly.to_csv(tmp, index=False)
    os.replace(tmp, path)
    return load_metrics(coin)


def load_metrics(coin: str) -> pd.DataFrame:
    path = CACHE_DIR / f"{coin}_metrics_1h.csv.gz"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True)
    return df
