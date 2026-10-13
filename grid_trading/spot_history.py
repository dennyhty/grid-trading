"""幣安現貨的日 K（data.binance.vision），給 BTC 長期定投（DCA）研究用。

- 月檔從 2017-08 開始，只到上個月；這個月的部分用日檔補到昨天。
- 結果快取在 data/binance_spot/{代號}_{週期}.csv.gz，欄位：time（毫秒）、open、high、low、close、volume。
"""

from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path

import pandas as pd
import requests

FILES = "https://data.binance.vision/"
LISTING = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
CACHE_DIR = Path("data/binance_spot")
COLUMNS = ["time", "open", "high", "low", "close", "volume"]


def _keys(prefix: str, session: requests.Session) -> list[str]:
    keys: list[str] = []
    marker = ""
    while True:
        params = {"prefix": prefix}
        if marker:
            params["marker"] = marker
        body = session.get(LISTING, params=params, timeout=30).text
        page = re.findall(r"<Key>([^<]+\.zip)</Key>", body)
        keys += page
        if "<IsTruncated>true</IsTruncated>" not in body or not page:
            return sorted(keys)
        marker = page[-1]


def _read_zip(key: str, session: requests.Session) -> pd.DataFrame:
    for _ in range(3):
        try:
            resp = session.get(FILES + key, timeout=60)
            resp.raise_for_status()
            break
        except requests.RequestException:
            continue
    else:
        raise RuntimeError(f"download failed: {key}")
    with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
        raw = z.read(z.namelist()[0]).decode()
    df = pd.read_csv(io.StringIO(raw), header=None, usecols=range(6), names=COLUMNS)
    df = df[pd.to_numeric(df["time"], errors="coerce").notna()].astype(float)
    df.loc[df["time"] > 1e14, "time"] /= 1000  # 2025 年起的現貨檔用微秒
    df["time"] = df["time"].astype("int64")
    return df


def fetch_spot(symbol: str = "BTCUSDT", interval: str = "1d") -> pd.DataFrame:
    session = requests.Session()
    monthly = _keys(f"data/spot/monthly/klines/{symbol}/{interval}/", session)
    frames = [_read_zip(k, session) for k in monthly]
    last = pd.Period(monthly[-1][-11:-4], "M")
    daily = _keys(f"data/spot/daily/klines/{symbol}/{interval}/{symbol}-{interval}-{last + 1}", session)
    frames += [_read_zip(k, session) for k in daily]
    df = pd.concat(frames).drop_duplicates("time").sort_values("time").reset_index(drop=True)
    return df


def load_spot(symbol: str = "BTCUSDT", interval: str = "1d", refresh: bool = False) -> pd.DataFrame:
    """回傳以 UTC 日期為索引的 K 線。"""
    path = CACHE_DIR / f"{symbol}_{interval}.csv.gz"
    if refresh or not path.exists():
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        fetch_spot(symbol, interval).to_csv(path, index=False)
    df = pd.read_csv(path)
    df.index = pd.to_datetime(df["time"], unit="ms")
    return df


CMC_URL = "https://api.coinmarketcap.com/data-api/v3.1/cryptocurrency/historical"


def fetch_cmc_btc(start: str = "2013-01-01", end: str = "2017-09-01") -> pd.DataFrame:
    """CoinMarketCap 的 BTC 日 K（UTC），補幣安 2017-08 之前的歷史。一次最多回 400 天，所以分段抓。"""
    session = requests.Session()
    rows = []
    t, stop = pd.Timestamp(start), pd.Timestamp(end)
    while t < stop:
        e = t + pd.Timedelta(days=300)
        params = {"id": 1, "convertId": 2781, "interval": "daily",
                  "timeStart": int(t.timestamp()), "timeEnd": int(e.timestamp())}
        quotes = session.get(CMC_URL, params=params, timeout=30).json()["data"]["quotes"]
        for q in quotes:
            day = pd.Timestamp(q["timeOpen"][:10])
            rows.append({"time": int(day.timestamp() * 1000), **{k: float(q["quote"][k]) for k in COLUMNS[1:]}})
        t = e
    df = pd.DataFrame(rows, columns=COLUMNS).drop_duplicates("time").sort_values("time")
    return df[df["time"] < int(stop.timestamp() * 1000)].reset_index(drop=True)


def load_btc_long(refresh: bool = False) -> pd.DataFrame:
    """BTC 日 K：2017-08-17 以後用幣安現貨，之前用 CoinMarketCap 接上（重疊期間收盤價中位數差約 0.7%，USDT 和美元的價差）。"""
    binance = load_spot("BTCUSDT", "1d", refresh=refresh)
    path = CACHE_DIR / "BTC_cmc_1d.csv.gz"
    if refresh or not path.exists():
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        fetch_cmc_btc().to_csv(path, index=False)
    early = pd.read_csv(path)
    early.index = pd.to_datetime(early["time"], unit="ms")
    early = early[early.index < binance.index[0]]
    return pd.concat([early, binance])
