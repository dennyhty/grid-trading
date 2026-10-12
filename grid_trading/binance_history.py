"""幣安 USDT 永續合約的歷史 K 線（data.binance.vision 的月檔），用來做比派網更長的回測。

派網的 K 線最早只到 2023-01，幣安的月檔從 2020-01 開始，可以涵蓋 2022 年的熊市。
主流幣在兩邊的價格幾乎一樣，回測結果可以拿來參考在派網開單。

- 月檔只到上個月，所以資料停在上個月底。
- 改名的幣會接上舊代號（MATIC→POL、RNDR→RENDER，都是 1:1 換幣）。
- 下架後的月檔會繼續出現成交量 0 的平線，先刪掉。刪完後中間斷超過 10 天的（下架後又重新上架，
  可能已經是別的幣，例如 LIT），只取最後一段連續的資料。
- 結果快取在 data/binance/{幣}_{週期}.csv.gz，欄位和派網的快取相同（time 是毫秒）。
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

FILES = "https://data.binance.vision/"
LISTING = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
CACHE_DIR = Path("data/binance")
# 幣安合約代號和派網不同的幣：小單價的幣以 1000 枚為單位，改名的幣接上舊代號（由舊到新）
SYMBOLS = {
    "SHIB": ["1000SHIBUSDT"],
    "PEPE": ["1000PEPEUSDT"],
    "POL": ["MATICUSDT", "POLUSDT"],
    "RENDER": ["RNDRUSDT", "RENDERUSDT"],
}
COLUMNS = ["time", "open", "high", "low", "close", "volume"]


def binance_symbols(coin: str) -> list[str]:
    return SYMBOLS.get(coin, [f"{coin}USDT"])


def _months(symbol: str, interval: str, session: requests.Session) -> list[str]:
    prefix = f"data/futures/um/monthly/klines/{symbol}/{interval}/"
    body = session.get(LISTING, params={"delimiter": "/", "prefix": prefix}, timeout=30).text
    keys = sorted(re.findall(r"<Key>([^<]+\.zip)</Key>", body))
    if not keys:
        return []
    # 只取最後一段連續的月份
    periods = [pd.Period(k[-11:-4], "M") for k in keys]
    start = 0
    for i in range(1, len(periods)):
        if (periods[i] - periods[i - 1]).n > 1:
            start = i
    return keys[start:]


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
    df = df[pd.to_numeric(df["time"], errors="coerce").notna()]  # 新的檔案有標題列
    df = df.astype(float)
    df.loc[df["time"] > 1e14, "time"] /= 1000  # 少數檔案用微秒
    df["time"] = df["time"].astype("int64")
    return df


def fetch_binance(coin: str, interval: str = "1h", refresh: bool = False) -> pd.DataFrame:
    """回傳幣安合約的完整月檔 K 線（interval 用幣安的寫法：1h、4h、1d…），time 為 UTC。"""
    path = CACHE_DIR / f"{coin}_{interval}.csv.gz"
    if path.exists() and not refresh:
        return load(coin, interval)
    session = requests.Session()
    parts = []
    for sym in binance_symbols(coin):
        keys = _months(sym, interval, session)
        if keys:
            with ThreadPoolExecutor(8) as ex:
                parts += list(ex.map(lambda k: _read_zip(k, session), keys))
    if not parts:
        return pd.DataFrame()
    df = clean(pd.concat(parts))
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp.gz")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)
    return load(coin, interval)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """刪掉成交量 0 的 K 線（下架後的平線），只留最後一段沒有斷超過 10 天的資料。time 是毫秒。"""
    df = df[df["volume"] > 0].drop_duplicates("time", keep="last").sort_values("time").reset_index(drop=True)
    gaps = (df["time"].diff() > 10 * 86_400_000).to_numpy().nonzero()[0]
    return df.iloc[gaps[-1]:].reset_index(drop=True) if len(gaps) else df


def load(coin: str, interval: str) -> pd.DataFrame:
    path = CACHE_DIR / f"{coin}_{interval}.csv.gz"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True)
    return df.reset_index(drop=True)
