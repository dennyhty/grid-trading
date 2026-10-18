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
# K 線月檔的第 8~11 欄：成交額、成交筆數、主動買入量、主動買入額（訂單流研究用）
FLOW_COLUMNS = COLUMNS + ["quote_volume", "trades", "taker_buy_volume", "taker_buy_quote"]
_FLOW_USECOLS = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10]


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


def _read_zip(key: str, session: requests.Session, flow: bool = False) -> pd.DataFrame:
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
    if flow:
        df = pd.read_csv(io.StringIO(raw), header=None, usecols=_FLOW_USECOLS, names=FLOW_COLUMNS)
    else:
        df = pd.read_csv(io.StringIO(raw), header=None, usecols=range(6), names=COLUMNS)
    df = df[pd.to_numeric(df["time"], errors="coerce").notna()]  # 新的檔案有標題列
    df = df.astype(float)
    df.loc[df["time"] > 1e14, "time"] /= 1000  # 少數檔案用微秒
    df["time"] = df["time"].astype("int64")
    return df


def fetch_binance(coin: str, interval: str = "1h", refresh: bool = False, flow: bool = False) -> pd.DataFrame:
    """回傳幣安合約的完整月檔 K 線（interval 用幣安的寫法：1h、4h、1d…），time 為 UTC。

    flow=True 時多留成交額、筆數和主動買入量（FLOW_COLUMNS），另存在 {幣}_{週期}_flow.csv.gz。
    """
    name = f"{interval}_flow" if flow else interval
    path = CACHE_DIR / f"{coin}_{name}.csv.gz"
    if path.exists() and not refresh:
        return load(coin, name)
    session = requests.Session()
    parts = []
    for sym in binance_symbols(coin):
        keys = _months(sym, interval, session)
        if keys:
            with ThreadPoolExecutor(8) as ex:
                parts += list(ex.map(lambda k: _read_zip(k, session, flow), keys))
    if not parts:
        return pd.DataFrame()
    df = clean(pd.concat(parts))
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp.gz")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)
    return load(coin, name)


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


# 合約上市前用現貨補：現貨月檔從 2017-08 開始。1000 倍合約的現貨價要乘 1000 才接得上。
SPOT_SYMBOLS = {
    "SHIB": [("SHIBUSDT", 1000.0)],
    "PEPE": [("PEPEUSDT", 1000.0)],
    "POL": [("MATICUSDT", 1.0), ("POLUSDT", 1.0)],
    "RENDER": [("RNDRUSDT", 1.0), ("RENDERUSDT", 1.0)],
}
SPOT_CACHE_DIR = Path("data/binance_spot")


def fetch_spot_before(coin: str, interval: str, before_ms: int) -> pd.DataFrame:
    """幣安現貨 K 線，只抓合約第一根 K 線那個月以前的月檔（time 毫秒）。"""
    from grid_trading.spot_history import _keys, _read_zip as _read_spot

    path = SPOT_CACHE_DIR / f"{coin}_{interval}_pre.csv.gz"
    if path.exists():
        return pd.read_csv(path)
    session = requests.Session()
    first = pd.Timestamp(before_ms, unit="ms").to_period("M")
    parts = []
    for sym, scale in SPOT_SYMBOLS.get(coin, [(f"{coin}USDT", 1.0)]):
        for attempt in range(4):
            try:
                listed = _keys(f"data/spot/monthly/klines/{sym}/{interval}/", session)
                break
            except requests.RequestException:
                if attempt == 3:
                    raise
        keys = [k for k in listed if pd.Period(k[-11:-4], "M") <= first]
        for k in keys:
            d = _read_spot(k, session)
            d[["open", "high", "low", "close"]] *= scale
            d["volume"] /= scale
            parts.append(d)
    df = pd.concat(parts) if parts else pd.DataFrame(columns=COLUMNS)
    df = df[df["time"] < before_ms]
    SPOT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def load_extended(coin: str, interval: str) -> pd.DataFrame:
    """合約 K 線，前面接上合約上市前的現貨 K 線（time 為 UTC）。"""
    fut = load(coin, interval)
    if fut.empty:
        return fut
    start = int(fut["time"].iloc[0].value // 1_000_000)
    pre = fetch_spot_before(coin, interval, start)
    if pre.empty:
        return fut
    ms = (fut["time"] - pd.Timestamp(0, tz="UTC")) // pd.Timedelta(milliseconds=1)
    raw = fut.assign(time=ms)[COLUMNS]
    df = clean(pd.concat([pre[COLUMNS], raw]))
    df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True)
    return df.reset_index(drop=True)
