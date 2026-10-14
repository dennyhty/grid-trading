"""即時掃描用的幣安合約持倉量（價漲倉增觀察組要用；派網沒有持倉量）。

fapi.binance.com 會擋美國的連線（GitHub Actions 的伺服器在美國，回 HTTP 451），
www.binance.com 底下的同一支 openInterestHist 不擋，所以走這裡（2026-10-13 在 Actions 上確認過）。只能查最近 30 天。
每筆的 timestamp T 是 T 前 5 分鐘的快照（和每日 metrics 檔 T-5 分那一筆相同），
所以收盤在 T 的 K 線用 timestamp = T 那筆，正好是回測用的「收盤前最後一筆」。
"""

from __future__ import annotations

import time

import pandas as pd
import requests

from .binance_history import binance_symbols

OI_HIST = "https://www.binance.com/futures/data/openInterestHist"
PERIOD = {"4H": ("4h", 4), "2H": ("2h", 2), "60M": ("1h", 1)}


def fetch_open_interest(coin: str, interval: str = "4H", session: requests.Session | None = None) -> pd.DataFrame:
    """最近 30 天的持倉量（欄位 time = API 的 timestamp（UTC）、oi = 持倉量，以幣計）。抓不到回空表。"""
    period, _ = PERIOD[interval]
    params = {"symbol": binance_symbols(coin)[-1], "period": period, "limit": 500}
    rows = None
    for i in range(3):  # 偶爾被限流或逾時，等一下再試
        try:
            resp = (session or requests).get(OI_HIST, params=params, timeout=15)
            if resp.status_code == 400:  # 幣安沒有這個合約
                break
            resp.raise_for_status()
            rows = resp.json()
            break
        except (requests.RequestException, ValueError):
            if i < 2:
                time.sleep(1.5 * 2 ** i)
    if not isinstance(rows, list) or not rows:
        return pd.DataFrame(columns=["time", "oi"])
    df = pd.DataFrame(rows)
    out = pd.DataFrame({"time": pd.to_datetime(df["timestamp"].astype("int64"), unit="ms", utc=True),
                        "oi": pd.to_numeric(df["sumOpenInterest"], errors="coerce")})
    return out.drop_duplicates("time").sort_values("time").reset_index(drop=True)


def attach_open_interest(df: pd.DataFrame, oi: pd.DataFrame, interval: str = "4H") -> pd.DataFrame:
    """把持倉量接到 K 線上：每根 K 線用 timestamp = 收盤時間的那筆（收盤前 5 分鐘的快照）。還沒有的是 NaN。"""
    _, hours = PERIOD[interval]
    out = df.copy()
    close = out["time"] + pd.Timedelta(hours=hours)
    out["oi"] = close.map(dict(zip(oi["time"], oi["oi"]))) if len(oi) else float("nan")
    out["oi"] = pd.to_numeric(out["oi"], errors="coerce")
    return out
