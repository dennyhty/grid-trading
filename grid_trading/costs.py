"""回測沒算進去的兩種成本：資金費率與滑價（以 R 為單位）。

回測（research/exit_unify.py、research/entry_research.py 的 simulate）只含每邊 0.05% 手續費。
這裡補上：

- 資金費率：幣安 USDT 永續每 8 小時（部分幣 4 小時或 1 小時）結算一次，多單在費率為正時付錢、
  空單收錢。月檔在 data/futures/um/monthly/fundingRate/{幣}/{幣}-fundingRate-YYYY-MM.zip，
  欄位 calc_time（毫秒）、funding_interval_hours、last_funding_rate，新的檔案有標題列。
- 滑價：依幣的日均成交額分級，市價進場與止損市價出場每邊吃幾個 bp。

單位換算：1R = 進場到止損的距離，risk_pct = 止損距離 / 進場價，所以名目金額的 x% 成本 = x / risk_pct R。
結果快取在 data/binance_funding/{代號}.csv.gz（time 為毫秒）。
"""

from __future__ import annotations

import io
import os
import re
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import requests

FILES = "https://data.binance.vision/"
LISTING = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
FUNDING_CACHE_DIR = Path("data/binance_funding")
FUNDING_COLUMNS = ["calc_time", "funding_interval_hours", "last_funding_rate"]

# 滑價分級（每邊 bp）：(30 日平均日成交額 USDT 下限, bp)，由大到小。
# 這些是假設值，適用小額單（風險 0.25% 的單）：市價進場、止損用市價單出場；
# 止盈（限價附近）為了簡單也照樣收，所以偏保守。
SLIP_TIERS = [
    (1e9, 1.0),
    (2e8, 2.0),
    (5e7, 4.0),
    (1e7, 8.0),
]
SLIP_DEFAULT_BPS = 15.0


# ---------------------------------------------------------------- 資金費率下載

def _months(symbol: str, session: requests.Session) -> list[str]:
    prefix = f"data/futures/um/monthly/fundingRate/{symbol}/"
    body = session.get(LISTING, params={"delimiter": "/", "prefix": prefix}, timeout=30).text
    return sorted(re.findall(r"<Key>([^<]+\.zip)</Key>", body))


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
    df = pd.read_csv(io.StringIO(raw), header=None, usecols=range(3), names=FUNDING_COLUMNS)
    df = df[pd.to_numeric(df["calc_time"], errors="coerce").notna()]  # 去掉標題列
    df = df.astype(float)
    df.loc[df["calc_time"] > 1e14, "calc_time"] /= 1000  # 以防有微秒的檔案
    return df


def _load_funding(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return pd.DataFrame({"time": pd.to_datetime(df["time"], unit="ms", utc=True), "rate": df["rate"].astype(float)})


def fetch_funding(symbol: str, refresh: bool = False) -> pd.DataFrame:
    """回傳幣安合約的歷史資金費率：time（UTC）、rate（每次結算的費率，0.0001 = 0.01%）。

    symbol 是幣安代號（例如 BTCUSDT、1000PEPEUSDT）。沒有檔案時回傳空表。
    """
    path = FUNDING_CACHE_DIR / f"{symbol}.csv.gz"
    if path.exists() and not refresh:
        return _load_funding(path)
    session = requests.Session()
    keys = _months(symbol, session)
    if not keys:
        return pd.DataFrame({"time": pd.Series(dtype="datetime64[ns, UTC]"), "rate": pd.Series(dtype=float)})
    with ThreadPoolExecutor(8) as ex:
        parts = list(ex.map(lambda k: _read_zip(k, session), keys))
    df = pd.concat(parts).drop_duplicates("calc_time", keep="last").sort_values("calc_time")
    out = pd.DataFrame({"time": df["calc_time"].astype("int64"), "rate": df["last_funding_rate"]})
    FUNDING_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp.gz")
    out.to_csv(tmp, index=False)
    os.replace(tmp, path)
    return _load_funding(path)


# ---------------------------------------------------------------- 資金費率換成 R

def _ns(x) -> np.ndarray:
    """時間（Timestamp、字串、datetime64 陣列、Series）轉成 UTC 的 int64 奈秒陣列。"""
    idx = pd.DatetimeIndex([pd.to_datetime(x, utc=True)]) if np.isscalar(x) or isinstance(x, pd.Timestamp) \
        else pd.DatetimeIndex(pd.to_datetime(x, utc=True))
    return idx.tz_localize(None).values.astype("datetime64[ns]").astype("int64")


def funding_r_many(funding_df: pd.DataFrame, entry_ts, exit_ts, sides, risk_pcts, held_fraction=1.0) -> np.ndarray:
    """向量化版：每筆交易的資金費率成本（R）。負數 = 成本。

    付款時間點落在 (entry, exit]（進場時刻不算、出場時刻算）。多單（side=+1）在費率為正時付錢，
    空單（-1）收錢：成本佔名目金額的比例 = side * 區間內費率加總，換成 R 要除以 risk_pct。
    簡化：預設整張倉位抱到出場（held_fraction=1）。實際上策略會在 TP 分批平 1/4，所以這是保守估計；
    要更準確可傳入每筆「平均持有的倉位比例」。
    """
    entry = _ns(entry_ts)
    exit_ = _ns(exit_ts)
    sides = np.asarray(sides, dtype=float)
    risk = np.asarray(risk_pcts, dtype=float)
    held = np.broadcast_to(np.asarray(held_fraction, dtype=float), entry.shape)
    if funding_df is None or len(funding_df) == 0:
        return np.zeros(len(entry))
    f = funding_df.sort_values("time")
    ft = _ns(f["time"]) // 1_000_000_000 * 1_000_000_000  # 付款時間常是整點後幾毫秒，抹到秒才不會忽有忽無地算到進場那一刻
    cum = np.concatenate([[0.0], np.cumsum(f["rate"].to_numpy(dtype=float))])
    # side="right"：time <= x 的筆數。 (entry, exit] = cum[right(exit)] - cum[right(entry)]
    paid = cum[np.searchsorted(ft, exit_, side="right")] - cum[np.searchsorted(ft, entry, side="right")]
    with np.errstate(divide="ignore", invalid="ignore"):
        out = -(sides * paid * held) / risk
    return np.where(np.isfinite(out), out, 0.0)


def funding_r(times_rate: pd.DataFrame, entry_t, exit_t, side: float, risk_pct: float, held_fraction: float = 1.0) -> float:
    """單筆交易的資金費率成本（R），負數 = 成本。見 funding_r_many。"""
    return float(funding_r_many(times_rate, [entry_t], [exit_t], [side], [risk_pct], held_fraction)[0])


# ---------------------------------------------------------------- 滑價

def slippage_bps(quote_volume_30d_usdt: float) -> float:
    """依 30 日平均日成交額（USDT）回傳每邊滑價 bp。沒有資料（NaN）時用最差一級。"""
    v = quote_volume_30d_usdt
    if v is None or not np.isfinite(v):
        return SLIP_DEFAULT_BPS
    for floor, bps in SLIP_TIERS:
        if v >= floor:
            return bps
    return SLIP_DEFAULT_BPS


def slippage_r(bps, risk_pct, sides: float = 2.0):
    """滑價成本（R），負數 = 成本。sides 是收幾邊（進出各一邊 = 2）。bps、risk_pct 可為純量或陣列。"""
    out = -(np.asarray(bps, dtype=float) / 1e4) * sides / np.asarray(risk_pct, dtype=float)
    return float(out) if out.ndim == 0 else out


# ---------------------------------------------------------------- 套用到交易表

def apply_costs(trades: pd.DataFrame, funding: dict[str, pd.DataFrame] | None = None, *, symbol_map=None,
                slip_sides: float = 2.0, held_fraction=1.0, fetch: bool = True) -> pd.DataFrame:
    """回傳加上 funding_r、slip_r、r_net（= r + funding_r + slip_r）三欄的副本。

    trades 需要欄位 symbol、side（+1/-1）、time（進場，UTC）、exit_t、r、risk_pct，
    可選 adv_usdt（30 日平均日成交額，沒有就用最差一級滑價）。
    funding 是 {symbol: funding_df}；沒給的 symbol 在 fetch=True 時用 fetch_funding 抓（有快取），
    抓不到或沒有資料的 symbol 資金費率記 0，funding_missing 為 True。
    symbol_map 可把 trades 的 symbol 轉成幣安代號（函式或 dict）。
    """
    out = trades.copy()
    n = len(out)
    out["funding_r"] = 0.0
    out["funding_missing"] = False
    cache = dict(funding or {})

    def bsym(s):
        if symbol_map is None:
            return s
        return symbol_map(s) if callable(symbol_map) else symbol_map.get(s, s)

    if n:
        times = pd.to_datetime(out["time"], utc=True)
        exits = pd.to_datetime(out["exit_t"], utc=True)
        for sym, idx in out.groupby("symbol").indices.items():
            key = bsym(sym)
            df = cache.get(key, cache.get(sym))
            if df is None and fetch:
                try:
                    df = fetch_funding(key)
                except Exception:
                    df = None
            if df is None or len(df) == 0:
                out.iloc[idx, out.columns.get_loc("funding_missing")] = True
                continue
            held = held_fraction if np.ndim(held_fraction) == 0 else np.asarray(held_fraction)[idx]
            out.iloc[idx, out.columns.get_loc("funding_r")] = funding_r_many(
                df, times.iloc[idx], exits.iloc[idx], out["side"].to_numpy()[idx], out["risk_pct"].to_numpy()[idx], held)

    if "adv_usdt" in out.columns:
        bps = out["adv_usdt"].map(slippage_bps).to_numpy(dtype=float)
    else:
        bps = np.full(n, SLIP_DEFAULT_BPS)
    out["slip_bps"] = bps
    out["slip_r"] = slippage_r(bps, out["risk_pct"].to_numpy(dtype=float), slip_sides) if n else 0.0
    out["r_net"] = out["r"] + out["funding_r"] + out["slip_r"]
    return out
