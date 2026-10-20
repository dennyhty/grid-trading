"""幣安「現貨」時點（point-in-time）幣池，給 2017-09 ~ 2019-12 這段合約還沒上市的期間用。

grid_trading.pit_universe 用的是 USDT 永續合約的月檔（2020-01 起才有），這裡用同一套做法換成現貨：

- 資料：data.binance.vision 現貨 1d 月檔（data/spot/monthly/klines/{代號}/1d/），含已下架的代號
  （VEN、BCC、HSR…在它們活著的月份照樣入選）。每天保留收盤價和成交額（quote_volume，USDT）。
- 只取 USDT 計價；排除穩定幣/法幣（沿用 pit_universe.STABLE，另加 PAX 等舊名）、槓桿代幣
  （BULL/BEAR/UP/DOWN 後綴，例如 ETHBULLUSDT、BTCUPUSDT）。
- 每個月依「上個月」成交額排名取前 N（N = min(100, 當月可選數量)），上個月至少 min_days 天有量才算；
  排名與 pit_universe.membership 同一個函式（同額用代號排序）。
- 成交量 0 的日子（停牌、下架前的平線）收盤當成沒有資料（NaN），不刪掉前後段（和合約版只留最後一段不同，
  2017-2019 很少有同代號下架又重新上架的情況）。
- 下載結果快取在 data/binance_spot_pit/spot_1d.csv.gz（data/ 不進 git），中斷後重跑會接著抓。

注意：2017-2018 年幣安的 USDT 交易對很少（2017-09 只有 BTC、ETH 等少數幾個），幣池常常不到 100 個，
早期甚至不到 10 個；使用端（research/candidates_xsec_ext.py）合格少於 10 個就空手。
"""

from __future__ import annotations

import io
import os
import re
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests

from grid_trading.binance_history import FILES, LISTING
from grid_trading.pit_universe import STABLE, TABLE_COLUMNS, _list_prefixes, _session, membership

SPOT_KLINES = "data/spot/monthly/klines/"
CACHE_DIR = Path("data/binance_spot_pit")
DAILY_CACHE = CACHE_DIR / "spot_1d.csv.gz"
DAILY_COLUMNS = ["symbol", "time", "open", "high", "low", "close", "volume", "quote_volume"]
FIRST_MONTH = "2017-08"          # 現貨月檔最早的月份
END_MONTH = "2020-01"            # 下載到這個月（含），讓 2019-12 最後一次再平衡的部位能持有到 2020-01 第一個週日
POOL_FIRST, POOL_LAST = "2017-09", "2019-12"   # 現貨幣池使用的月份範圍（2020-01 起換合約幣池）

# 舊的穩定幣代號（pit_universe.STABLE 沒有的）
SPOT_STABLE = STABLE | {"PAX", "USDSB", "USDS", "BKRW", "EURS", "USDN", "VAI", "USDK", "PAXG", "XAUT"}
LEVERAGED_SUFFIXES = ("BULL", "BEAR", "DOWN", "UP")


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def is_leveraged(base: str, bases: set[str] | None = None) -> bool:
    """槓桿代幣：BULL/BEAR 本身、*BULL、*BEAR、*DOWN；*UP 只有在去掉 UP 後是另一個 USDT 交易對的幣時才算
    （BTCUP、ETHUP…），避免把 JUP 這種名字剛好以 UP 結尾的幣排掉。bases 不給時 *UP 一律當槓桿代幣。"""
    if base in ("BULL", "BEAR"):
        return True
    for suf in ("BULL", "BEAR", "DOWN"):
        if base.endswith(suf) and len(base) > len(suf):
            return True
    if base.endswith("UP") and len(base) > 2:
        return bases is None or base[:-2] in bases
    return False


def keep_spot_symbol(symbol: str, bases: set[str] | None = None) -> bool:
    """只留 USDT 計價的現貨：排除穩定幣/法幣、槓桿代幣。bases：所有 USDT 交易對的幣（判斷 *UP 用）。"""
    if not symbol.endswith("USDT"):
        return False
    base = symbol[:-4]
    if not base or base in SPOT_STABLE:
        return False
    return not is_leveraged(base, bases)


def list_spot_usdt_symbols(session: requests.Session | None = None) -> list[str]:
    """data.binance.vision 上所有（含已下架）USDT 計價現貨代號，已排除穩定幣與槓桿代幣。"""
    session = session or requests.Session()
    names = [p[len(SPOT_KLINES):].strip("/") for p in _list_prefixes(SPOT_KLINES, session)]
    usdt = [s for s in names if s.endswith("USDT")]
    bases = {s[:-4] for s in usdt}
    return sorted(s for s in usdt if keep_spot_symbol(s, bases))


def _list_keys(symbol: str, session: requests.Session, last_month: str) -> list[str]:
    prefix = f"{SPOT_KLINES}{symbol}/1d/"
    for attempt in range(4):
        try:
            resp = session.get(LISTING, params={"delimiter": "/", "prefix": prefix}, timeout=30)
            resp.raise_for_status()
            keys = re.findall(r"<Key>([^<]+\.zip)</Key>", resp.text)
            return sorted(k for k in keys if k[-11:-4] <= last_month)
        except requests.RequestException:
            if attempt == 3:
                raise
            time.sleep(1 + attempt)
    return []


def parse_klines(raw: str, symbol: str) -> pd.DataFrame:
    """一個月檔的 CSV 文字 → 日線列（DAILY_COLUMNS，time 毫秒）。有標題列、微秒時間戳的檔案都處理。"""
    df = pd.read_csv(io.StringIO(raw), header=None, usecols=[0, 1, 2, 3, 4, 5, 7],
                     names=["time", "open", "high", "low", "close", "volume", "quote_volume"])
    df = df[pd.to_numeric(df["time"], errors="coerce").notna()].astype(float)
    df.loc[df["time"] > 1e14, "time"] /= 1000
    df["time"] = df["time"].astype("int64")
    df.insert(0, "symbol", symbol)
    return df[DAILY_COLUMNS]


def _fetch(key: str, symbol: str, session: requests.Session) -> pd.DataFrame:
    last: Exception | None = None
    for attempt in range(3):
        try:
            resp = session.get(FILES + key, timeout=60)
            resp.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
                raw = z.read(z.namelist()[0]).decode()
            return parse_klines(raw, symbol)
        except (requests.RequestException, zipfile.BadZipFile, ValueError, KeyError) as e:
            last = e
            time.sleep(1 + attempt)
    raise RuntimeError(f"download failed: {key}: {last}")


def _save(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp.gz")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def load_spot_daily(refresh: bool = False, last_month: str = END_MONTH, workers: int = 16) -> pd.DataFrame:
    """所有 USDT 現貨代號、FIRST_MONTH ~ last_month 的日線（快取，斷點續抓）。time 轉成 UTC。"""
    have = pd.DataFrame(columns=DAILY_COLUMNS)
    if DAILY_CACHE.exists() and not refresh:
        have = pd.read_csv(DAILY_CACHE)
    done = set(zip(have["symbol"], pd.to_datetime(have["time"], unit="ms").dt.strftime("%Y-%m"))) if len(have) else set()
    if not (DAILY_CACHE.exists() and not refresh and (CACHE_DIR / "complete").exists()):
        session = _session(workers)
        symbols = list_spot_usdt_symbols(session)
        _log(f"[spot-pit] {len(symbols)} 個 USDT 現貨代號（已排除穩定幣/槓桿代幣）")
        todo = []
        with ThreadPoolExecutor(workers) as ex:
            futs = {ex.submit(_list_keys, s, session, last_month): s for s in symbols}
            for fut in as_completed(futs):
                s = futs[fut]
                for k in fut.result():
                    if (s, k[-11:-4]) not in done:
                        todo.append((s, k))
        _log(f"[spot-pit] 待下載 {len(todo)} 個月檔")
        parts, failed = [have] if len(have) else [], 0
        with ThreadPoolExecutor(workers) as ex:
            futs = {ex.submit(_fetch, k, s, session): (s, k) for s, k in todo}
            for i, fut in enumerate(as_completed(futs), 1):
                try:
                    parts.append(fut.result())
                except Exception as e:  # noqa: BLE001 - 單一檔案失敗不中斷，下次再試
                    failed += 1
                    _log(f"[spot-pit] 警告：{futs[fut][1]}（{e}）")
                if i % 500 == 0:
                    _log(f"[spot-pit] {i}/{len(todo)}")
        out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=DAILY_COLUMNS)
        out = out.drop_duplicates(["symbol", "time"], keep="last").sort_values(["symbol", "time"]).reset_index(drop=True)
        _save(out, DAILY_CACHE)
        if failed == 0:
            (CACHE_DIR / "complete").write_text(f"{last_month}\n")
        have = out
    df = have.copy()
    df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True)
    return df


def monthly_table(daily: pd.DataFrame) -> pd.DataFrame:
    """日線 → 每個 (代號, 月) 的成交額加總、有量天數、第一天、最後一天（欄位同 pit_universe.TABLE_COLUMNS）。"""
    d = daily[daily["volume"] > 0].copy()
    if d.empty:
        return pd.DataFrame(columns=TABLE_COLUMNS)
    t = pd.to_datetime(d["time"], utc=True)
    d["month"] = t.dt.strftime("%Y-%m")
    d["day"] = t.dt.strftime("%Y-%m-%d")
    g = d.groupby(["symbol", "month"])
    out = g.agg(quote_volume=("quote_volume", "sum"), days=("day", "nunique"), first_day=("day", "min"),
                last_day=("day", "max")).reset_index()
    return out[TABLE_COLUMNS]


def spot_membership(table: pd.DataFrame, n: int = 100, min_days: int = 20, first: str = POOL_FIRST,
                    last: str = POOL_LAST) -> pd.DataFrame:
    """現貨幣池：每個月依上個月成交額取前 n（不足 n 就全取），只回傳 first ~ last 的月份。"""
    mem = membership(n=n, min_days=min_days, table=table)
    mem = mem[(mem["month"] >= first) & (mem["month"] <= last)]
    return mem.reset_index(drop=True)


def main() -> None:
    daily = load_spot_daily()
    table = monthly_table(daily)
    mem = spot_membership(table)
    print(f"現貨日線 {len(daily)} 列、{daily['symbol'].nunique()} 個代號；幣池 {mem['month'].min()} ~ {mem['month'].max()}")
    print(mem.groupby("month").size().to_string())


if __name__ == "__main__":
    main()
