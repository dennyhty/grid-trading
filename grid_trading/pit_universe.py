"""時點（point-in-time）幣池：消除回測的存活者偏差。

universe.TOP100 是 2026-10-12 的快照，只含「今天還活著的大幣」，拿它回測 2020~2025 年等於偷看未來。
這裡改成每個月用「上個月」的成交額重新排名，取幣安 USDT 永續合約的前 N 名，下架的幣（LUNA、FTT、SRM…）
在它們活著的月份照樣入選。

- 成交額來自 data.binance.vision 的 1d 月檔（um/monthly/klines），每個 (代號, 月) 加總 quote_volume。
- 上市不滿 min_days 個交易日的月份不算，新幣要等上市後第一個夠長的月份結束才會進池（下個月才入選）。
- 只取 *USDT 的永續合約：排除 BUSD/USDC 計價、帶 "_" 的季度合約、穩定幣/法幣、指數商品。
- 用法：membership() 回傳 (symbol, month) 清單，進場時間 t 的交易只在 (symbol, t 所在月) 屬於清單時才算數。
- 結果快取在 data/binance/pit_monthly_volume.csv（斷點可續傳），單一代號 K 線在 data/binance_sym/。
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
from requests.adapters import HTTPAdapter

from grid_trading.binance_history import COLUMNS, FILES, LISTING, _months, _read_zip, clean

UM_KLINES = "data/futures/um/monthly/klines/"
CACHE_DIR = Path("data/binance")
VOLUME_CACHE = CACHE_DIR / "pit_monthly_volume.csv"
SYM_CACHE_DIR = Path("data/binance_sym")
TABLE_COLUMNS = ["symbol", "month", "quote_volume", "days", "first_day", "last_day"]

# 穩定幣和法幣：不是要交易的標的
STABLE = {
    "USDC", "BUSD", "TUSD", "FDUSD", "USDP", "DAI", "USDE", "EUR", "GBP", "AEUR", "USD1", "RLUSD",
    "USDS", "PYUSD", "UST", "USTC", "SUSD", "GUSD", "USDD", "XUSD", "BFUSD", "USDSB", "EURI",
    "TRY", "BRL", "AUD", "RUB", "UAH", "NGN", "ARS", "PLN", "RON", "CZK", "ZAR", "IDRT", "BIDR", "BVND",
}
# 指數商品（BTCDOM 這類是一籃子指數，不是單一幣），另外所有以 DOM 結尾的也排除
INDEXES = {"BTCDOM", "DEFI", "FOOTBALL", "BLUEBIRD"}


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _session(workers: int = 16) -> requests.Session:
    s = requests.Session()
    adapter = HTTPAdapter(pool_connections=workers, pool_maxsize=workers)
    s.mount("https://", adapter)
    return s


def coin_of(symbol: str) -> str:
    """去掉 USDT 後綴。1000SHIB、1000000MOG 這類是獨立的合約（單位不同），保持原樣。"""
    return symbol[:-4] if symbol.endswith("USDT") else symbol


# 幣安 2025 年起的美股、ETF、商品永續（不是加密貨幣；之前研究過美股/商品沒有優勢），從幣池排除
TRADFI = {
    "AAOI", "AAPL", "AMD", "AMZN", "ARM", "AXTI", "COIN", "CRCL", "DRAM", "EWY", "GOOGL", "HOOD", "INTC", "KORU",
    "LITE", "META", "MRVL", "MSFT", "MSTR", "MU", "NBIS", "NVDA", "PLTR", "QQQ", "SAMSUNG", "SKHY", "SKHYNIX",
    "SNDK", "SNXX", "SOXL", "SOXS", "SPCX", "SPY", "TSLA", "XAG", "XAU", "XAUT", "PAXG", "NATGAS", "CL", "BZ",
}


def keep_symbol(symbol: str) -> bool:
    """只留 *USDT 永續：排除季度合約（含 "_"）、穩定幣/法幣、指數商品。"""
    if not symbol.endswith("USDT") or "_" in symbol:
        return False
    base = coin_of(symbol)
    if not base or base in STABLE or base in INDEXES or base in TRADFI or base.endswith("DOM"):
        return False
    return True


def _list_prefixes(prefix: str, session: requests.Session) -> list[str]:
    """列出 S3 prefix 底下的子目錄（分頁：IsTruncated 為 true 時用 NextMarker，沒有就用最後一個 Prefix）。"""
    out: list[str] = []
    marker = ""
    while True:
        params = {"delimiter": "/", "prefix": prefix}
        if marker:
            params["marker"] = marker
        for attempt in range(4):
            try:
                resp = session.get(LISTING, params=params, timeout=30)
                resp.raise_for_status()
                break
            except requests.RequestException:
                if attempt == 3:
                    raise
                time.sleep(1 + attempt)
        body = resp.text
        # 第一個 <Prefix> 是請求本身的 prefix，子目錄都包在 <CommonPrefixes> 裡
        out += re.findall(r"<CommonPrefixes><Prefix>([^<]+)</Prefix></CommonPrefixes>", body)
        if "<IsTruncated>true</IsTruncated>" not in body:
            return out
        nxt = re.search(r"<NextMarker>([^<]+)</NextMarker>", body)
        marker = nxt.group(1) if nxt else out[-1]


def list_um_symbols(session: requests.Session | None = None) -> list[str]:
    """幣安 USDT 永續合約（含已下架）的所有代號，已排除季度合約、穩定幣和指數商品。"""
    session = session or requests.Session()
    names = [p[len(UM_KLINES):].strip("/") for p in _list_prefixes(UM_KLINES, session)]
    return sorted({s for s in names if keep_symbol(s)})


def _list_months(symbol: str, session: requests.Session) -> list[str]:
    """該代號所有 1d 月檔的 key（不像 _months 只取最後一段連續的）。"""
    prefix = f"{UM_KLINES}{symbol}/1d/"
    for attempt in range(4):
        try:
            resp = session.get(LISTING, params={"delimiter": "/", "prefix": prefix}, timeout=30)
            resp.raise_for_status()
            return sorted(re.findall(r"<Key>([^<]+\.zip)</Key>", resp.text))
        except requests.RequestException:
            if attempt == 3:
                raise
            time.sleep(1 + attempt)
    return []


def summarize_month(raw: str) -> dict:
    """把一個月的 1d K 線 CSV 文字加總成 quote_volume、有量的天數、第一天和最後一天。"""
    df = pd.read_csv(io.StringIO(raw), header=None, usecols=[0, 5, 7], names=["time", "volume", "quote_volume"])
    df = df[pd.to_numeric(df["time"], errors="coerce").notna()].astype(float)  # 新的檔案有標題列
    df.loc[df["time"] > 1e14, "time"] /= 1000  # 少數檔案用微秒
    live = df[df["volume"] > 0]
    day = lambda ms: pd.Timestamp(int(ms), unit="ms").strftime("%Y-%m-%d")  # noqa: E731
    return {
        "quote_volume": float(live["quote_volume"].sum()),
        "days": int(len(live)),
        "first_day": day(live["time"].min()) if len(live) else "",
        "last_day": day(live["time"].max()) if len(live) else "",
    }


def _fetch_month(key: str, session: requests.Session) -> dict:
    last: Exception | None = None
    for attempt in range(3):
        try:
            resp = session.get(FILES + key, timeout=60)
            resp.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
                raw = z.read(z.namelist()[0]).decode()
            return summarize_month(raw)
        except (requests.RequestException, zipfile.BadZipFile, ValueError, KeyError) as e:
            last = e
            time.sleep(1 + attempt)
    raise RuntimeError(f"download failed: {key}: {last}")


def _save(table: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    table.to_csv(tmp, index=False)
    os.replace(tmp, path)


def _load_table(path: Path = VOLUME_CACHE) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=TABLE_COLUMNS)
    df = pd.read_csv(path, dtype={"month": str}, keep_default_na=False)
    for c in ("first_day", "last_day"):
        df[c] = df[c].astype(str)
    df = df[df["symbol"].map(keep_symbol)]  # 排除清單改過時，舊快取也照新的清單篩
    return df[TABLE_COLUMNS].reset_index(drop=True)


def monthly_volume(refresh: bool = False, workers: int = 16) -> pd.DataFrame:
    """每個 (代號, 月) 的 1d 月檔加總：symbol, month('YYYY-MM'), quote_volume, days, first_day, last_day。

    已經抓過的月份不會重抓（月檔是定案的），中斷後重跑會接著抓；refresh=True 才整個重建。
    下載連續失敗 3 次的檔案會跳過並警告，下次執行會再試。
    """
    session = _session(workers)
    table = pd.DataFrame(columns=TABLE_COLUMNS) if refresh else _load_table()
    done = set(zip(table["symbol"], table["month"]))

    symbols = list_um_symbols(session)
    _log(f"[pit] {len(symbols)} 個 USDT 永續代號，已有 {len(done)} 筆 (代號, 月)")

    def list_one(sym: str) -> tuple[str, list[str]]:
        return sym, _list_months(sym, session)

    todo: list[tuple[str, str, str]] = []  # (symbol, month, key)
    with ThreadPoolExecutor(workers) as ex:
        for i, fut in enumerate(as_completed([ex.submit(list_one, s) for s in symbols]), 1):
            try:
                sym, keys = fut.result()
            except requests.RequestException as e:
                _log(f"[pit] 警告：列不出檔案 {e}")
                continue
            for k in keys:
                month = k[-11:-4]
                if (sym, month) not in done:
                    todo.append((sym, month, k))
            if i % 200 == 0:
                _log(f"[pit] 列檔 {i}/{len(symbols)}")
    _log(f"[pit] 待下載 {len(todo)} 個月檔")
    if not todo:
        return table.sort_values(["symbol", "month"]).reset_index(drop=True)

    rows: list[dict] = []
    failed = 0
    t0 = time.time()

    def flush() -> pd.DataFrame:
        new = pd.DataFrame(rows, columns=TABLE_COLUMNS)
        out = pd.concat([table, new], ignore_index=True) if len(new) else table
        out = out.drop_duplicates(["symbol", "month"], keep="last").sort_values(["symbol", "month"])
        out = out.reset_index(drop=True)
        _save(out, VOLUME_CACHE)
        return out

    with ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(_fetch_month, k, session): (sym, month) for sym, month, k in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            sym, month = futs[fut]
            try:
                rows.append({"symbol": sym, "month": month, **fut.result()})
            except Exception as e:  # noqa: BLE001 - 單一檔案失敗不該中斷整個建表
                failed += 1
                _log(f"[pit] 警告：略過 {sym} {month}（{e}）")
            if i % 500 == 0:
                _log(f"[pit] 下載 {i}/{len(todo)}（失敗 {failed}），{time.time() - t0:.0f}s")
                flush()
    out = flush()
    _log(f"[pit] 完成：{len(rows)} 筆新增，失敗 {failed}，{time.time() - t0:.0f}s")
    return out


def _prev_month(month: pd.Period | str) -> str:
    return str(pd.Period(month, "M") - 1)


def _rank(table: pd.DataFrame, prev: str, n: int, min_days: int) -> list[str]:
    t = table[(table["month"] == prev) & (table["days"] >= min_days) & (table["quote_volume"] > 0)]
    t = t.sort_values(["quote_volume", "symbol"], ascending=[False, True])
    return t["symbol"].head(n).tolist()


def universe(month: pd.Period | str, n: int = 100, min_days: int = 20, table: pd.DataFrame | None = None) -> list[str]:
    """month 這個月的幣池：依上個月成交額排名前 n，上個月至少有 min_days 天有量。回傳幣安代號。"""
    if table is None:
        table = _load_table() if VOLUME_CACHE.exists() else monthly_volume()
    return _rank(table, _prev_month(month), n, min_days)


def membership(n: int = 100, min_days: int = 20, table: pd.DataFrame | None = None) -> pd.DataFrame:
    """所有月份的幣池成員：欄位 symbol、month('YYYY-MM')。

    涵蓋從第一個資料月的下個月，到最後一個資料月的下個月（最後一個月檔結束後，下個月的池子已經確定）。
    """
    if table is None:
        table = _load_table() if VOLUME_CACHE.exists() else monthly_volume()
    t = table[(table["days"] >= min_days) & (table["quote_volume"] > 0)].copy()
    if t.empty:
        return pd.DataFrame(columns=["symbol", "month"])
    # 同額時用代號排序，和 universe() 的結果一致
    t = t.sort_values(["month", "quote_volume", "symbol"], ascending=[True, False, True])
    t["rank"] = t.groupby("month").cumcount() + 1
    t = t[t["rank"] <= n]
    t["month"] = (pd.PeriodIndex(t["month"], freq="M") + 1).astype(str)
    return t[["symbol", "month"]].sort_values(["month", "symbol"]).reset_index(drop=True)


def fetch_symbol(symbol: str, interval: str, refresh: bool = False) -> pd.DataFrame:
    """用幣安原始代號抓合約 K 線（不做幣種對應和改名接續），欄位 time(UTC)、open、high、low、close、volume。

    和 fetch_binance 一樣用 clean() 刪掉成交量 0 的平線並只留最後一段連續資料；快取在
    data/binance_sym/{代號}_{週期}.csv.gz。
    """
    path = SYM_CACHE_DIR / f"{symbol}_{interval}.csv.gz"
    if not path.exists() or refresh:
        session = _session(8)
        keys = _months(symbol, interval, session)
        if not keys:
            return pd.DataFrame(columns=COLUMNS)
        with ThreadPoolExecutor(8) as ex:
            parts = list(ex.map(lambda k: _read_zip(k, session), keys))
        df = clean(pd.concat(parts))
        SYM_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp.gz")
        df.to_csv(tmp, index=False)
        os.replace(tmp, path)
    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True)
    return df.reset_index(drop=True)


def main() -> None:
    from grid_trading.binance_history import binance_symbols
    from grid_trading.universe import TOP100

    t0 = time.time()
    table = monthly_volume()
    print(f"成交額表：{len(table)} 筆，{table['symbol'].nunique()} 個代號，"
          f"{table['month'].min()} ~ {table['month'].max()}，耗時 {time.time() - t0:.0f}s")
    mem = membership(100, table=table)
    for m in ["2020-06", "2022-06", "2024-06", "2026-09"]:
        u = universe(m, 100, table=table)
        print(f"\n{m}：{len(u)} 個 → {' '.join(u[:15])}")

    have = set(table["symbol"])
    top = {s for c in TOP100 for s in binance_symbols(c)}
    on_binance = [c for c in TOP100 if any(s in have for s in binance_symbols(c))]
    print(f"\nTOP100 快照（{len(TOP100)} 個）裡幣安有永續合約的：{len(on_binance)} 個。每年一月在 PIT 前 100 的數量：")
    for y in range(2020, 2027):
        month = f"{y}-01"
        pool = set(mem.loc[mem["month"] == month, "symbol"])
        if not pool:
            continue
        hit = sum(1 for c in on_binance if any(s in pool for s in binance_symbols(c)))
        print(f"  {month}：池 {len(pool)} 個，其中 TOP100 {hit} 個（{len(pool & top)} 個代號），"
              f"池裡已下架/不在快照的 {len(pool - top)} 個")


if __name__ == "__main__":
    main()
