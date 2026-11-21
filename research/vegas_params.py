"""維加斯通道換參數：EMA 12 / 144 / 169 / 288 / 338 / 576 / 676（只做研究，不改實盤規則）。2026-11-20。

Denny 圖上的維加斯有 EMA 12、144/169、288/338、576/676 三條通道。實盤只用 144/169（EMA12 也已經是條件之一：
EMA12 要和收盤一起在通道外）。這裡測換通道或加上長通道排列有沒有比較好。

事先登記（跑之前就定好，不看結果再加）：每個變體 2H、4H 各一組，共 6 × 2 = 12 組新設定。
  base     實盤規則（144/169 通道、EMA12 在通道外、MACD 柱翻轉、RSI 54-70/30-46、離通道 3 ATR 內、2H ADX<24）
  ch288    通道改成 288/338，其他一樣
  ch576    通道改成 576/676，其他一樣
  no_e12   拿掉 EMA12 條件（看 EMA12 有沒有用）
  e12_trig 觸發改成「收盤由 EMA12 錯的一側穿到對的一側」，取代 MACD 柱翻轉，其他一樣
  stack2   base ＋ 144/169 整條在 576/676 上方（空單相反）
  stack3   base ＋ 144/169 > 288/338 > 576/676 依序排列、收盤在全部之上（空單相反）
出場全部用實盤：止損 1.5 ATR、TP 1/3/5R 各 1/4、TP1 後保本、TP2 後移到 TP1、最後 1/4 追蹤 3 ATR，不設持倉上限
（PR #73 起實盤已拿掉 240 根上限）。所有變體都從第 700 根 K 棒開始（576/676 需要暖機），基準也一樣，才公平。
幣池：時點幣池（當月成交額前 100，含下市幣）＋ 滑價 + 資金費率；另看「再加日均成交額 >= 5,000 萬 U」（接近實盤）。
判斷：和 base 比（同一個帳本規則），加上驗證關卡（多重檢定 t、群集 SE、樣本內外、年度、與實盤相關性）。

先 python -m grid_trading.pit_universe（建幣池），再
python -m research.vegas_params run      → output/vp_trades.csv.gz
python -m research.vegas_params report   → output/vp_summary.csv、vp_yearly.csv，印出比較
報告：reports/signals/report_vegas_params_20261120.md
"""

from __future__ import annotations

import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

import research.entry_research as er
from grid_trading import indicators as ind
from grid_trading.costs import apply_costs, fetch_funding
from grid_trading.pit_universe import coin_of, fetch_symbol, membership
from research.pit_backtest import KLINE, PLAN, START, book, snapshot_symbols

OUT = Path("output")
VARIANTS = ("base", "ch288", "ch576", "no_e12", "e12_trig", "stack2", "stack3")
WARMUP = 700
MAX_ADX = {"2H": 24.0}
RSI_L, RSI_S, EXT = (54, 70), (30, 46), 3.0
ADV_FLOOR = 50e6
er.MAX_BARS = 10 ** 7  # 實盤沒有持倉時間上限


def signals(df: pd.DataFrame, interval: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """每個變體的訊號 K 棒位置和方向（+1/-1）。向量化，base 和 VegasTrend.check 相同。"""
    c = df["close"]
    e = {n: ind.ema(c, n).to_numpy() for n in (12, 144, 169, 288, 338, 576, 676)}
    cl = c.to_numpy()
    rsi = ind.rsi(c).to_numpy()
    h = ind.macd(c)["hist"].to_numpy()
    atr = ind.atr(df).to_numpy()
    adx = ind.adx(df).to_numpy()
    ph = np.r_[np.nan, h[:-1]]
    pc = np.r_[np.nan, cl[:-1]]
    pe12 = np.r_[np.nan, e[12][:-1]]
    flip_up, flip_dn = (ph <= 0) & (h > 0), (ph >= 0) & (h < 0)
    cross_up, cross_dn = (pc <= pe12) & (cl > e[12]), (pc >= pe12) & (cl < e[12])
    rsi_l = (rsi >= RSI_L[0]) & (rsi <= RSI_L[1])
    rsi_s = (rsi >= RSI_S[0]) & (rsi <= RSI_S[1])
    n = len(cl)
    ok = (np.arange(n) >= WARMUP) & (np.arange(n) + 1 < n) & (atr > 0)
    lim = MAX_ADX.get(interval)
    if lim is not None:
        ok &= ~(adx >= lim)

    def chan(a, b):
        return np.maximum(e[a], e[b]), np.minimum(e[a], e[b]), e[a] > e[b], e[a] < e[b]

    out = {}
    for v in VARIANTS:
        fa, fb = {"ch288": (288, 338), "ch576": (576, 676)}.get(v, (144, 169))
        top, bot, rising, falling = chan(fa, fb)
        use_e12 = v != "no_e12"
        up = (cl > top) & rising & ((e[12] > top) if use_e12 else True)
        dn = (cl < bot) & falling & ((e[12] < bot) if use_e12 else True)
        if v == "stack2":
            up &= bot > np.maximum(e[576], e[676])
            dn &= top < np.minimum(e[576], e[676])
        if v == "stack3":
            t2, b2 = np.maximum(e[288], e[338]), np.minimum(e[288], e[338])
            t3, b3 = np.maximum(e[576], e[676]), np.minimum(e[576], e[676])
            up &= (bot > t2) & (b2 > t3) & (e[288] > e[338]) & (e[576] > e[676])
            dn &= (top < b2) & (t2 < b3) & (e[288] < e[338]) & (e[576] < e[676])
        tl, ts = (cross_up, cross_dn) if v == "e12_trig" else (flip_up, flip_dn)
        lg = ok & up & tl & rsi_l & (cl - top <= EXT * atr)
        sh = ok & dn & ts & rsi_s & (bot - cl <= EXT * atr)
        idx = np.flatnonzero(lg | sh)
        out[v] = (idx, np.where(lg[idx], 1, -1))
    return out


def _work(args):
    sym, interval, fee = args
    for k in range(6):
        try:
            df = fetch_symbol(sym, KLINE[interval])
            break
        except Exception:
            if k == 5:
                raise
            time.sleep(5 * (k + 1))
    if len(df) < WARMUP + 50:
        return None
    df = df.reset_index(drop=True)
    arr = {"open": df["open"].to_numpy(float), "high": df["high"].to_numpy(float), "low": df["low"].to_numpy(float),
           "close": df["close"].to_numpy(float), "atr": ind.atr(df).to_numpy(float)}
    t = pd.to_datetime(df["time"], utc=True)
    bpd = 6 if interval == "4H" else 12
    adv = ((df["volume"] * df["close"]).rolling(30 * bpd, min_periods=bpd).sum() / 30).to_numpy()
    er.PLANS = {interval: PLAN}
    parts = []
    for v, (idx, ds) in signals(df, interval).items():
        if not len(idx):
            continue
        s = er.simulate(arr, idx, ds, interval, fee, 0.0)
        parts.append(pd.DataFrame({
            "variant": v, "symbol": sym, "coin": coin_of(sym), "interval": interval, "i": idx, "side": ds,
            "time": t.iloc[idx + 1].to_numpy(), "exit_t": t.iloc[np.minimum(s["j"], len(t) - 1)].to_numpy(),
            "r": s["r"], "hit": s["hit"], "kind": s["kind"], "j": s["j"], "risk_pct": s["risk_pct"],
            "adv_usdt": adv[idx]}))
    return pd.concat(parts) if parts else None


def run(fee: float = 0.0005, workers: int = 4) -> pd.DataFrame:
    mem = membership(n=100)
    mem["month"] = mem["month"].astype(str)
    syms = sorted(set(mem.loc[mem["month"] >= "2020-01", "symbol"]) | snapshot_symbols())
    jobs = [(s, iv, fee) for iv in ("4H", "2H") for s in syms]
    print(f"{len(jobs)} 個工作", file=sys.stderr)
    parts = []
    with ProcessPoolExecutor(workers) as ex:
        for k, got in enumerate(ex.map(_work, jobs, chunksize=4)):
            if got is not None:
                parts.append(got)
            if k % 200 == 0:
                print(f"  {k}/{len(jobs)}", file=sys.stderr, flush=True)
    tr = pd.concat(parts, ignore_index=True)
    for c in ("time", "exit_t"):
        tr[c] = pd.to_datetime(tr[c], utc=True)
    tr = tr[tr["time"] >= START]
    pit = set(zip(mem["symbol"], mem["month"]))
    tr["in_pit"] = [(s, m) in pit for s, m in zip(tr["symbol"], tr["time"].dt.strftime("%Y-%m"))]
    tr["in_snapshot"] = tr["symbol"].isin(snapshot_symbols())
    tr["liquid"] = tr["in_pit"] & (tr["adv_usdt"] >= ADV_FLOOR)
    return apply_costs(tr, funding=_funding(sorted(tr["symbol"].unique())))


def _funding(syms: list[str]) -> dict[str, pd.DataFrame]:
    """資金費率先抓好（有重試），免得網路斷一下就被記成 0。"""
    def one(s):
        for k in range(6):
            try:
                return s, fetch_funding(s)
            except Exception:
                time.sleep(5 * (k + 1))
        print(f"  資金費率抓不到：{s}", file=sys.stderr)
        return s, None
    with ThreadPoolExecutor(8) as ex:
        got = dict(ex.map(one, syms))
    return {s: d for s, d in got.items() if d is not None}


def books(tr: pd.DataFrame, pool: str) -> dict[str, pd.DataFrame]:
    return {v: book(tr[tr["variant"] == v], pool) for v in VARIANTS}


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    OUT.mkdir(exist_ok=True)
    if cmd == "run":
        tr = run()
        tr.to_csv(OUT / "vp_trades.csv.gz", index=False)
        print(tr.groupby(["variant", "interval"]).size())
    elif cmd == "report":
        from research.vegas_params_report import report
        report()


if __name__ == "__main__":
    main()
