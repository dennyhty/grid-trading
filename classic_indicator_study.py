"""基礎指標策略研究（只做分析，不改實盤規則）。

把常見的單純指標（均線、EMA、布林帶、維加斯通道、RSI、KD、MFI、威廉、CCI、SMI、MACD、DMI、超級趨勢、PSAR、
唐奇安、肯特納、一目均衡表、HMA、OBV、量價背離…）各自寫成進出場規則，用同一套資料、手續費和出場方式回測，
再和隨機進場、實盤的維加斯趨勢比較。

python classic_indicator_study.py --fetch                  下載幣安 4h、1h 月檔 K 線（data/binance/，已下載的跳過）
python classic_indicator_study.py --fetch-funding          下載幣安資金費率月檔（data/binance/funding/）
python classic_indicator_study.py --run --tf 4H             第一步：每個策略的訊號和出場結果（2H 同理；加 --slip 0.0005 是滑價版）
python classic_indicator_study.py --analyze                 第二步：統計、樣本外、過度擬合、相關性、勝率與期望值
python classic_indicator_study.py --market                  市場本身的性質：變異數比、時間序列動能
python classic_indicator_study.py --mechanism               獲利來自大盤還是個別幣；市場寬度、同時出訊號的幣數
python classic_indicator_study.py --robust                  區塊重抽、多重檢定、幣種子集、同時持倉（--analyze、--mechanism 之後）
python classic_indicator_study.py --run --pionex --tf 4H    派網 K 線複核（4H 約 2023 年起、2H 約 14 個月）
python classic_indicator_study.py --pionex-check            派網和幣安同一段時間、同一批幣對照
python classic_indicator_study.py --regime                  BTC 200 日均線多空過濾（事先定好的單一規則，50 日線當對照）
python classic_indicator_study.py --cap                     每天最多開 1/2/3/5 張時的結果（手動下單做不了每個訊號）
python classic_indicator_study.py --export DIR --prefix P   彙總表和各項結果存成 DIR/P*.csv（utf-8-sig，Excel 可直接開）

需要 numba（pip install numba，只有研究程式用到）。

做法：
- 資料：幣安 USDT 永續月檔 2020-01 ~ 2026-09，派網有合約的市值前 100 名（84 個幣安有合約），2H 由 1H 合成。
- 訊號：第 i 根收盤時成立，用收盤價進場，第 i+1 根開始算出場；只用第 i 根（含）以前的資料。
- 出場（每個策略都跑）：
  - live：和實盤維加斯相同（止損 1.5 ATR；4H 1R 平一半、保本、剩下 3 ATR 追蹤；2H 1/3/5R 各 1/4、最後 1/4 追蹤）。
  - tp0.5 / tp1 / tp2 / tp3：止損 1.5 ATR、單一停利 0.5/1/2/3R，最多 120 根（用來看勝率和期望值的關係）。
  - natural：指標本身的出場（順勢類：反向訊號；反轉類：回到均值），止損 2 ATR；沒有公認出場的策略不跑。
  - 固定天數報酬（1/3/7/14 天，不設停損停利，以 ATR 計）：看訊號本身有沒有方向資訊。
- 隨機進場對照：每一根 K 棒都當成進場點做多、做空各模擬一次，取「同幣、同方向、同一年（或同一月）」的平均，
  當作這筆交易的對照組。差值就是進場時機的價值。
- 同幣同時只拿一張（前一張出場前的新訊號略過），同一根 K 棒同時碰到止損和停利算止損先到，手續費每邊 0.05%。

輸出在 output/classic/。
"""

from __future__ import annotations

import argparse
import io
import json
import math
import re
import sys
import zipfile
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from numba import njit

from grid_trading import classic_indicators as ci
from grid_trading import indicators as ind
from grid_trading.binance_history import FILES, LISTING, binance_symbols, load
from grid_trading.history import resample
from grid_trading.signals.strategy import LONG
from grid_trading.signals.vegas_trend import VegasTrend
from grid_trading.universe import TOP100

OUT = Path("output/classic")
FUNDING_DIR = Path("data/binance/funding")
SPLIT = pd.Timestamp("2024-01-01")  # 時間一律用 UTC、不帶時區
FEE = 0.0005
MIN_STOP_PCT = 0.001
MAX_BARS = 240
FIXED_MAX_BARS = 120
MR_MAX_BARS = 30
WARMUP = 200
LIVE_ATR = 1.5
NATURAL_ATR = 2.0
BAR_HOURS = {"4H": 4, "2H": 2}
LIVE_PLANS = {"4H": ((1.0,), (0.5,), 3.0), "2H": ((1.0, 3.0, 5.0), (0.25, 0.25, 0.25), 3.0)}
FIXED_TPS = (0.5, 1.0, 2.0, 3.0)
HORIZON_DAYS = (1, 3, 7, 14)
EXITS = ["live"] + [f"tp{k:g}" for k in FIXED_TPS]


# ---------------------------------------------------------------- 資料

def load_frame(coin: str, tf: str, source: str = "binance") -> pd.DataFrame:
    """幣安月檔（主）或派網 K 線（輔，4H 約 2023 年起、1H 約 14 個月）。2H 都由 1H 合成。"""
    if source == "pionex":
        from grid_trading.history import fetch_history
        sym = f"{coin}_USDT_PERP"
        try:
            if tf == "4H":
                df = fetch_history(sym, "4H", 1300)
            else:
                df = fetch_history(sym, "60M", 420)
                if len(df):
                    df = resample(df, "2h", origin="epoch")
        except Exception:  # 派網沒有這個合約或暫時連不上
            return pd.DataFrame()
        return df.reset_index(drop=True)
    if tf == "4H":
        return load(coin, "4h").reset_index(drop=True)
    df = load(coin, "1h")
    if len(df):
        df = resample(df, "2h", origin="epoch")
    return df.reset_index(drop=True)


def fetch_funding(coin: str, refresh: bool = False) -> pd.DataFrame:
    """幣安資金費率月檔（calc_time 毫秒、funding_interval_hours、last_funding_rate），快取在 data/binance/funding/。"""
    path = FUNDING_DIR / f"{coin}.csv.gz"
    if path.exists() and not refresh:
        return pd.read_csv(path)
    session = requests.Session()
    parts = []
    for sym in binance_symbols(coin):
        prefix = f"data/futures/um/monthly/fundingRate/{sym}/"
        body = session.get(LISTING, params={"delimiter": "/", "prefix": prefix}, timeout=30).text
        keys = sorted(re.findall(r"<Key>([^<]+\.zip)</Key>", body))

        def read(key):
            for _ in range(3):
                try:
                    resp = session.get(FILES + key, timeout=60)
                    resp.raise_for_status()
                    break
                except requests.RequestException:
                    continue
            else:
                return pd.DataFrame()
            with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
                raw = z.read(z.namelist()[0]).decode()
            d = pd.read_csv(io.StringIO(raw))
            if "calc_time" not in d.columns:  # 少數舊檔沒有標題列
                d = pd.read_csv(io.StringIO(raw), header=None, names=["calc_time", "funding_interval_hours",
                                                                        "last_funding_rate"])
            return d

        with ThreadPoolExecutor(8) as ex:
            parts += list(ex.map(read, keys))
    parts = [p for p in parts if len(p)]
    if not parts:
        return pd.DataFrame()
    df = pd.concat(parts)
    df = df[pd.to_numeric(df["calc_time"], errors="coerce").notna()].astype(float)
    df = df.drop_duplicates("calc_time").sort_values("calc_time")
    FUNDING_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


# ---------------------------------------------------------------- 指標與策略

def _prev(a: np.ndarray, k: int = 1) -> np.ndarray:
    if a.dtype == bool:
        return np.r_[np.zeros(k, bool), a[:-k]]
    return np.r_[np.full(k, np.nan), a[:-k]]


def xup(a, b) -> np.ndarray:
    """a 由下往上穿過 b（b 可以是常數）。"""
    b = np.broadcast_to(np.asarray(b, float), a.shape)
    return (a > b) & (_prev(a) <= _prev(np.array(b)))


def xdn(a, b) -> np.ndarray:
    b = np.broadcast_to(np.asarray(b, float), a.shape)
    return (a < b) & (_prev(a) >= _prev(np.array(b)))


def fresh(m: np.ndarray) -> np.ndarray:
    """連續成立的第一根。"""
    return m & ~_prev(m)


def features(df: pd.DataFrame, tf: str) -> dict:
    """一次算好所有策略要用的指標（numpy 陣列）。"""
    bpd = 24 // BAR_HOURS[tf]
    c, h, l, o, v = (df[k].to_numpy(float) for k in ("close", "high", "low", "open", "volume"))
    cs = df["close"]
    x = {"c": c, "h": h, "l": l, "o": o, "v": v, "bpd": bpd}
    x["atr"] = ind.atr(df).to_numpy(float)
    for n in (5, 10, 20, 30, 50, 100, 200):
        x[f"sma{n}"] = ci.sma(cs, n).to_numpy(float)
    for n in (9, 12, 20, 21, 26, 50, 144, 169, 200):
        x[f"ema{n}"] = ind.ema(cs, n).to_numpy(float)
    x["rsi"] = ind.rsi(cs).to_numpy(float)
    x["rsi2"] = ind.rsi(cs, 2).to_numpy(float)
    m = ind.macd(cs)
    x["macd"], x["macd_sig"], x["hist"] = (m[k].to_numpy(float) for k in ("macd", "signal", "hist"))
    for k in (2.0, 2.5):
        bb = ci.bollinger(cs, 20, k)
        tag = "" if k == 2.0 else "25"
        x[f"bb_mid{tag}"], x[f"bb_up{tag}"], x[f"bb_lo{tag}"] = (bb[q].to_numpy(float) for q in ("mid", "upper", "lower"))
        if k == 2.0:
            w = bb["width"]
            x["bb_width"] = w.to_numpy(float)
            x["bb_width_q20"] = w.rolling(120).quantile(0.2).to_numpy(float)
    kdf = ci.kd(df)
    x["k"], x["d"] = kdf["k"].to_numpy(float), kdf["d"].to_numpy(float)
    x["wr"] = ci.williams_r(df).to_numpy(float)
    x["cci"] = ci.cci(df).to_numpy(float)
    x["mfi"] = ci.mfi(df).to_numpy(float)
    ob = ci.obv(df)
    x["obv"] = ob.to_numpy(float)
    x["obv_e20"], x["obv_e50"] = ind.ema(ob, 20).to_numpy(float), ind.ema(ob, 50).to_numpy(float)
    dm = ci.dmi(df)
    x["pdi"], x["mdi"], x["adx"] = (dm[k].to_numpy(float) for k in ("plus_di", "minus_di", "adx"))
    for n, mult in ((10, 3.0), (10, 2.0)):
        x[f"st_{n}_{mult:g}"] = ci.supertrend(df, n, mult).to_numpy(float)
    x["psar"] = ind.psar(df).to_numpy(float)
    for n in (10, 20, 55):
        dc = ci.donchian(df, n)
        x[f"dc{n}_hi"], x[f"dc{n}_lo"] = dc["high"].to_numpy(float), dc["low"].to_numpy(float)
    kc = ci.keltner(df)
    x["kc_mid"], x["kc_up"], x["kc_lo"] = (kc[q].to_numpy(float) for q in ("mid", "upper", "lower"))
    ic = ci.ichimoku(df)
    x["kijun"] = ic["kijun"].to_numpy(float)
    x["cloud_top"] = np.fmax(ic["span_a"].to_numpy(float), ic["span_b"].to_numpy(float))
    x["cloud_bot"] = np.fmin(ic["span_a"].to_numpy(float), ic["span_b"].to_numpy(float))
    x["cloud_top"][~np.isfinite(ic["span_b"].to_numpy(float))] = np.nan
    x["cloud_bot"][~np.isfinite(ic["span_b"].to_numpy(float))] = np.nan
    x["hma"] = ind.hma(cs, 55).to_numpy(float)
    sm = ci.smi(df)
    x["smi"], x["smi_sig"] = sm["smi"].to_numpy(float), sm["signal"].to_numpy(float)
    vol = df["volume"]
    x["vol_ma20_prev"] = vol.rolling(20).mean().shift(1).to_numpy(float)
    x["vol5"] = vol.rolling(5).mean().to_numpy(float)
    lag = 30 * bpd
    x["roc30d"] = np.r_[np.full(lag, np.nan), c[lag:] / c[:-lag] - 1] if len(c) > lag else np.full(len(c), np.nan)
    x["vg_top"] = np.maximum(x["ema144"], x["ema169"])
    x["vg_bot"] = np.minimum(x["ema144"], x["ema169"])
    # 背離：轉折點左右各 3 根，兩個轉折點相隔 5~60 根
    x["div_rsi"] = ci.divergence(h, l, x["rsi"])
    x["div_macd"] = ci.divergence(h, l, x["macd"])
    x["div_pv"] = ci.divergence(h, l, x["vol5"], osc_lo=-x["vol5"])
    x["div_obv"] = ci.divergence(h, l, x["obv"])
    return x


def S(name, zh, family, fn, natural="trend", base=None):
    """一個策略：fn(x) 回傳 (多單進場, 空單進場, 多單出場, 空單出場)；出場是收盤時的狀態，None 表示沒有公認出場。
    natural：trend（反向訊號出場，最多 240 根）、mr（回到均值出場，最多 30 根）、None（只跑 live / tp）。
    base：參數鄰近版本對應的預設版本名稱（None 表示自己就是預設）。"""
    return {"name": name, "zh": zh, "family": family, "fn": fn, "natural": natural, "base": base}


def _ma_cross(fast, slow):
    def fn(x):
        a, b = x[fast], x[slow]
        return xup(a, b), xdn(a, b), a < b, a > b
    return fn


def _price_cross(line):
    def fn(x):
        c, m = x["c"], x[line]
        return xup(c, m), xdn(c, m), c < m, c > m
    return fn


def _macd_cross(x):
    hst = x["hist"]
    up, dn = xup(hst, 0.0), xdn(hst, 0.0)
    return up, dn, hst < 0, hst > 0


def _macd_zero_side(x):
    hst, mc = x["hist"], x["macd"]
    up, dn = xup(hst, 0.0), xdn(hst, 0.0)
    return up & (mc > 0), dn & (mc < 0), hst < 0, hst > 0


def _rsi50(x):
    r = x["rsi"]
    return xup(r, 50.0), xdn(r, 50.0), r < 50, r > 50


def _bb_break(tag=""):
    def fn(x):
        c = x["c"]
        return xup(c, x[f"bb_up{tag}"]), xdn(c, x[f"bb_lo{tag}"]), c < x[f"bb_mid{tag}"], c > x[f"bb_mid{tag}"]
    return fn


def _bb_squeeze(x):
    c = x["c"]
    sq = _prev(x["bb_width"] <= x["bb_width_q20"])
    return sq & xup(c, x["bb_up"]), sq & xdn(c, x["bb_lo"]), c < x["bb_mid"], c > x["bb_mid"]


def _donchian(n, exit_n):
    def fn(x):
        c = x["c"]
        up = fresh(c > x[f"dc{n}_hi"])
        dn = fresh(c < x[f"dc{n}_lo"])
        return up, dn, c < x[f"dc{exit_n}_lo"], c > x[f"dc{exit_n}_hi"]
    return fn


def _supertrend(key):
    def fn(x):
        st = x[key]
        up = (st > 0) & (_prev(st) < 0)
        dn = (st < 0) & (_prev(st) > 0)
        return up, dn, st < 0, st > 0
    return fn


def _psar(x):
    bull = x["psar"] < x["c"]
    valid = np.isfinite(x["psar"])
    return fresh(bull & valid), fresh(~bull & valid), ~bull & valid, bull & valid


def _dmi(x):
    p, m, a = x["pdi"], x["mdi"], x["adx"]
    return xup(p, m) & (a > 20), xup(m, p) & (a > 20), m > p, p > m


def _keltner(x):
    c = x["c"]
    return xup(c, x["kc_up"]), xdn(c, x["kc_lo"]), c < x["kc_mid"], c > x["kc_mid"]


def _ichimoku(x):
    c = x["c"]
    return xup(c, x["cloud_top"]), xdn(c, x["cloud_bot"]), c < x["kijun"], c > x["kijun"]


def _hma(x):
    hm = x["hma"]
    rising = hm > _prev(hm)
    falling = hm < _prev(hm)
    return fresh(rising), fresh(falling), falling, rising


def _obv_trend(x):
    a, b = x["obv_e20"], x["obv_e50"]
    return xup(a, b), xdn(a, b), a < b, a > b


def _vol_breakout(x):
    c, v = x["c"], x["v"]
    loud = v > 1.5 * x["vol_ma20_prev"]
    up = fresh(c > x["dc20_hi"]) & loud
    dn = fresh(c < x["dc20_lo"]) & loud
    return up, dn, c < x["dc10_lo"], c > x["dc10_hi"]


def _tsmom(x):
    r = x["roc30d"]
    return xup(r, 0.0), xdn(r, 0.0), r < 0, r > 0


def _vegas_break(x):
    c = x["c"]
    return xup(c, x["vg_top"]), xdn(c, x["vg_bot"]), c < x["vg_bot"], c > x["vg_top"]


def _vegas_ema12(x):
    e = x["ema12"]
    return xup(e, x["vg_top"]), xdn(e, x["vg_bot"]), e < x["vg_bot"], e > x["vg_top"]


def _vegas_pullback(x):
    c, h, l, o, a = x["c"], x["h"], x["l"], x["o"], x["atr"]
    top, bot, e144, e169 = x["vg_top"], x["vg_bot"], x["ema144"], x["ema169"]
    rising = (e144 > e169) & (e144 > _prev(e144, 20))
    falling = (e144 < e169) & (e144 < _prev(e144, 20))
    up = fresh(rising & (c > top) & (l <= top + 0.25 * a) & (c > o))
    dn = fresh(falling & (c < bot) & (h >= bot - 0.25 * a) & (c < o))
    return up, dn, c < bot, c > top


def _rsi_band(lo, hi):
    def fn(x):
        r = x["rsi"]
        return xup(r, lo), xdn(r, hi), r >= 50, r <= 50
    return fn


def _rsi2(x):
    r, c = x["rsi2"], x["c"]
    return xdn(r, 10.0), xup(r, 90.0), c > x["sma5"], c < x["sma5"]


def _bb_revert(tag=""):
    def fn(x):
        c = x["c"]
        return xup(c, x[f"bb_lo{tag}"]), xdn(c, x[f"bb_up{tag}"]), c >= x[f"bb_mid{tag}"], c <= x[f"bb_mid{tag}"]
    return fn


def _kd(x):
    k, d = x["k"], x["d"]
    return xup(k, d) & (d < 20), xdn(k, d) & (d > 80), k < d, k > d


def _wr(x):
    w = x["wr"]
    return xup(w, -80.0), xdn(w, -20.0), w >= -50, w <= -50


def _cci(x):
    q = x["cci"]
    return xup(q, -100.0), xdn(q, 100.0), q >= 0, q <= 0


def _mfi(x):
    q = x["mfi"]
    return xup(q, 20.0), xdn(q, 80.0), q >= 50, q <= 50


def _smi(x):
    s_, g = x["smi"], x["smi_sig"]
    return xup(s_, g) & (s_ < -40), xdn(s_, g) & (s_ > 40), s_ < g, s_ > g


def _bias(x):
    c, e, a = x["c"], x["ema20"], x["atr"]
    dev = (c - e) / a
    return xup(dev, -2.5), xdn(dev, 2.5), c >= e, c <= e


def _div(key):
    def fn(x):
        bull, bear = x[key]
        return bull, bear, None, None
    return fn


def _trend_dirs(x):
    c, e = x["c"], x["ema200"]
    return (c > e) & (e > _prev(e, 10)), (c < e) & (e < _prev(e, 10))


def _trend_rsi(lo, hi):
    def fn(x):
        up, dn = _trend_dirs(x)
        r = x["rsi"]
        return up & xup(r, lo), dn & xdn(r, hi), None, None
    return fn


def _trend_rsi2(x):
    r, c = x["rsi2"], x["c"]
    return (c > x["sma200"]) & xdn(r, 10.0), (c < x["sma200"]) & xup(r, 90.0), c > x["sma5"], c < x["sma5"]


def _trend_kd(x):
    up, dn = _trend_dirs(x)
    k, d = x["k"], x["d"]
    return up & xup(k, d) & (d < 30), dn & xdn(k, d) & (d > 70), None, None


def _trend_bb(x):
    up, dn = _trend_dirs(x)
    c = x["c"]
    return up & xup(c, x["bb_lo"]), dn & xdn(c, x["bb_up"]), None, None


def _ema_stack(x):
    c, h, l = x["c"], x["h"], x["l"]
    e20, e50, e200 = x["ema20"], x["ema50"], x["ema200"]
    up = fresh((e20 > e50) & (e50 > e200) & (l <= e20) & (c > e20))
    dn = fresh((e20 < e50) & (e50 < e200) & (h >= e20) & (c < e20))
    return up, dn, None, None


def _macd_ema200(x):
    c, e, mc, hst = x["c"], x["ema200"], x["macd"], x["hist"]
    return (c > e) & xup(hst, 0.0) & (mc < 0), (c < e) & xdn(hst, 0.0) & (mc > 0), None, None


STRATEGIES = [
    # 順勢：均線、動能、突破
    S("sma_20_50", "SMA20/50 交叉", "trend", _ma_cross("sma20", "sma50")),
    S("sma_50_200", "SMA50/200 黃金/死亡交叉", "trend", _ma_cross("sma50", "sma200")),
    S("ema_20_50", "EMA20/50 交叉", "trend", _ma_cross("ema20", "ema50")),
    S("ema_12_26", "EMA12/26 交叉（MACD 零軸）", "trend", _ma_cross("ema12", "ema26")),
    S("price_sma200", "收盤穿越 SMA200", "trend", _price_cross("sma200")),
    S("price_ema50", "收盤穿越 EMA50", "trend", _price_cross("ema50")),
    S("macd_cross", "MACD 金叉/死叉", "trend", _macd_cross),
    S("macd_zero_side", "MACD 零軸上金叉/零軸下死叉", "trend", _macd_zero_side),
    S("rsi_50", "RSI 穿越 50", "trend", _rsi50),
    S("bb_break", "布林帶突破（收在帶外順勢）", "trend", _bb_break()),
    S("bb_squeeze", "布林帶收口後突破", "trend", _bb_squeeze),
    S("donchian_20", "唐奇安 20 根突破（海龜一）", "trend", _donchian(20, 10)),
    S("donchian_55", "唐奇安 55 根突破（海龜二）", "trend", _donchian(55, 20)),
    S("supertrend", "超級趨勢 10/3 翻轉", "trend", _supertrend("st_10_3")),
    S("psar", "拋物線 SAR 翻轉", "trend", _psar),
    S("dmi", "DMI 交叉（ADX>20）", "trend", _dmi),
    S("keltner_break", "肯特納通道突破", "trend", _keltner),
    S("ichimoku", "一目均衡表雲帶突破", "trend", _ichimoku),
    S("hma_turn", "HMA55 轉向", "trend", _hma),
    S("obv_trend", "OBV 均線交叉", "trend", _obv_trend),
    S("vol_breakout", "量價配合突破（放量破 20 根高低）", "trend", _vol_breakout),
    S("tsmom_30d", "30 天漲跌翻正/翻負（時間序列動能）", "trend", _tsmom),
    # 維加斯通道
    S("vegas_live", "實盤維加斯趨勢（對照）", "vegas", None, natural=None),
    S("vegas_break", "收盤突破維加斯通道", "vegas", _vegas_break),
    S("vegas_ema12", "EMA12 穿越維加斯通道", "vegas", _vegas_ema12),
    S("vegas_pullback", "維加斯通道回踩反彈", "vegas", _vegas_pullback),
    # 反轉（超買超賣）
    S("rsi_30_70", "RSI 30/70 超買超賣", "reversal", _rsi_band(30.0, 70.0), natural="mr"),
    S("rsi2", "RSI(2) 極端值（Connors，不看趨勢）", "reversal", _rsi2, natural="mr"),
    S("bb_revert", "布林帶回歸（跌破下軌後收回）", "reversal", _bb_revert(), natural="mr"),
    S("kd_cross", "KD 低檔金叉/高檔死叉", "reversal", _kd, natural="mr"),
    S("williams_r", "威廉指標 -80/-20", "reversal", _wr, natural="mr"),
    S("cci", "CCI ±100", "reversal", _cci, natural="mr"),
    S("mfi", "MFI 資金流量 20/80", "reversal", _mfi, natural="mr"),
    S("smi", "SMI 隨機動量 ±40 交叉", "reversal", _smi, natural="mr"),
    S("bias", "乖離率（離 EMA20 2.5 ATR 後收回）", "reversal", _bias, natural="mr"),
    # 背離
    S("rsi_div", "RSI 背離", "divergence", _div("div_rsi"), natural=None),
    S("macd_div", "MACD 背離", "divergence", _div("div_macd"), natural=None),
    S("pv_div", "量價背離（價創新高量縮→空、價創新低量縮→多）", "divergence", _div("div_pv"), natural=None),
    S("obv_div", "OBV 背離", "divergence", _div("div_obv"), natural=None),
    # 順勢回檔（趨勢過濾＋反轉觸發）
    S("trend_rsi_pullback", "EMA200 趨勢＋RSI 回檔 40/60", "pullback", _trend_rsi(40.0, 60.0), natural=None),
    S("trend_rsi2", "SMA200 趨勢＋RSI(2)（Connors 原版）", "pullback", _trend_rsi2, natural="mr"),
    S("trend_kd", "EMA200 趨勢＋KD 回檔金叉", "pullback", _trend_kd, natural=None),
    S("trend_bb", "EMA200 趨勢＋布林下軌收回", "pullback", _trend_bb, natural=None),
    S("ema_stack_pullback", "均線多頭排列回踩 EMA20", "pullback", _ema_stack, natural=None),
    S("macd_ema200", "EMA200＋MACD 零軸下金叉", "pullback", _macd_ema200, natural=None),
    # 參數鄰近版本（看結果是不是只在預設參數附近才好）
    S("sma_10_30", "SMA10/30 交叉", "trend", _ma_cross("sma10", "sma30"), base="sma_20_50"),
    S("sma_50_100", "SMA50/100 交叉", "trend", _ma_cross("sma50", "sma100"), base="sma_20_50"),
    S("ema_9_21", "EMA9/21 交叉", "trend", _ma_cross("ema9", "ema21"), base="ema_20_50"),
    S("ema_50_200", "EMA50/200 交叉", "trend", _ma_cross("ema50", "ema200"), base="ema_20_50"),
    S("supertrend_10_2", "超級趨勢 10/2 翻轉", "trend", _supertrend("st_10_2"), base="supertrend"),
    S("bb25_break", "布林帶 2.5 倍突破", "trend", _bb_break("25"), base="bb_break"),
    S("rsi_25_75", "RSI 25/75 超買超賣", "reversal", _rsi_band(25.0, 75.0), natural="mr", base="rsi_30_70"),
    S("rsi_20_80", "RSI 20/80 超買超賣", "reversal", _rsi_band(20.0, 80.0), natural="mr", base="rsi_30_70"),
    S("bb25_revert", "布林帶 2.5 倍回歸", "reversal", _bb_revert("25"), natural="mr", base="bb_revert"),
    S("trend_rsi_45_55", "EMA200 趨勢＋RSI 回檔 45/55", "pullback", _trend_rsi(45.0, 55.0), natural=None,
      base="trend_rsi_pullback"),
    S("trend_rsi_35_65", "EMA200 趨勢＋RSI 回檔 35/65", "pullback", _trend_rsi(35.0, 65.0), natural=None,
      base="trend_rsi_pullback"),
]
BY_NAME = {s["name"]: s for s in STRATEGIES}


def vegas_live_signals(df: pd.DataFrame, tf: str) -> tuple[np.ndarray, np.ndarray]:
    """實盤 vegas_trend 的訊號（和 VegasTrend.check / accept 逐根相同）。"""
    strat = VegasTrend()
    p = strat.prepare(df)
    n = len(p)
    long_, short_ = np.zeros(n, bool), np.zeros(n, bool)
    for i in np.flatnonzero(strat.candidates(p)):
        if i < strat.min_bars:
            continue
        hit = strat.check(p, int(i))
        if hit is None or not strat.accept(p, int(i), tf):
            continue
        (long_ if hit[0] == LONG else short_)[i] = True
    return long_, short_


# ---------------------------------------------------------------- 模擬

@njit(cache=True)
def simulate(op, hi, lo, cl, starts, ds, entries, risks, tps, portions, trails, fee, slip, max_bars, xl, xs, use_x):
    """每個進場各自跑一次。tps：(進場數, TP 數) 的價格；portions：各 TP 平倉比例；trails：TP 都碰到後
    追蹤止損的距離（價格，0 = 不追蹤）；xl / xs：多單 / 空單在第 j 根收盤時要出場（use_x=True 才看）。

    和 tracker / stop_research.sim_many 同一套規則：每根先看止損、再看 TP；TP1 後止損移到保本（含來回手續費），
    TP2 後移到 TP1；TP 用掛單價成交（只付手續費），止損和收盤出場另外付滑價。
    回傳 R、碰到幾個 TP、出場那根、出場種類（1 止損、2 保本/保護/追蹤止損、3 TP 全中、4 指標出場、5 超時、-1 資料結束）。
    """
    m = len(starts)
    out_r = np.zeros(m)
    out_hit = np.zeros(m, np.int64)
    out_j = np.zeros(m, np.int64)
    out_kind = np.zeros(m, np.int64)
    n = len(hi)
    ntp = tps.shape[1]
    for s in range(m):
        d = ds[s]
        entry = entries[s]
        risk = risks[s]
        stop = entry - d * risk
        trail = trails[s]
        be = entry * (1 + d * 2 * fee)
        ecost = (fee + slip) * entry
        remaining = 1.0
        r = 0.0
        hit = 0
        extreme = entry
        bars = 0
        kind = -1
        j = starts[s]
        while j < n:
            a = lo[j] if d > 0 else hi[j]
            if d * (a - stop) <= 0:
                r += remaining * (d * (stop - entry) - ecost - (fee + slip) * stop) / risk
                kind = 1 if hit == 0 else 2
                break
            f = hi[j] if d > 0 else lo[j]
            while hit < ntp and d * (f - tps[s, hit]) >= 0:
                p = portions[hit]
                r += p * (d * (tps[s, hit] - entry) - ecost - fee * tps[s, hit]) / risk
                remaining -= p
                ns = stop
                if hit == 0:
                    ns = be
                elif hit == 1:
                    ns = tps[s, 0]
                if d * (ns - stop) > 0:
                    stop = ns
                hit += 1
            if remaining <= 1e-9:
                kind = 3
                break
            if trail > 0 and hit == ntp:
                if d * (f - extreme) > 0:
                    extreme = f
                s2 = extreme - d * trail
                if d * (s2 - stop) > 0:
                    stop = s2
            bars += 1
            c = cl[j]
            if use_x and ((d > 0 and xl[j]) or (d < 0 and xs[j])):
                r += remaining * (d * (c - entry) - ecost - (fee + slip) * c) / risk
                kind = 4
                break
            if bars >= max_bars:
                r += remaining * (d * (c - entry) - ecost - (fee + slip) * c) / risk
                kind = 5
                break
            j += 1
        if kind == -1:
            j = n - 1
        out_r[s] = r
        out_hit[s] = hit
        out_j[s] = j
        out_kind[s] = kind
    return out_r, out_hit, out_j, out_kind


def run_exit(arr: dict, idx: np.ndarray, ds: np.ndarray, tf: str, exit_name: str, slip: float,
             xl=None, xs=None, natural: str | None = None) -> tuple:
    """idx / ds：進場那根與方向。回傳 simulate 的四個陣列。"""
    price = arr["c"][idx]
    atr = arr["atr"][idx]
    n = len(arr["c"])
    zeros_b = np.zeros(n, np.bool_)
    if exit_name == "live":
        tp_r, portions, trail_k = LIVE_PLANS[tf]
        k_stop, max_bars, trails = LIVE_ATR, MAX_BARS, trail_k * atr
    elif exit_name == "natural":
        tp_r, portions = (), ()
        k_stop, trails = NATURAL_ATR, np.zeros(len(idx))
        max_bars = MR_MAX_BARS if natural == "mr" else MAX_BARS
    else:
        tp_r, portions = (float(exit_name[2:]),), (1.0,)
        k_stop, max_bars, trails = LIVE_ATR, FIXED_MAX_BARS, np.zeros(len(idx))
    risk = np.maximum(k_stop * atr, MIN_STOP_PCT * price)
    if len(tp_r):
        tps = np.stack([price + ds * k * risk for k in tp_r], axis=1)
    else:
        tps = np.zeros((len(idx), 0))
    use_x = exit_name == "natural"
    return simulate(arr["o"], arr["h"], arr["l"], arr["c"], idx + 1, ds.astype(np.int64), price, risk, tps,
                    np.array(portions, float), trails, FEE, slip, max_bars,
                    xl if use_x else zeros_b, xs if use_x else zeros_b, use_x)


def _coin_work(args):
    coin, tf, slip, source = args
    df = load_frame(coin, tf, source)
    if len(df) < WARMUP + 100:
        return None
    x = features(df, tf)
    n = len(df)
    t = df["time"].to_numpy(dtype="datetime64[ns]")
    bar = np.timedelta64(BAR_HOURS[tf], "h")
    atr = x["atr"]
    valid = (np.arange(n) >= WARMUP) & (np.arange(n) + 1 < n) & np.isfinite(atr) & (atr > 0)
    arr = {k: x[k] for k in ("o", "h", "l", "c", "atr")}
    horizons = [hd * x["bpd"] for hd in HORIZON_DAYS]
    c = x["c"]

    def fwd(idx, ds):
        out = {}
        for hd, hb in zip(HORIZON_DAYS, horizons):
            j = idx + hb
            ok = j < n
            val = np.full(len(idx), np.nan)
            val[ok] = ds[ok] * (c[j[ok]] - c[idx[ok]]) / atr[idx[ok]]
            out[f"fwd{hd}d"] = val
        return out

    rows = []
    for st in STRATEGIES:
        if st["name"] == "vegas_live":
            lg, sh = vegas_live_signals(df, tf)
            xl = xs = None
        else:
            lg, sh, xl, xs = st["fn"](x)
        lg = np.asarray(lg, bool) & valid
        sh = np.asarray(sh, bool) & valid
        idx = np.r_[np.flatnonzero(lg), np.flatnonzero(sh)]
        if len(idx) == 0:
            continue
        ds = np.r_[np.ones(lg.sum(), np.int64), -np.ones(sh.sum(), np.int64)]
        order = np.argsort(idx, kind="stable")
        idx, ds = idx[order], ds[order]
        rec = {"strategy": st["name"], "coin": coin, "i": idx, "side": ds, "time": t[idx] + bar,
               "atr_pct": atr[idx] / c[idx]}
        for ex in EXITS:
            r, hit, j, kind = run_exit(arr, idx, ds, tf, ex, slip)
            rec[f"r_{ex}"], rec[f"j_{ex}"], rec[f"k_{ex}"] = r, j, kind
            if ex == "live":
                rec["hit_live"] = hit
        if st["natural"] and xl is not None:
            xl_ = np.asarray(xl, bool) & np.isfinite(c)
            xs_ = np.asarray(xs, bool) & np.isfinite(c)
            r, hit, j, kind = run_exit(arr, idx, ds, tf, "natural", slip, xl_, xs_, st["natural"])
            rec["r_natural"], rec["j_natural"], rec["k_natural"] = r, j, kind
        else:
            rec["r_natural"] = np.full(len(idx), np.nan)
            rec["j_natural"] = np.full(len(idx), -1, np.int64)
            rec["k_natural"] = np.full(len(idx), -9, np.int64)
        rec.update(fwd(idx, ds))
        rows.append(pd.DataFrame(rec))
    sig = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if len(sig):
        for ex in EXITS + ["natural"]:
            jj = sig[f"j_{ex}"].to_numpy()
            sig[f"exit_t_{ex}"] = np.where(jj >= 0, t[np.clip(jj, 0, n - 1)], np.datetime64("NaT", "ns"))

    # 隨機進場對照：每一根都做多、做空各一次
    bidx = np.flatnonzero(valid)
    base = []
    years = pd.DatetimeIndex(t[bidx] + bar).year.to_numpy()
    months = pd.DatetimeIndex(t[bidx] + bar).to_period("M").astype(str).to_numpy()
    for d in (1, -1):
        ds = np.full(len(bidx), d, np.int64)
        meas = {}
        for ex in EXITS:
            r, _, _, kind = run_exit(arr, bidx, ds, tf, ex, slip)
            meas[f"r_{ex}"] = np.where(kind == -1, np.nan, r)
        meas.update(fwd(bidx, ds))
        vals = pd.DataFrame(meas)
        pos = (vals > 0).astype(float).where(vals.notna())
        for key, lab in (("year", years), ("month", months)):
            g = vals.groupby(lab)
            mean, cnt = g.mean(), g.count()
            win = pos.groupby(lab).mean()
            for col in meas:
                base.append(pd.DataFrame({"coin": coin, "level": key, "key": mean.index.astype(str), "side": d,
                                          "measure": col, "mean": mean[col].to_numpy(), "win": win[col].to_numpy(),
                                          "count": cnt[col].to_numpy()}))
    base = pd.concat(base, ignore_index=True)
    span = (str(pd.Timestamp(t[0]).date()), str(pd.Timestamp(t[-1]).date()), int(n))
    return coin, sig, base, span


def run_tag(tf: str, slip: float = 0.0, source: str = "binance") -> str:
    return tf + (f"_slip{slip:g}" if slip else "") + ("_pionex" if source == "pionex" else "")


def run(tf: str, slip: float, workers: int, out: Path, source: str = "binance") -> None:
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(c, tf, slip, source) for c in TOP100]
    sigs, bases, spans = [], [], {}
    with ProcessPoolExecutor(workers) as ex:
        for res in ex.map(_coin_work, jobs):
            if res is None:
                continue
            coin, sig, base, span = res
            sigs.append(sig)
            bases.append(base)
            spans[coin] = span
            print(coin, len(sig), span, flush=True)
    tag = run_tag(tf, slip, source)
    sig = pd.concat(sigs, ignore_index=True)
    base = pd.concat(bases, ignore_index=True)
    sig.to_pickle(out / f"signals_{tag}.pkl")
    base.to_pickle(out / f"baseline_{tag}.pkl")
    (out / f"spans_{tag}.json").write_text(json.dumps(spans, ensure_ascii=False, indent=1))
    print("saved", len(sig), "signals", flush=True)


# ---------------------------------------------------------------- 統計

@njit(cache=True)
def busy_filter(coin_id, i, j, kind):
    """同一個幣同時只持有一張：前一張出場那根之前的新訊號跳過（和 stop_research.busy_filter 相同）。
    訊號要先依（幣, i）排序。"""
    take = np.zeros(len(i), np.bool_)
    busy = -1
    prev = -1
    for s in range(len(i)):
        if coin_id[s] != prev:
            busy = -1
            prev = coin_id[s]
        if i[s] + 1 < busy or kind[s] == -1:
            continue
        take[s] = True
        busy = j[s]
    return take


def cluster_stats(x: np.ndarray, key: np.ndarray) -> tuple[float, float, float, float, int]:
    """平均、群集穩健標準誤（同一天同方向算一群）、群集等權平均和它的標準誤、群數。"""
    n = len(x)
    if n == 0:
        return np.nan, np.nan, np.nan, np.nan, 0
    s = pd.Series(x).groupby(key)
    sums, cnt = s.sum(), s.count()
    mean = x.mean()
    se = np.sqrt(((sums - cnt * mean) ** 2).sum()) / n
    cm = sums / cnt
    return mean, se, cm.mean(), (cm.std() / np.sqrt(len(cm)) if len(cm) > 1 else np.nan), len(cm)


def daily_series(r: np.ndarray, t_out: np.ndarray, start, end, risk: float = 0.0025) -> pd.Series:
    day = pd.Series(risk * r, index=pd.DatetimeIndex(t_out).floor("D")).groupby(level=0).sum()
    rng = pd.date_range(pd.Timestamp(start).floor("D"), pd.Timestamp(end).floor("D"), freq="D")
    return day.reindex(rng, fill_value=0.0)


def account(r, t_in, t_out, start, end, risk: float = 0.0025) -> dict:
    """每筆風險 risk 時的 Sharpe（每日損益，依出場日入帳）、年化報酬、最大回撤。"""
    if len(r) == 0:
        return {"sharpe": np.nan, "ann_pct": np.nan, "max_dd_pct": np.nan}
    order = np.argsort(t_out, kind="stable")
    eq = np.cumprod(1 + risk * r[order])
    dd = (eq / np.maximum.accumulate(eq) - 1).min()
    years = max((pd.Timestamp(end) - pd.Timestamp(start)).days / 365.25, 0.1)
    day = daily_series(r, t_out, start, end, risk)
    sd = day.std()
    return {"sharpe": day.mean() / sd * np.sqrt(365) if sd > 0 else np.nan,
            "ann_pct": (eq[-1] ** (1 / years) - 1) * 100 if eq[-1] > 0 else -100.0, "max_dd_pct": dd * 100}


def base_lookup(sig: pd.DataFrame, base: pd.DataFrame, measure: str, level: str) -> tuple[np.ndarray, np.ndarray]:
    """每筆訊號對應的隨機進場平均（同幣、同方向、同一年 / 同一月）和隨機進場勝率。"""
    b = base[(base["measure"] == measure) & (base["level"] == level)]
    key = sig["time"].dt.year.astype(str) if level == "year" else sig["time"].dt.to_period("M").astype(str)
    m = pd.DataFrame({"coin": sig["coin"].to_numpy(), "key": key.to_numpy(), "side": sig["side"].to_numpy()})
    m = m.merge(b[["coin", "key", "side", "mean", "win"]], on=["coin", "key", "side"], how="left")
    return m["mean"].to_numpy(), m["win"].to_numpy()


PERIODS = {"全期": (None, None), "2020-23": (None, SPLIT), "2024+": (SPLIT, None)}


def period_mask(t: pd.Series, period: str) -> np.ndarray:
    lo, hi = PERIODS[period] if period in PERIODS else (pd.Timestamp(f"{period}-01-01"),
                                                        pd.Timestamp(f"{int(period) + 1}-01-01"))
    m = np.ones(len(t), bool)
    if lo is not None:
        m &= (t >= lo).to_numpy()
    if hi is not None:
        m &= (t < hi).to_numpy()
    return m


def stat_block(g: pd.DataFrame, ex: str, start, end, take: np.ndarray, base_y, base_m, base_wy) -> dict:
    """一組交易（已經套用同幣一張、期間）的統計。"""
    r = g[f"r_{ex}"].to_numpy()
    ok = take & np.isfinite(r)
    r = r[ok]
    if len(r) == 0:
        return {"trades": 0}
    t_in = g["time"].to_numpy()[ok]
    t_out = g[f"exit_t_{ex}"].to_numpy()[ok]
    side = g["side"].to_numpy()[ok]
    key = (pd.DatetimeIndex(t_in).floor("D").astype(str) + side.astype(str)).to_numpy()
    mean, se_cl, cmean, cse, ncl = cluster_stats(r, key)
    by = base_y[ok]
    bm = base_m[ok]
    edge_y = r - by
    edge_m = r - bm
    fy = np.isfinite(edge_y)
    fm = np.isfinite(edge_m)
    ey, ey_se, ey_c, ey_cse, _ = cluster_stats(edge_y[fy], key[fy])
    em, em_se, em_c, em_cse, _ = cluster_stats(edge_m[fm], key[fm])
    wins, losses = r[r > 0].sum(), -r[r <= 0].sum()
    days = max((pd.Timestamp(end) - pd.Timestamp(start)).days, 1)
    acc = account(r, t_in, t_out, start, end)
    kinds = g[f"k_{ex}"].to_numpy()[ok]
    bars = (g[f"j_{ex}"].to_numpy()[ok] - g["i"].to_numpy()[ok])
    return {
        "trades": len(r), "per_day": len(r) / days, "win_rate": (r > 0).mean(), "exp_r": mean,
        "se_r": r.std() / np.sqrt(len(r)), "se_r_cluster": se_cl, "cluster_r": cmean, "cluster_se": cse,
        "clusters": ncl, "pf": wins / losses if losses > 0 else np.inf,
        "base_r_year": np.nanmean(by), "base_win_year": np.nanmean(base_wy[ok]),
        "edge_year": ey, "edge_year_se": ey_se, "edge_year_cluster": ey_c, "edge_year_cluster_se": ey_cse,
        "edge_month": em, "edge_month_se": em_se, "edge_month_cluster": em_c, "edge_month_cluster_se": em_cse,
        "long_trades": int((side > 0).sum()), "short_trades": int((side < 0).sum()),
        "long_r": r[side > 0].mean() if (side > 0).any() else np.nan,
        "short_r": r[side < 0].mean() if (side < 0).any() else np.nan,
        "stop_rate": (kinds == 1).mean(), "avg_bars": bars.mean(), **acc,
    }


def build_tables(tf: str, out: Path, tag_extra: str = "") -> pd.DataFrame:
    tag = tf + tag_extra
    sig = pd.read_pickle(out / f"signals_{tag}.pkl")
    base = pd.read_pickle(out / f"baseline_{tag}.pkl")
    sig["time"] = pd.to_datetime(sig["time"])
    sig = sig.sort_values(["strategy", "coin", "i"], kind="stable").reset_index(drop=True)
    t_all_start = sig["time"].min()
    t_all_end = max(pd.to_datetime(sig[f"exit_t_{ex}"]).max() for ex in EXITS)
    bounds = {"全期": (t_all_start, t_all_end), "2020-23": (t_all_start, SPLIT), "2024+": (SPLIT, t_all_end)}
    for y in range(2020, 2027):
        bounds[str(y)] = (max(pd.Timestamp(f"{y}-01-01"), t_all_start),
                          min(pd.Timestamp(f"{y + 1}-01-01"), t_all_end))
    rows = []
    for ex in EXITS + ["natural"]:
        by, bwy = base_lookup(sig, base, f"r_{ex}", "year") if ex != "natural" else (None, None)
        bm, _ = base_lookup(sig, base, f"r_{ex}", "month") if ex != "natural" else (None, None)
        if ex == "natural":
            by = bm = bwy = np.full(len(sig), np.nan)
        for name, g in sig.groupby("strategy", sort=False):
            gi = g.index.to_numpy()
            r = g[f"r_{ex}"].to_numpy()
            if not np.isfinite(r).any():
                continue
            coin_id = pd.factorize(g["coin"])[0]
            take = busy_filter(coin_id, g["i"].to_numpy(np.int64), g[f"j_{ex}"].to_numpy(np.int64),
                               g[f"k_{ex}"].to_numpy(np.int64))
            for per, (start, end) in bounds.items():
                pm = period_mask(g["time"], per)
                s = stat_block(g, ex, start, end, take & pm, by[gi], bm[gi], bwy[gi])
                rows.append({"tf": tf, "strategy": name, "exit": ex, "period": per, **s})
    tab = pd.DataFrame(rows)
    tab.to_csv(out / f"table_{tag}.csv", index=False)
    return tab


def signal_info(tf: str, out: Path) -> pd.DataFrame:
    """訊號本身的方向資訊：固定天數報酬（ATR 計，不扣成本）和同幣同方向同年/同月隨機的差，全部訊號都算（不套同幣一張）。"""
    sig = pd.read_pickle(out / f"signals_{tf}.pkl")
    base = pd.read_pickle(out / f"baseline_{tf}.pkl")
    sig["time"] = pd.to_datetime(sig["time"])
    key = (sig["time"].dt.floor("D").astype(str) + sig["side"].astype(str)).to_numpy()
    rows = []
    for hd in HORIZON_DAYS:
        col = f"fwd{hd}d"
        by, bwy = base_lookup(sig, base, col, "year")
        bm, bwm = base_lookup(sig, base, col, "month")
        v = sig[col].to_numpy()
        for name, g in sig.groupby("strategy", sort=False):
            gi = g.index.to_numpy()
            for per in PERIODS:
                pm = period_mask(g["time"], per) & np.isfinite(v[gi])
                ii = gi[pm]
                if len(ii) < 30:
                    continue
                ey = v[ii] - by[ii]
                em = v[ii] - bm[ii]
                f1, f2 = np.isfinite(ey), np.isfinite(em)
                _, _, cy, cyse, ncl = cluster_stats(ey[f1], key[ii][f1])
                _, _, cmn, cmse, _ = cluster_stats(em[f2], key[ii][f2])
                rows.append({"tf": tf, "strategy": name, "horizon_d": hd, "period": per, "signals": len(ii),
                             "clusters": ncl, "fwd_atr": v[ii].mean(), "hit": (v[ii] > 0).mean(),
                             "base_hit_year": np.nanmean(bwy[ii]), "base_hit_month": np.nanmean(bwm[ii]),
                             "edge_year_atr": np.nanmean(ey), "edge_year_cluster": cy, "edge_year_cluster_se": cyse,
                             "edge_month_atr": np.nanmean(em), "edge_month_cluster": cmn,
                             "edge_month_cluster_se": cmse})
    tab = pd.DataFrame(rows)
    tab.to_csv(out / f"info_{tf}.csv", index=False)
    return tab


def monthly_matrix(tf: str, out: Path, exits=("live", "natural", "tp1")) -> pd.DataFrame:
    """每個（策略, 出場）每月損益（每筆風險 0.25%，同幣一張，依出場月入帳）。給 PBO 和相關性用。"""
    sig = pd.read_pickle(out / f"signals_{tf}.pkl")
    sig["time"] = pd.to_datetime(sig["time"])
    sig = sig.sort_values(["strategy", "coin", "i"], kind="stable").reset_index(drop=True)
    cols = {}
    daily = {}
    for ex in exits:
        for name, g in sig.groupby("strategy", sort=False):
            r = g[f"r_{ex}"].to_numpy()
            if not np.isfinite(r).any():
                continue
            coin_id = pd.factorize(g["coin"])[0]
            take = busy_filter(coin_id, g["i"].to_numpy(np.int64), g[f"j_{ex}"].to_numpy(np.int64),
                               g[f"k_{ex}"].to_numpy(np.int64)) & np.isfinite(r)
            t_out = pd.to_datetime(g[f"exit_t_{ex}"].to_numpy()[take])
            s = pd.Series(0.0025 * r[take], index=t_out)
            cols[(tf, name, ex)] = s.groupby(s.index.to_period("M")).sum()
            daily[(tf, name, ex)] = s.groupby(s.index.floor("D")).sum()
    mm = pd.DataFrame(cols).fillna(0.0).sort_index()
    dd = pd.DataFrame(daily).fillna(0.0).sort_index()
    return mm, dd


# ---------------------------------------------------------------- 過度擬合

def pbo_cscv(mm: pd.DataFrame, blocks: int = 16) -> dict:
    """Bailey 等人的 CSCV：月報酬切成 blocks 段，任取一半當樣本內挑 Sharpe 最高的設定，看它在另一半的排名。
    PBO = 樣本內最好的設定在樣本外低於中位數的比例。"""
    x = mm.to_numpy()
    T, N = x.shape
    edges = np.linspace(0, T, blocks + 1).astype(int)
    parts = [x[edges[b]:edges[b + 1]] for b in range(blocks)]
    sums = np.stack([p.sum(0) for p in parts])          # (blocks, N)
    sq = np.stack([(p ** 2).sum(0) for p in parts])
    cnt = np.array([len(p) for p in parts], float)
    logits, is_best, oos_best = [], [], []
    for comb in combinations(range(blocks), blocks // 2):
        a = np.zeros(blocks, bool)
        a[list(comb)] = True
        out = []
        for m in (a, ~a):
            n_ = cnt[m].sum()
            mu = sums[m].sum(0) / n_
            var = sq[m].sum(0) / n_ - mu ** 2
            out.append(mu / np.sqrt(np.maximum(var, 1e-18)))
        sr_is, sr_oos = out
        best = int(np.argmax(sr_is))
        rank = (sr_oos < sr_oos[best]).sum() + 0.5 * ((sr_oos == sr_oos[best]).sum() - 1)
        w = (rank + 0.5) / N
        logits.append(math.log(w / (1 - w)))
        is_best.append(sr_is[best])
        oos_best.append(sr_oos[best])
    logits = np.array(logits)
    return {"configs": N, "months": T, "splits": len(logits), "pbo": float((logits < 0).mean()),
            "median_logit": float(np.median(logits)), "oos_sr_of_is_best_median": float(np.median(oos_best)) * math.sqrt(12),
            "is_sr_best_median": float(np.median(is_best)) * math.sqrt(12),
            "oos_sr_of_is_best_p_pos": float((np.array(oos_best) > 0).mean())}


def _norm_cdf(z):
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def _norm_ppf(p):
    # Acklam 近似（不想為了這個裝 scipy）
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02, 1.383577518672690e+02,
         -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01,
         -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00]
    lo, hi = 0.02425, 1 - 0.02425
    if p < lo:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > hi:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
        (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


def deflated_sharpe(daily: pd.Series, n_trials: int, sr_var: float) -> dict:
    """Bailey & López de Prado 的 Deflated Sharpe：考慮試了 n_trials 個設定後，最好那個的 Sharpe 是真的 > 0 的機率。
    daily 是每日報酬；sr_var 是所有設定（每日）Sharpe 的變異數。"""
    x = daily.to_numpy()
    T = len(x)
    sr = x.mean() / x.std()
    g3 = ((x - x.mean()) ** 3).mean() / x.std() ** 3
    g4 = ((x - x.mean()) ** 4).mean() / x.std() ** 4
    emc = 0.5772156649
    sr0 = math.sqrt(sr_var) * ((1 - emc) * _norm_ppf(1 - 1 / n_trials) + emc * _norm_ppf(1 - 1 / (n_trials * math.e)))
    z = (sr - sr0) * math.sqrt(T - 1) / math.sqrt(max(1 - g3 * sr + (g4 - 1) / 4 * sr ** 2, 1e-12))
    return {"sr_daily": sr, "sr_ann": sr * math.sqrt(365), "sr0_ann": sr0 * math.sqrt(365), "dsr": _norm_cdf(z),
            "psr0": _norm_cdf(sr * math.sqrt(T - 1) / math.sqrt(max(1 - g3 * sr + (g4 - 1) / 4 * sr ** 2, 1e-12))),
            "skew": g3, "kurt": g4, "days": T}


# ---------------------------------------------------------------- 市場本身

def variance_ratio(r: np.ndarray, q: int, robust: bool = True) -> tuple[float, float]:
    """Lo-MacKinlay 變異數比 VR(q) 和異質變異穩健 z 值。VR > 1 代表有趨勢（動能），< 1 代表會反轉。"""
    r = r[np.isfinite(r)]
    T = len(r)
    mu = r.mean()
    var1 = ((r - mu) ** 2).sum() / (T - 1)
    rq = np.convolve(r, np.ones(q), "valid")
    m = q * (T - q + 1) * (1 - q / T)
    varq = ((rq - q * mu) ** 2).sum() / m
    vr = varq / var1
    if not robust:
        return vr, np.nan
    dev2 = (r - mu) ** 2
    theta = 0.0
    for j in range(1, q):
        delta = (dev2[j:] * dev2[:-j]).sum() / dev2.sum() ** 2 * T
        theta += (2 * (q - j) / q) ** 2 * delta
    return vr, (vr - 1) / math.sqrt(theta / T) if theta > 0 else np.nan


def market(out: Path) -> None:
    rows, tsm = [], []
    for tf in ("4H", "2H"):
        bpd = 24 // BAR_HOURS[tf]
        rets = {}
        for coin in TOP100:
            df = load_frame(coin, tf)
            if len(df) < 500:
                continue
            s = np.log(df.set_index("time")["close"]).diff()
            s.index = s.index.tz_convert(None)
            rets[coin] = s
        R = pd.DataFrame(rets).sort_index()
        R = R[R.index < pd.Timestamp("2026-10-01")]
        ew = R.mean(axis=1, skipna=True)
        for per, (lo, hi) in PERIODS.items():
            m = np.ones(len(R), bool)
            if lo is not None:
                m &= R.index >= lo
            if hi is not None:
                m &= R.index < hi
            for days in (0.5, 1, 3, 7, 14, 30):
                q = max(int(days * bpd), 2) if days >= 1 else max(int(days * bpd), 2)
                vr_ew, z_ew = variance_ratio(ew[m].to_numpy(), q)
                vrs = [variance_ratio(R.loc[m, c].dropna().to_numpy(), q, False)[0] for c in R.columns
                       if R.loc[m, c].notna().sum() > 10 * q]
                # 個別幣扣掉大盤（等權指數）之後剩下的漲跌：是延續還是反轉
                vrr = [variance_ratio((R.loc[m, c] - ew[m]).dropna().to_numpy(), q, False)[0] for c in R.columns
                       if R.loc[m, c].notna().sum() > 10 * q]
                vr_btc = variance_ratio(R.loc[m, "BTC"].dropna().to_numpy(), q)
                rows.append({"tf": tf, "period": per, "q_bars": q, "q_days": q / bpd, "vr_index": vr_ew, "z_index": z_ew,
                             "vr_btc": vr_btc[0], "z_btc": vr_btc[1],
                             "vr_coin_median": float(np.median(vrs)), "vr_coin_share_above1": float(np.mean(np.array(vrs) > 1)),
                             "vr_resid_median": float(np.median(vrr)),
                             "vr_resid_share_below1": float(np.mean(np.array(vrr) < 1)), "coins": len(vrs)})
        # 時間序列動能：過去 L 天漲就做多、跌就做空，接下來 H 天的平均報酬（以 ATR 計，每天每方向算一注）
        if tf == "4H":
            for coin in TOP100:
                df = load_frame(coin, tf)
                if len(df) < 500:
                    continue
                c = df["close"].to_numpy(float)
                atr = ind.atr(df).to_numpy(float)
                t = df["time"].dt.tz_convert(None)
                for L in (1, 3, 7, 14, 30):
                    for H in (1, 3, 7):
                        lb, hb = L * bpd, H * bpd
                        i = np.arange(max(lb, WARMUP), len(c) - hb, bpd)  # 每天取一根，避免同一段重疊太多
                        sgn = np.sign(c[i] - c[i - lb])
                        fw = sgn * (c[i + hb] - c[i]) / atr[i]
                        tsm.append(pd.DataFrame({"coin": coin, "L": L, "H": H, "time": t.iloc[i].to_numpy(),
                                                 "side": sgn, "fwd": fw}))
    vr = pd.DataFrame(rows)
    vr.to_csv(out / "market_vr.csv", index=False)
    tsm = pd.concat(tsm, ignore_index=True)
    tsm["time"] = pd.to_datetime(tsm["time"])
    trows = []
    for (L, H), g in tsm.groupby(["L", "H"]):
        for per in PERIODS:
            pm = period_mask(g["time"], per) & np.isfinite(g["fwd"].to_numpy()) & (g["side"].to_numpy() != 0)
            gg = g[pm]
            key = (gg["time"].dt.floor("D").astype(str) + gg["side"].astype(str)).to_numpy()
            mean, se, cm, cse, ncl = cluster_stats(gg["fwd"].to_numpy(), key)
            trows.append({"L_days": L, "H_days": H, "period": per, "obs": len(gg), "fwd_atr": mean, "se_cluster": se,
                          "cluster_mean": cm, "cluster_se": cse, "hit": (gg["fwd"] > 0).mean()})
    pd.DataFrame(trows).to_csv(out / "market_tsmom.csv", index=False)
    print(vr.to_string())
    print(pd.DataFrame(trows).to_string())


# ---------------------------------------------------------------- 第二步：彙整

DEFAULTS = [s_["name"] for s_ in STRATEGIES if s_["base"] is None]
FAMILY_ZH = {"trend": "順勢", "vegas": "維加斯", "reversal": "反轉", "divergence": "背離", "pullback": "順勢回檔"}


def _t(a, b):
    return a / b if (b is not None and np.isfinite(b) and b > 0) else np.nan


def selection_table(tab: pd.DataFrame) -> pd.DataFrame:
    """事先定好的篩選規則（看結果前就寫好）：
    - 挑選段：至少 100 筆；群集層級（同一天同方向算一注）每注平均 > 0 且 t ≥ 2；對隨機進場（同幣同方向同年）
      的群集差 > 0 且 t ≥ 2（natural 出場沒有隨機對照，不看這條）；Sharpe ≥ 0.5。
    - 驗證段：至少 50 筆；群集平均 > 0、對隨機的群集差 > 0、Sharpe > 0。
    兩個方向都做：2020-23 挑 → 2024+ 驗證，2024+ 挑 → 2020-23 驗證。"""
    rows = []
    for (tf, name, ex), g in tab.groupby(["tf", "strategy", "exit"], sort=False):
        p = g.set_index("period")
        if not {"2020-23", "2024+", "全期"} <= set(p.index):
            continue

        def disc(x):
            ok = x.trades >= 100 and x.cluster_r > 0 and _t(x.cluster_r, x.cluster_se) >= 2 and x.sharpe >= 0.5
            if ex != "natural":
                ok = ok and x.edge_year_cluster > 0 and _t(x.edge_year_cluster, x.edge_year_cluster_se) >= 2
            return bool(ok)

        def val(x):
            ok = x.trades >= 50 and x.cluster_r > 0 and x.sharpe > 0
            if ex != "natural":
                ok = ok and x.edge_year_cluster > 0
            return bool(ok)

        a, b, w = p.loc["2020-23"], p.loc["2024+"], p.loc["全期"]
        yrs = [str(y) for y in range(2020, 2027) if str(y) in p.index and p.loc[str(y)].trades >= 20]
        st = BY_NAME[name]
        rows.append({
            "tf": tf, "strategy": name, "zh": st["zh"], "family": st["family"], "default": st["base"] is None,
            "exit": ex, "trades": w.trades, "per_day": w.per_day, "win_rate": w.win_rate, "exp_r": w.exp_r,
            "pf": w.pf, "cluster_r": w.cluster_r, "cluster_t": _t(w.cluster_r, w.cluster_se),
            "edge_year": w.edge_year, "edge_year_cluster": w.edge_year_cluster,
            "edge_year_cluster_t": _t(w.edge_year_cluster, w.edge_year_cluster_se),
            "edge_month_cluster": w.edge_month_cluster, "edge_month_cluster_t": _t(w.edge_month_cluster, w.edge_month_cluster_se),
            "base_win_year": w.base_win_year, "sharpe": w.sharpe, "ann_pct": w.ann_pct, "max_dd_pct": w.max_dd_pct,
            "long_r": w.long_r, "short_r": w.short_r, "avg_bars": w.avg_bars,
            "a_trades": a.trades, "a_r": a.exp_r, "a_cluster_r": a.cluster_r, "a_cluster_t": _t(a.cluster_r, a.cluster_se),
            "a_edge_cluster": a.edge_year_cluster, "a_sharpe": a.sharpe, "a_win": a.win_rate,
            "b_trades": b.trades, "b_r": b.exp_r, "b_cluster_r": b.cluster_r, "b_cluster_t": _t(b.cluster_r, b.cluster_se),
            "b_edge_cluster": b.edge_year_cluster, "b_sharpe": b.sharpe, "b_win": b.win_rate,
            "years_pos": int(sum(p.loc[y].exp_r > 0 for y in yrs)), "years": len(yrs),
            "pick_a_val_b": disc(a) and val(b), "pick_b_val_a": disc(b) and val(a),
            "disc_a": disc(a), "disc_b": disc(b),
        })
    out = pd.DataFrame(rows)
    out["robust"] = out["pick_a_val_b"] & out["pick_b_val_a"]
    return out


def corr_clusters(C: pd.DataFrame, threshold: float = 0.5) -> list[list[str]]:
    """平均連結的階層分群（距離 = 1 - 相關係數），群和群的平均相關低於 threshold 就不再合併。"""
    names = list(C.columns)
    D = 1 - C.to_numpy()
    groups = [[i] for i in range(len(names))]
    while len(groups) > 1:
        best = None
        for a in range(len(groups)):
            for b in range(a + 1, len(groups)):
                d = D[np.ix_(groups[a], groups[b])].mean()
                if best is None or d < best[0]:
                    best = (d, a, b)
        if best[0] > 1 - threshold:
            break
        _, a, b = best
        groups[a] = groups[a] + groups[b]
        del groups[b]
    return [[names[i] for i in g] for g in sorted(groups, key=len, reverse=True)]


def funding_r(sig: pd.DataFrame, ex: str) -> np.ndarray:
    """每筆交易的資金費率成本（R，負的是付出去）：進場到出場之間每次結算 × 方向，倉位當成一直全倉持有到最後出場
    （TP1 先平掉的部分也算，所以多單在正費率時會略為高估成本）。"""
    k_stop = NATURAL_ATR if ex == "natural" else LIVE_ATR
    out = np.full(len(sig), np.nan)
    for coin, g in sig.groupby("coin"):
        fp = FUNDING_DIR / f"{coin}.csv.gz"
        if not fp.exists():
            continue
        f = pd.read_csv(fp)
        ft = pd.to_datetime(f["calc_time"], unit="ms").to_numpy()
        cum = np.r_[0.0, np.cumsum(f["last_funding_rate"].to_numpy(float))]
        tin = pd.to_datetime(g["time"]).to_numpy()
        tout = pd.to_datetime(g[f"exit_t_{ex}"]).to_numpy()
        a = np.searchsorted(ft, tin, side="right")
        b = np.searchsorted(ft, tout, side="right")
        risk_pct = np.maximum(k_stop * g["atr_pct"].to_numpy(), MIN_STOP_PCT)
        out[g.index.to_numpy()] = -g["side"].to_numpy() * (cum[b] - cum[a]) / risk_pct
    return out


def analyze_more(out: Path) -> None:
    """選出穩健的策略、過度擬合檢查、指標之間的相關、勝率與期望值、資金費率、幣種子集。"""
    tabs = [pd.read_csv(out / f"table_{tf}.csv") for tf in ("4H", "2H") if (out / f"table_{tf}.csv").exists()]
    tab = pd.concat(tabs, ignore_index=True)
    sel = selection_table(tab)
    sel.to_csv(out / "selection.csv", index=False)
    summary = {"robust": sel[sel.robust][["tf", "strategy", "exit"]].values.tolist()}

    # 過度擬合：所有設定（週期 × 策略 × 出場 live/natural/tp1）的每月損益
    mms, dds = [], []
    for tf in ("4H", "2H"):
        if (out / f"signals_{tf}.pkl").exists():
            mm, dd = monthly_matrix(tf, out)
            mms.append(mm)
            dds.append(dd)
    mm = pd.concat(mms, axis=1).fillna(0.0)
    dd = pd.concat(dds, axis=1).fillna(0.0)
    mm = mm[mm.index < pd.Period("2026-10", "M")]
    mm.to_pickle(out / "monthly_matrix.pkl")
    dd.to_pickle(out / "daily_matrix.pkl")
    res = {"all": pbo_cscv(mm)}
    for fam in FAMILY_ZH:
        cols = [c for c in mm.columns if BY_NAME[c[1]]["family"] == fam]
        if len(cols) >= 4:
            res[fam] = pbo_cscv(mm[cols])
    live_cols = [c for c in mm.columns if c[2] == "live"]
    res["live_exit_only"] = pbo_cscv(mm[live_cols])
    summary["pbo"] = res
    # Deflated Sharpe：全期每日 Sharpe 最高的設定
    full = dd.reindex(pd.date_range(dd.index.min(), dd.index.max(), freq="D"), fill_value=0.0)
    sr = full.mean() / full.std()
    best = sr.idxmax()
    clusters_all = corr_clusters(full.corr().fillna(0.0), 0.5)
    summary["dsr"] = {"best": list(best), "n_trials": len(sr), "n_eff": len(clusters_all),
                      "all": deflated_sharpe(full[best], len(sr), float(sr.var())),
                      "eff": deflated_sharpe(full[best], len(clusters_all), float(sr.var()))}
    # 2024 年後：用 2020-23 每日 Sharpe 最高的設定
    pre = full[full.index < SPLIT]
    post = full[full.index >= SPLIT]
    sr_pre = pre.mean() / pre.std()
    top = sr_pre.sort_values(ascending=False).index[:10]
    summary["is_top10_2020_23"] = [{"cfg": list(c), "sr_2020_23": float(sr_pre[c] * math.sqrt(365)),
                                    "sr_2024": float(post[c].mean() / post[c].std() * math.sqrt(365))} for c in top]
    summary["rank_corr_sr"] = float(sr_pre.rank().corr((post.mean() / post.std()).rank()))

    # 指標之間的相關（4H、live 出場、預設參數）
    for tf in ("4H", "2H"):
        cols = [c for c in full.columns if c[0] == tf and c[2] == "live" and c[1] in DEFAULTS]
        if not cols:
            continue
        C = full[cols].corr().fillna(0.0)
        C.index = C.columns = [c[1] for c in cols]
        C.to_csv(out / f"corr_{tf}.csv")
        groups = corr_clusters(C, 0.5)
        summary[f"clusters_{tf}"] = [{"members": g, "avg_corr": float(C.loc[g, g].to_numpy()[np.triu_indices(len(g), 1)].mean())
                                      if len(g) > 1 else 1.0} for g in groups]
        summary[f"corr_with_vegas_{tf}"] = C["vegas_live"].drop("vegas_live").sort_values().to_dict()

    # 資金費率：live 和 natural 出場
    frows = []
    for tf in ("4H", "2H"):
        if not (out / f"signals_{tf}.pkl").exists():
            continue
        sig = pd.read_pickle(out / f"signals_{tf}.pkl")
        sig = sig.sort_values(["strategy", "coin", "i"], kind="stable").reset_index(drop=True)
        sig["time"] = pd.to_datetime(sig["time"])
        for ex in ("live", "natural"):
            fr = funding_r(sig, ex)
            for name, g in sig.groupby("strategy", sort=False):
                r = g[f"r_{ex}"].to_numpy()
                if not np.isfinite(r).any():
                    continue
                take = busy_filter(pd.factorize(g["coin"])[0], g["i"].to_numpy(np.int64), g[f"j_{ex}"].to_numpy(np.int64),
                                   g[f"k_{ex}"].to_numpy(np.int64)) & np.isfinite(r)
                f_ = fr[g.index.to_numpy()][take]
                ok = np.isfinite(f_)
                for per in ("全期", "2020-23", "2024+"):
                    pm = period_mask(g["time"][take], per) & ok
                    if pm.sum() == 0:
                        continue
                    frows.append({"tf": tf, "strategy": name, "exit": ex, "period": per, "trades": int(pm.sum()),
                                  "r": float(r[take][pm].mean()), "funding_r": float(f_[pm].mean()),
                                  "r_with_funding": float((r[take][pm] + f_[pm]).mean()),
                                  "long_funding_r": float(f_[pm][g["side"].to_numpy()[take][pm] > 0].mean())
                                  if (g["side"].to_numpy()[take][pm] > 0).any() else np.nan})
    pd.DataFrame(frows).to_csv(out / "funding.csv", index=False)
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=float))
    print(json.dumps(summary, ensure_ascii=False, indent=1, default=float)[:6000])


# ---------------------------------------------------------------- 獲利從哪裡來：大盤 vs 個別幣

def market_panel(tf: str) -> pd.DataFrame:
    """每根 K 棒（開盤時間）的大盤狀態：等權指數（累積對數報酬）、站上 EMA50 的幣的比例（市場寬度）、有資料的幣數。"""
    closes, above = {}, {}
    for coin in TOP100:
        df = load_frame(coin, tf)
        if len(df) < WARMUP + 100:
            continue
        t = df["time"].dt.tz_convert(None)
        c = df["close"].astype(float)
        closes[coin] = pd.Series(c.to_numpy(), index=t)
        a = pd.Series((c > ind.ema(c, 50)).astype(float).to_numpy(), index=t)
        a.iloc[:WARMUP] = np.nan
        above[coin] = a
    C = pd.DataFrame(closes).sort_index()
    ew = np.log(C).diff().mean(axis=1, skipna=True).fillna(0.0).cumsum()
    breadth = pd.DataFrame(above).reindex(C.index).mean(axis=1, skipna=True)
    return pd.DataFrame({"ew": ew, "breadth": breadth, "coins": C.notna().sum(axis=1)})


def _take(g: pd.DataFrame, ex: str, mask: np.ndarray) -> np.ndarray:
    """先用 mask 篩掉訊號，再套同幣一張；回傳和 g 一樣長的布林陣列。"""
    out = np.zeros(len(g), bool)
    ii = np.flatnonzero(mask & np.isfinite(g[f"r_{ex}"].to_numpy()))
    if len(ii) == 0:
        return out
    h = g.iloc[ii]
    t = busy_filter(pd.factorize(h["coin"])[0], h["i"].to_numpy(np.int64), h[f"j_{ex}"].to_numpy(np.int64),
                    h[f"k_{ex}"].to_numpy(np.int64))
    out[ii[t]] = True
    return out


def _quick(g: pd.DataFrame, ex: str, take: np.ndarray, start, end) -> dict:
    r = g[f"r_{ex}"].to_numpy()[take]
    if len(r) == 0:
        return {"trades": 0}
    t_in = g["time"].to_numpy()[take]
    side = g["side"].to_numpy()[take]
    key = (pd.DatetimeIndex(t_in).floor("D").astype(str) + side.astype(str)).to_numpy()
    mean, se, cm, cse, ncl = cluster_stats(r, key)
    acc = account(r, t_in, g[f"exit_t_{ex}"].to_numpy()[take], start, end)
    return {"trades": len(r), "win_rate": (r > 0).mean(), "exp_r": mean, "se_cluster": se, "cluster_r": cm,
            "cluster_se": cse, **acc}


def mechanism(out: Path) -> None:
    """1. 同一根 K 棒有幾個幣一起出同方向訊號 → 每筆 R；2. 固定 3 天報酬拆成大盤和個別幣兩部分；
    3. 事先定好的兩條規則：順勢類只做「市場寬度同方向」的訊號、反轉類只做「同一根最多 2 個幣」的訊號。"""
    res = {}
    for tf in ("4H", "2H"):
        if not (out / f"signals_{tf}.pkl").exists():
            continue
        bar = pd.Timedelta(hours=BAR_HOURS[tf])
        panel = market_panel(tf)
        panel.to_pickle(out / f"market_panel_{tf}.pkl")
        sig = pd.read_pickle(out / f"signals_{tf}.pkl")
        sig["time"] = pd.to_datetime(sig["time"])
        sig = sig.sort_values(["strategy", "coin", "i"], kind="stable").reset_index(drop=True)
        sig["n_same"] = sig.groupby(["strategy", "time", "side"])["coin"].transform("size")
        bt = sig["time"] - bar  # 訊號那根的開盤時間
        pm = panel.reindex(bt.to_numpy())
        sig["breadth"] = pm["breadth"].to_numpy()
        sig["coins_now"] = pm["coins"].to_numpy()
        bpd = 24 // BAR_HOURS[tf]
        ew = panel["ew"]
        for hd in (3,):
            fut = ew.reindex((bt + pd.Timedelta(hours=hd * 24)).to_numpy()).to_numpy()
            now = pm["ew"].to_numpy()
            mret = np.exp(fut - now) - 1
            sig[f"mkt{hd}d"] = sig["side"].to_numpy() * mret / sig["atr_pct"].to_numpy()
        start, end = sig["time"].min(), pd.to_datetime(sig["exit_t_live"]).max()
        bounds = {"全期": (start, end), "2020-23": (start, SPLIT), "2024+": (SPLIT, end)}
        brows, drows, frows = [], [], []
        bins = [0, 1, 3, 7, 10 ** 6]
        labels = ["1", "2-3", "4-7", "8+"]
        for name, g in sig.groupby("strategy", sort=False):
            if name not in DEFAULTS:
                continue
            fam = BY_NAME[name]["family"]
            g = g.reset_index(drop=True)
            take = _take(g, "live", np.ones(len(g), bool))
            bucket = pd.cut(g["n_same"], bins, labels=labels).astype(str).to_numpy()
            side = g["side"].to_numpy()
            br = g["breadth"].to_numpy()
            aligned = np.where(side > 0, br >= 0.6, br <= 0.4)
            against = np.where(side > 0, br <= 0.4, br >= 0.6)
            for per, (a, b) in bounds.items():
                pmask = period_mask(g["time"], per)
                r = g["r_live"].to_numpy()
                for lab in labels:
                    m = take & pmask & (bucket == lab)
                    if m.sum():
                        brows.append({"tf": tf, "strategy": name, "family": fam, "period": per, "n_same": lab,
                                      "trades": int(m.sum()), "share": m.sum() / max((take & pmask).sum(), 1),
                                      "exp_r": r[m].mean(), "win_rate": (r[m] > 0).mean()})
                for lab, m0 in (("寬度同向", aligned), ("寬度中性", ~aligned & ~against), ("寬度反向", against)):
                    m = take & pmask & m0
                    if m.sum():
                        brows.append({"tf": tf, "strategy": name, "family": fam, "period": per, "breadth": lab,
                                      "trades": int(m.sum()), "share": m.sum() / max((take & pmask).sum(), 1),
                                      "exp_r": r[m].mean(), "win_rate": (r[m] > 0).mean()})
                v = g["fwd3d"].to_numpy()
                mk = g["mkt3d"].to_numpy()
                ok = pmask & np.isfinite(v) & np.isfinite(mk)
                if ok.sum() > 30:
                    drows.append({"tf": tf, "strategy": name, "family": fam, "period": per, "signals": int(ok.sum()),
                                  "fwd3d": v[ok].mean(), "market_part": mk[ok].mean(), "coin_part": (v[ok] - mk[ok]).mean(),
                                  "coin_part_isolated": (v[ok & (g["n_same"].to_numpy() <= 2)]
                                                         - mk[ok & (g["n_same"].to_numpy() <= 2)]).mean()})
            # 事先定好的規則
            rules = {}
            if fam in ("trend", "vegas"):
                rules["寬度同向"] = aligned
            if fam in ("reversal", "pullback", "divergence"):
                rules["同根最多2個"] = g["n_same"].to_numpy() <= 2
                rules["寬度不極端"] = np.where(side > 0, br > 0.25, br < 0.75)
            rules["全部"] = np.ones(len(g), bool)
            for rn, m0 in rules.items():
                tk = _take(g, "live", m0)
                for per, (a, b) in bounds.items():
                    pmask = period_mask(g["time"], per)
                    frows.append({"tf": tf, "strategy": name, "family": fam, "rule": rn, "period": per,
                                  **_quick(g, "live", tk & pmask, a, b)})
        pd.DataFrame(brows).to_csv(out / f"mech_buckets_{tf}.csv", index=False)
        pd.DataFrame(drows).to_csv(out / f"mech_decomp_{tf}.csv", index=False)
        ft = pd.DataFrame(frows)
        ft.to_csv(out / f"mech_rules_{tf}.csv", index=False)
        fam_sum = ft.groupby(["family", "rule", "period"])[["exp_r", "cluster_r", "sharpe", "trades"]].mean()
        res[tf] = fam_sum.reset_index().to_dict("records")
        print(fam_sum.to_string())
    (out / "mechanism.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float))


# ---------------------------------------------------------------- 穩健性：區塊重抽、多重檢定、少數幾天、帳戶層級對隨機

MAJORS = ["BTC", "ETH", "BNB", "XRP", "SOL", "DOGE", "ADA", "LINK", "LTC", "BCH"]  # 歷史最久的主流幣
BOOT_REPS = 2000
BOOT_BLOCK = 20  # 平均區塊長度（天）


def stationary_bootstrap_idx(T: int, reps: int, mean_block: float, seed: int = 0) -> np.ndarray:
    """Politis-Romano 平穩區塊重抽的索引 (reps, T)：每一天有 1/mean_block 的機率跳到隨機一天，否則接著前一天。"""
    rng = np.random.default_rng(seed)
    idx = np.empty((reps, T), np.int64)
    idx[:, 0] = rng.integers(0, T, reps)
    jump = rng.random((reps, T)) < 1.0 / mean_block
    rnd = rng.integers(0, T, (reps, T))
    for t in range(1, T):
        idx[:, t] = np.where(jump[:, t], rnd[:, t], (idx[:, t - 1] + 1) % T)
    return idx


def boot_sharpe(X: np.ndarray, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """每次重抽的年化 Sharpe（給信賴區間）和扣掉樣本平均後的 Sharpe（虛無假設：真實 Sharpe = 0，給多重檢定）。"""
    mu = X.mean(0)
    raw = np.empty((len(idx), X.shape[1]))
    null = np.empty_like(raw)
    for b, ii in enumerate(idx):
        xb = X[ii]
        sd = xb.std(0)
        sd = np.where(sd > 0, sd, np.nan)
        m = xb.mean(0)
        raw[b] = m / sd
        null[b] = (m - mu) / sd
    return raw * math.sqrt(365), null * math.sqrt(365)


def _daily_vec(r: np.ndarray, t_out: np.ndarray, days: pd.DatetimeIndex) -> np.ndarray:
    s = pd.Series(0.0025 * r, index=pd.DatetimeIndex(t_out).floor("D")).groupby(level=0).sum()
    return s.reindex(days, fill_value=0.0).to_numpy()


def _open_positions(t_in: np.ndarray, t_out: np.ndarray, days: pd.DatetimeIndex) -> np.ndarray:
    """每天收盤時還開著的單數。"""
    ends = (days + pd.Timedelta(days=1)).to_numpy()
    a = np.searchsorted(np.sort(t_in), ends, side="left")
    b = np.searchsorted(np.sort(t_out), ends, side="left")
    return a - b


def robust(out: Path) -> None:
    """1. 每個設定的每日損益做區塊重抽：Sharpe 信賴區間、單一檢定 p 值、Romano-Wolf 多重檢定調整後 p 值
    （312 個設定一起比：樣本內最好的那個是不是也只是運氣）；2. 拿掉最好的 1% 天數後還剩多少；
    3. 帳戶層級的「進場時機價值」：每筆 R 扣掉隨機進場（同幣同方向同年）後的每日 Sharpe；
    4. 只做主流幣 / 只做 BTC、ETH；5. 同一根 K 棒同方向出訊號的幣數過濾；6. 同時持倉數；
    7. 和實盤維加斯一起做會不會更好；8. 大盤大漲大跌的日子賺多少（順勢 = 買波動）。"""
    dd = pd.read_pickle(out / "daily_matrix.pkl")
    days_all = pd.date_range(dd.index.min(), dd.index.max(), freq="D")
    full = dd.reindex(days_all, fill_value=0.0)
    full = full[full.index < pd.Timestamp("2026-10-01")]
    cfgs = list(full.columns)
    rows = []
    reality = {}
    for per, (lo, hi) in PERIODS.items():
        m = np.ones(len(full), bool)
        if lo is not None:
            m &= full.index >= lo
        if hi is not None:
            m &= full.index < hi
        X = full[m].to_numpy()
        T = len(X)
        sd = X.std(0)
        sr = X.mean(0) / np.where(sd > 0, sd, np.nan) * math.sqrt(365)
        idx = stationary_bootstrap_idx(T, BOOT_REPS, BOOT_BLOCK, seed=1)
        raw, null = boot_sharpe(X, idx)
        mx = np.nanmax(null, axis=1)
        k = max(int(math.ceil(0.01 * T)), 1)
        reality[per] = {"days": T, "configs": len(cfgs), "best": list(cfgs[int(np.nanargmax(sr))]),
                        "best_sr": float(np.nanmax(sr)), "p_best": float((mx >= np.nanmax(sr)).mean()),
                        "null_max_sr_95": float(np.nanpercentile(mx, 95))}
        for c, cfg in enumerate(cfgs):
            x = X[:, c]
            if not np.isfinite(sr[c]):
                continue
            order = np.argsort(x)[::-1]
            top = order[:k]
            rest = np.delete(x, top)
            tot = x.sum()
            rows.append({"tf": cfg[0], "strategy": cfg[1], "exit": cfg[2], "period": per, "sr": sr[c],
                         "ci5": np.nanpercentile(raw[:, c], 5), "ci95": np.nanpercentile(raw[:, c], 95),
                         "p_boot": float((raw[:, c] <= 0).mean()), "p_adj": float((mx >= sr[c]).mean()),
                         "top1pct_share": x[top].sum() / tot if tot > 0 else np.nan,
                         "sr_ex_top1pct": rest.mean() / rest.std() * math.sqrt(365) if rest.std() > 0 else np.nan,
                         "sr_ex_top10": (np.delete(x, order[:10]).mean() / np.delete(x, order[:10]).std() * math.sqrt(365))})
    boot = pd.DataFrame(rows)
    boot.to_csv(out / "robust_boot.csv", index=False)

    trows, srows, nrows, crows, combo, strad = [], [], [], [], [], []
    for tf in ("4H", "2H"):
        if not (out / f"signals_{tf}.pkl").exists():
            continue
        sig = pd.read_pickle(out / f"signals_{tf}.pkl")
        base = pd.read_pickle(out / f"baseline_{tf}.pkl")
        sig["time"] = pd.to_datetime(sig["time"])
        sig = sig.sort_values(["strategy", "coin", "i"], kind="stable").reset_index(drop=True)
        sig["n_same"] = sig.groupby(["strategy", "time", "side"])["coin"].transform("size")
        by, _ = base_lookup(sig, base, "r_live", "year")
        sig["base_live"] = by
        start = sig["time"].min()
        end = min(pd.to_datetime(sig["exit_t_live"]).max(), pd.Timestamp("2026-10-01"))
        days = pd.date_range(start.floor("D"), end.floor("D"), freq="D")
        bounds = {"全期": (start, end), "2020-23": (start, SPLIT), "2024+": (SPLIT, end)}
        dmask = {per: ((days >= (lo if lo is not None else days[0])) & (days < (hi if hi is not None else days[-1] + pd.Timedelta(days=1))))
                 for per, (lo, hi) in PERIODS.items()}
        # 大盤每天漲跌（等權指數）
        panel_p = out / f"market_panel_{tf}.pkl"
        mkt = None
        if panel_p.exists():
            ew = pd.read_pickle(panel_p)["ew"]
            mkt = ew.groupby(ew.index.floor("D")).last().diff().reindex(days).to_numpy()
        vegas_daily = {}
        for name, g in sig.groupby("strategy", sort=False):
            if name not in DEFAULTS:
                continue
            g = g.reset_index(drop=True)
            fam = BY_NAME[name]["family"]
            for ex in ("live", "natural"):
                r = g[f"r_{ex}"].to_numpy()
                if not np.isfinite(r).any():
                    continue
                take = _take(g, ex, np.ones(len(g), bool))
                t_in = g["time"].to_numpy()
                t_out = pd.to_datetime(g[f"exit_t_{ex}"]).to_numpy()
                dv = _daily_vec(r[take], t_out[take], days)
                if ex == "live":
                    vegas_daily[name] = dv
                    e = (r - g["base_live"].to_numpy())
                    ok = take & np.isfinite(e)
                    de = _daily_vec(e[ok], t_out[ok], days)
                    for per in PERIODS:
                        x = de[dmask[per]]
                        trows.append({"tf": tf, "strategy": name, "family": fam, "period": per,
                                      "edge_r": float(np.nanmean(e[ok & period_mask(g["time"], per)])),
                                      "timing_sr": x.mean() / x.std() * math.sqrt(365) if x.std() > 0 else np.nan})
                # 同時持倉
                op = _open_positions(t_in[take], t_out[take], days)
                crows.append({"tf": tf, "strategy": name, "exit": ex, "trades_per_day": take.sum() / len(days),
                              "open_mean": op.mean(), "open_p95": np.percentile(op, 95), "open_max": op.max(),
                              "avg_hold_days": float(np.mean((t_out[take] - t_in[take]) / np.timedelta64(1, "D")))})
                # 大盤大漲大跌的日子
                if mkt is not None:
                    am = np.abs(mkt)
                    for per in PERIODS:
                        mm = dmask[per] & np.isfinite(am)
                        q = np.nanpercentile(am[mm], 95)
                        big = mm & (am >= q)
                        strad.append({"tf": tf, "strategy": name, "family": fam, "exit": ex, "period": per,
                                      "pnl_share_big5pct_days": dv[big].sum() / dv[mm].sum() if dv[mm].sum() != 0 else np.nan,
                                      "mean_big": dv[big].mean() / 0.0025, "mean_other": dv[mm & ~big].mean() / 0.0025,
                                      "corr_abs_mkt": np.corrcoef(dv[mm], am[mm])[0, 1],
                                      "corr_mkt": np.corrcoef(dv[mm], mkt[mm])[0, 1]})
                # 幣種子集
                coin = g["coin"].to_numpy()
                subsets = {"全部": np.ones(len(g), bool), "主流10幣": np.isin(coin, MAJORS),
                           "BTC+ETH": np.isin(coin, ["BTC", "ETH"]), "BTC": coin == "BTC"}
                ns = g["n_same"].to_numpy()
                rules = {"全部": np.ones(len(g), bool), "同根只有1個": ns == 1, "同根2個以上": ns >= 2,
                         "同根4個以上": ns >= 4, "同根8個以上": ns >= 8}
                for kind, dct, sink in (("subset", subsets, srows), ("rule", rules, nrows)):
                    for lab, m0 in dct.items():
                        tk = _take(g, ex, m0)
                        for per, (a, b) in bounds.items():
                            q_ = _quick(g, ex, tk & period_mask(g["time"], per), a, b)
                            sink.append({"tf": tf, "strategy": name, "family": fam, "exit": ex, kind: lab,
                                         "period": per, **q_})
        # 和實盤維加斯一起做
        if "vegas_live" in vegas_daily:
            v = vegas_daily["vegas_live"]
            for name, x in vegas_daily.items():
                if name == "vegas_live":
                    continue
                for per in PERIODS:
                    mm = dmask[per]
                    a_, b_ = v[mm], x[mm]
                    s = a_ + b_
                    combo.append({"tf": tf, "strategy": name, "period": per,
                                  "sr_vegas": a_.mean() / a_.std() * math.sqrt(365),
                                  "sr_x": b_.mean() / b_.std() * math.sqrt(365) if b_.std() > 0 else np.nan,
                                  "sr_combo": s.mean() / s.std() * math.sqrt(365), "corr": np.corrcoef(a_, b_)[0, 1]})
    pd.DataFrame(trows).to_csv(out / "robust_timing.csv", index=False)
    pd.DataFrame(srows).to_csv(out / "robust_subsets.csv", index=False)
    pd.DataFrame(nrows).to_csv(out / "robust_nsame.csv", index=False)
    pd.DataFrame(crows).to_csv(out / "robust_conc.csv", index=False)
    pd.DataFrame(combo).to_csv(out / "robust_combo.csv", index=False)
    pd.DataFrame(strad).to_csv(out / "robust_straddle.csv", index=False)
    (out / "robust_reality.json").write_text(json.dumps(reality, ensure_ascii=False, indent=1, default=float))
    print(json.dumps(reality, ensure_ascii=False, indent=1, default=float))


CAP_CANDIDATES = [("4H", "vegas_live", "live"), ("2H", "vegas_live", "live"), ("4H", "rsi_50", "live"),
                  ("4H", "price_ema50", "live"), ("4H", "price_sma200", "natural"), ("4H", "vegas_break", "natural"),
                  ("4H", "ema_20_50", "natural"), ("2H", "sma_20_50", "natural"), ("2H", "ema_20_50", "natural")]


def _cap_take(t_in, t_out, coin_rank, coin_id, r, k):
    """依時間順序（同一根依市值排名）處理訊號：該幣沒有持倉、而且今天開的單還不到 k 張才做。"""
    order = np.lexsort((coin_rank, t_in))
    busy_until = {}
    per_day = {}
    take = np.zeros(len(t_in), bool)
    days = t_in.astype("datetime64[D]")
    for s in order:
        if not np.isfinite(r[s]):
            continue
        c = coin_id[s]
        if busy_until.get(c, np.datetime64("NaT", "ns")) > t_in[s]:
            continue
        d = days[s]
        if per_day.get(d, 0) >= k:
            continue
        take[s] = True
        per_day[d] = per_day.get(d, 0) + 1
        busy_until[c] = t_out[s]
    return take


def daily_cap(out: Path) -> None:
    """手動下單做不了每個訊號：每天最多開 k 張（先出的先做，同一根先做市值大的）時，每筆、每天、帳戶層級各是多少。"""
    rank = {c: i for i, c in enumerate(TOP100)}
    rows = []
    for tf in ("4H", "2H"):
        cands = [c for c in CAP_CANDIDATES if c[0] == tf]
        if not cands or not (out / f"signals_{tf}.pkl").exists():
            continue
        sig = pd.read_pickle(out / f"signals_{tf}.pkl")
        sig["time"] = pd.to_datetime(sig["time"])
        start = sig["time"].min()
        end = min(pd.to_datetime(sig["exit_t_live"]).max(), pd.Timestamp("2026-10-01"))
        bounds = {"全期": (start, end), "2020-23": (start, SPLIT), "2024+": (SPLIT, end)}
        for _, name, ex in cands:
            g = sig[sig["strategy"] == name].sort_values(["coin", "i"]).reset_index(drop=True)
            t_in = g["time"].to_numpy()
            t_out = pd.to_datetime(g[f"exit_t_{ex}"]).to_numpy()
            r = g[f"r_{ex}"].to_numpy()
            cid = pd.factorize(g["coin"])[0]
            crank = g["coin"].map(rank).fillna(999).to_numpy()
            for k in (1, 2, 3, 5, 10 ** 6):
                tk = _cap_take(t_in, t_out, crank, cid, r, k)
                for per, (a, b) in bounds.items():
                    q_ = _quick(g, ex, tk & period_mask(g["time"], per), a, b)
                    rows.append({"tf": tf, "strategy": name, "exit": ex, "cap": "不限" if k > 1000 else str(k),
                                 "period": per, "per_day": q_.get("trades", 0) / max((b - a).days, 1), **q_})
    tab = pd.DataFrame(rows)
    tab.to_csv(out / "daily_cap.csv", index=False)
    print(tab.pivot_table(index=["tf", "strategy", "exit", "cap"], columns="period",
                          values=["exp_r", "sharpe", "per_day"]).round(3).to_string())


def cap_all(out: Path) -> None:
    """所有預設策略（live、natural）每天最多 1 / 3 張的結果（cap_all.csv），以及「那天同方向一共出了幾個訊號」
    分組的每筆 R（cluster_size.csv）。一天會出幾個訊號要等那天結束才知道，所以後者只用來解釋「每注」等權平均
    為什麼和實際做得到的不一樣，不能當進場條件。"""
    rank = {c: i for i, c in enumerate(TOP100)}
    rows, crow = [], []
    for tf in ("4H", "2H"):
        if not (out / f"signals_{tf}.pkl").exists():
            continue
        sig = pd.read_pickle(out / f"signals_{tf}.pkl")
        sig["time"] = pd.to_datetime(sig["time"])
        start = sig["time"].min()
        end = min(pd.to_datetime(sig["exit_t_live"]).max(), pd.Timestamp("2026-10-01"))
        bounds = {"全期": (start, end), "2020-23": (start, SPLIT), "2024+": (SPLIT, end)}
        for name in DEFAULTS:
            g = sig[sig["strategy"] == name].sort_values(["coin", "i"]).reset_index(drop=True)
            fam = BY_NAME[name]["family"]
            for ex in ("live", "natural"):
                r = g[f"r_{ex}"].to_numpy()
                if not np.isfinite(r).any():
                    continue
                t_in = g["time"].to_numpy()
                t_out = pd.to_datetime(g[f"exit_t_{ex}"]).to_numpy()
                cid = pd.factorize(g["coin"])[0]
                crank = g["coin"].map(rank).fillna(999).to_numpy()
                for k in (1, 3):
                    tk = _cap_take(t_in, t_out, crank, cid, r, k)
                    for per, (a, b) in bounds.items():
                        rows.append({"tf": tf, "strategy": name, "family": fam, "exit": ex, "cap": k, "period": per,
                                     **_quick(g, ex, tk & period_mask(g["time"], per), a, b)})
                h = g[_take(g, ex, np.ones(len(g), bool))].copy()
                h["day"] = h["time"].dt.floor("D")
                size = h.groupby(["day", "side"])["coin"].transform("size")
                h["bucket"] = pd.cut(size, [0, 1, 3, 7, 15, 10 ** 6], labels=["1", "2-3", "4-7", "8-15", "16+"]).astype(str)
                h["per"] = np.where(h["time"] < SPLIT, "2020-23", "2024+")
                for (per, bk), q in h.groupby(["per", "bucket"]):
                    crow.append({"tf": tf, "strategy": name, "family": fam, "exit": ex, "period": per, "day_signals": bk,
                                 "trades": len(q), "exp_r": q[f"r_{ex}"].mean(), "win_rate": (q[f"r_{ex}"] > 0).mean()})
    tab = pd.DataFrame(rows)
    tab.to_csv(out / "cap_all.csv", index=False)
    pd.DataFrame(crow).to_csv(out / "cluster_size.csv", index=False)
    s = tab[tab.period != "全期"].groupby(["tf", "family", "exit", "cap", "period"])[["win_rate", "exp_r", "sharpe"]].mean()
    print(s.unstack("period").round(3).to_string())


def btc_regime(days: int) -> pd.Series:
    """BTC 日線收盤在 days 日均線之上 = 1（多頭）、之下 = -1。索引是那一天（UTC），那天收盤後才知道。"""
    df = load_frame("BTC", "4H")
    d = df.set_index(df["time"].dt.tz_convert(None))["close"].resample("1D").last().dropna()
    ma = d.rolling(days).mean()
    return pd.Series(np.where(d > ma, 1, -1), index=d.index).where(ma.notna())


def regime_filter(out: Path) -> None:
    """事先定好的大盤多空過濾（看結果前寫好、不調參數）：BTC 前一天日線收盤在 200 日均線之上只做多、之下只做空；
    50 日均線當敏感度對照。所有預設策略、live 和 natural 出場都跑。"""
    rows = []
    for tf in ("4H", "2H"):
        if not (out / f"signals_{tf}.pkl").exists():
            continue
        sig = pd.read_pickle(out / f"signals_{tf}.pkl")
        sig["time"] = pd.to_datetime(sig["time"])
        sig = sig[sig["strategy"].isin(DEFAULTS)]
        sig = sig.sort_values(["strategy", "coin", "i"], kind="stable").reset_index(drop=True)
        done = (sig["time"].dt.floor("D") - pd.Timedelta(days=1)).to_numpy()  # 訊號當下最近一根已收盤的日線
        reg = {n: btc_regime(n).reindex(done).to_numpy() for n in (200, 50)}
        start = sig["time"].min()
        end = min(pd.to_datetime(sig["exit_t_live"]).max(), pd.Timestamp("2026-10-01"))
        bounds = {"全期": (start, end), "2020-23": (start, SPLIT), "2024+": (SPLIT, end)}
        for name, g in sig.groupby("strategy", sort=False):
            gi = g.index.to_numpy()
            g = g.reset_index(drop=True)
            side = g["side"].to_numpy()
            rules = {"全部": np.ones(len(g), bool),
                     "BTC 200日線順向": side == reg[200][gi], "BTC 50日線順向": side == reg[50][gi]}
            for ex in ("live", "natural"):
                if not np.isfinite(g[f"r_{ex}"].to_numpy()).any():
                    continue
                for rn, m0 in rules.items():
                    tk = _take(g, ex, m0)
                    for per, (a, b) in bounds.items():
                        rows.append({"tf": tf, "strategy": name, "family": BY_NAME[name]["family"], "exit": ex,
                                     "rule": rn, "period": per, **_quick(g, ex, tk & period_mask(g["time"], per), a, b)})
    tab = pd.DataFrame(rows)
    tab.to_csv(out / "regime_filter.csv", index=False)
    s = tab[tab.period != "全期"].groupby(["tf", "family", "exit", "rule", "period"])[["exp_r", "cluster_r", "sharpe"]].mean()
    print(s.unstack("period").round(3).to_string())


def pionex_check(out: Path) -> None:
    """派網 K 線複核：每個幣只比兩邊都有資料的同一段時間（派網第一筆訊號 ~ 2026-09-30），同幣一張。"""
    rows = []
    end = pd.Timestamp("2026-10-01")
    for tf in ("4H", "2H"):
        pp = out / f"signals_{tf}_pionex.pkl"
        if not pp.exists():
            continue
        sp = pd.read_pickle(pp)
        sb = pd.read_pickle(out / f"signals_{tf}.pkl")
        for s_ in (sp, sb):
            s_["time"] = pd.to_datetime(s_["time"])
        first = sp.groupby("coin")["time"].min()
        sb = sb[sb["coin"].isin(first.index)]
        sb = sb[sb["time"].to_numpy() >= first.reindex(sb["coin"]).to_numpy()]
        sp = sp[sp["coin"].isin(sb["coin"].unique())]
        start = max(first.min(), pd.Timestamp("2020-01-01"))
        for src, s_ in (("幣安", sb), ("派網", sp)):
            s_ = s_[s_["time"] < end]
            s_ = s_.sort_values(["strategy", "coin", "i"], kind="stable").reset_index(drop=True)
            for name, g in s_.groupby("strategy", sort=False):
                if name not in DEFAULTS:
                    continue
                g = g.reset_index(drop=True)
                for ex in ("live", "natural"):
                    if not np.isfinite(g[f"r_{ex}"].to_numpy()).any():
                        continue
                    tk = _take(g, ex, np.ones(len(g), bool))
                    rows.append({"tf": tf, "source": src, "strategy": name, "exit": ex, "start": str(start.date()),
                                 "coins": g["coin"].nunique(), **_quick(g, ex, tk, start, end)})
    tab = pd.DataFrame(rows)
    tab.to_csv(out / "pionex_check.csv", index=False)
    if len(tab):
        p = tab.pivot_table(index=["tf", "exit", "strategy"], columns="source", values=["exp_r", "cluster_r", "sharpe", "trades"])
        print(p.round(3).to_string())


def fetch_klines() -> None:
    """幣安 4h（4H 用）和 1h（合成 2H 用）月檔 K 線，已經下載過的幣直接讀快取。"""
    from grid_trading.binance_history import fetch_binance

    def job(args):
        coin, iv = args
        try:
            return coin, iv, len(fetch_binance(coin, iv)), ""
        except Exception as e:  # 單一個幣失敗不影響其他幣
            return coin, iv, 0, repr(e)

    with ThreadPoolExecutor(6) as ex:
        for coin, iv, n, err in ex.map(job, [(c, iv) for iv in ("4h", "1h") for c in TOP100]):
            print(coin, iv, n, err, flush=True)


EXPORT_COPIES = {
    "daily_cap.csv": "daily_cap.csv", "cap_all.csv": "daily_cap_all_rules.csv", "cluster_size.csv": "day_signal_count.csv",
    "pionex_check.csv": "pionex_check.csv", "regime_filter.csv": "btc_regime_filter.csv",
    "robust_boot.csv": "bootstrap_sharpe.csv", "robust_subsets.csv": "coin_subsets.csv", "robust_nsame.csv": "same_bar_count.csv",
    "robust_conc.csv": "open_positions.csv", "robust_combo.csv": "combo_with_vegas.csv", "market_vr.csv": "market_variance_ratio.csv",
    "market_tsmom.csv": "market_tsmom.csv", "mech_decomp_4H.csv": "decomp_4H.csv", "mech_decomp_2H.csv": "decomp_2H.csv",
    "info_4H.csv": "signal_info_4H.csv", "info_2H.csv": "signal_info_2H.csv", "funding.csv": "funding.csv",
    "selection.csv": "preregistered_selection.csv", "corr_4H.csv": "corr_4H.csv", "corr_2H.csv": "corr_2H.csv",
    "robust_reality.json": "reality_check.json", "summary.json": "overfit_summary.json",
}


def export(out: Path, dst: Path, prefix: str) -> None:
    """一張中文欄位的彙總表（每個週期 × 策略 × 出場一列：全期、2020-23、2024 起各段的勝率、每筆、每注、對隨機、
    夏普、滑價、校正後 p、扣掉最好 1% 天數的夏普、派網同期），加上各項分析結果的複本。要先跑完上面所有步驟。"""
    dst.mkdir(parents=True, exist_ok=True)
    boot = pd.read_csv(out / "robust_boot.csv")
    pio = pd.read_csv(out / "pionex_check.csv")
    rows = []
    for tf in ("4H", "2H"):
        t = pd.read_csv(out / f"table_{tf}.csv")
        ts = pd.read_csv(out / f"table_{tf}_slip0.0005.csv")
        for (name, ex), g in t.groupby(["strategy", "exit"], sort=False):
            p = g.set_index("period")
            if "全期" not in p.index or p.loc["全期", "trades"] == 0 or not np.isfinite(p.loc["全期", "exp_r"]):
                continue
            st = BY_NAME[name]
            s = ts[(ts.strategy == name) & (ts.exit == ex)].set_index("period")
            bb = boot[(boot.tf == tf) & (boot.strategy == name) & (boot.exit == ex)].set_index("period")
            pp = pio[(pio.tf == tf) & (pio.strategy == name) & (pio.exit == ex)].set_index("source")
            r = {"週期": tf, "代號": name, "策略": st["zh"], "類別": FAMILY_ZH[st["family"]], "預設參數": st["base"] is None,
                 "出場": ex}
            for per, lab in (("全期", ""), ("2020-23", "_2020_23"), ("2024+", "_2024起")):
                if per not in p.index:
                    continue
                x = p.loc[per]
                r.update({f"筆數{lab}": int(x.trades), f"每天筆數{lab}": x.per_day, f"勝率{lab}": x.win_rate,
                          f"每筆R{lab}": x.exp_r, f"每注R{lab}": x.cluster_r,
                          f"每注R_t{lab}": x.cluster_r / x.cluster_se if x.cluster_se > 0 else np.nan,
                          f"對隨機每筆{lab}": x.edge_year, f"對隨機每注{lab}": x.edge_year_cluster,
                          f"隨機勝率{lab}": x.base_win_year, f"夏普{lab}": x.sharpe, f"最大回撤%{lab}": x.max_dd_pct})
                if per in s.index:
                    r[f"加滑價夏普{lab}"] = s.loc[per, "sharpe"]
                    r[f"加滑價每筆R{lab}"] = s.loc[per, "exp_r"]
                if per in bb.index:
                    r[f"校正後p{lab}"] = bb.loc[per, "p_adj"]
                    r[f"扣最好1%天夏普{lab}"] = bb.loc[per, "sr_ex_top1pct"]
            if {"派網", "幣安"} <= set(pp.index):
                r.update({"派網同期夏普": pp.loc["派網", "sharpe"], "幣安同期夏普": pp.loc["幣安", "sharpe"],
                          "派網同期每筆R": pp.loc["派網", "exp_r"], "幣安同期每筆R": pp.loc["幣安", "exp_r"],
                          "派網對照起點": pp.loc["派網", "start"]})
            rows.append(r)
    pd.DataFrame(rows).to_csv(dst / f"{prefix}summary.csv", index=False, encoding="utf-8-sig")
    for src, name in EXPORT_COPIES.items():
        if src.endswith(".csv"):
            pd.read_csv(out / src).to_csv(dst / f"{prefix}{name}", index=False, encoding="utf-8-sig")
        else:
            (dst / f"{prefix}{name}").write_bytes((out / src).read_bytes())
    print(len(rows), "rows →", dst)


# ---------------------------------------------------------------- 主程式

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tf", default="4H", choices=["4H", "2H"])
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--market", action="store_true")
    ap.add_argument("--mechanism", action="store_true", help="獲利來自大盤還是個別幣、市場寬度規則")
    ap.add_argument("--robust", action="store_true", help="區塊重抽、多重檢定、幣種子集、同時持倉（要先跑 --analyze）")
    ap.add_argument("--pionex-check", action="store_true", help="派網 K 線和幣安同一段時間對照（要先跑 --run --pionex）")
    ap.add_argument("--regime", action="store_true", help="BTC 200 日均線多空過濾（事先定好的單一規則）")
    ap.add_argument("--cap", action="store_true", help="每天最多開 k 張時的結果（手動下單的實際限制）")
    ap.add_argument("--fetch", action="store_true", help="下載幣安 4h、1h 月檔 K 線")
    ap.add_argument("--fetch-funding", action="store_true")
    ap.add_argument("--export", type=Path, help="把彙總表和各項結果輸出到這個資料夾")
    ap.add_argument("--prefix", default="basic_indicators_", help="--export 的檔名前綴")
    ap.add_argument("--slip", type=float, default=0.0)
    ap.add_argument("--pionex", action="store_true", help="用派網 K 線（和 --run 一起用）")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.fetch:
        fetch_klines()
    if args.fetch_funding:
        with ThreadPoolExecutor(4) as ex:
            for coin, df in zip(TOP100, ex.map(fetch_funding, TOP100)):
                print(coin, len(df), flush=True)
    if args.run:
        run(args.tf, args.slip, args.workers, OUT, "pionex" if args.pionex else "binance")
    if args.market:
        market(OUT)
    if args.mechanism:
        mechanism(OUT)
    if args.analyze:
        analyze(OUT)
    if args.robust:
        robust(OUT)
    if args.pionex_check:
        pionex_check(OUT)
    if args.regime:
        regime_filter(OUT)
    if args.cap:
        daily_cap(OUT)
        cap_all(OUT)
    if args.export:
        export(OUT, args.export, args.prefix)
    return 0


def analyze(out: Path) -> None:
    for tf in ("4H", "2H"):
        if (out / f"signals_{tf}.pkl").exists():
            build_tables(tf, out)
            signal_info(tf, out)
        for extra in ("_slip0.0005", "_pionex"):
            if (out / f"signals_{tf}{extra}.pkl").exists():
                build_tables(tf, out, extra)
    analyze_more(out)


if __name__ == "__main__":
    sys.exit(main())
