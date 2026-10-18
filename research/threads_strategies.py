"""Threads 截圖裡的三個想法，拆成能回測的規則（只做分析，不改實盤規則）。

python -m research.threads_strategies signals    # 日線/4H 吞沒 + EMA8/20、1H/4H 鯊魚諧波 → output/threads_signals.csv
python -m research.threads_strategies bot        # 「AI 挑最近 14 天最強的幣」機器人 → output/threads_bot.csv

需要 numba（和 stop_research.py 共用模擬程式）。資料：幣安 USDT 永續 1H（data.binance.vision），合成 4H/日線。

1. 吞沒 + EMA8/20（latte.lin「日線 EMA8/20 死叉、日線看跌吞沒、向上回測日線開收水平」）：
   - 空：EMA8 < EMA20，而且最近 5 根內剛死叉（另測只要 EMA8 < EMA20）；這根是看跌吞沒
     （前一根收紅、這根收黑、這根開盤 >= 前一根收盤、這根收盤 < 前一根開盤）。多單完全鏡像。
   - 進場：收盤市價；或掛單在吞沒 K 實體一半（回測開收水平），3 根內沒成交就取消。
   - 停損：吞沒 K 的最高點外 0.1 ATR（至少 0.5 ATR）；另測 1.5 ATR（實盤寫法）。
   - 另測加上日線 EMA200 同向濾網。
   - 基準：同樣 EMA8 < EMA20 的狀態（或剛交叉 5 根內），每 5 根隨機抽一根市價進場（同樣停損算法），
     看吞沒有沒有多出來的東西。
2. 鯊魚諧波 0.886 / 1.13（latte.lin 的 PRZ）：
   - 用 ATR 門檻的 zigzag 找轉折 A、B、C；C 超過 A（BC 是 AB 的 1.13 倍以上，另測嚴格版 1.618~2.24 倍）。
   - C 確認後掛單在 BC 的 0.886 回撤，停損在 1.13 外 0.1 ATR；下一個轉折確認或 100 根內沒成交就取消。
   - 例：A 低、B 高、C 更低 → 反彈到 0.886 做空（截圖那種）；鏡像做多。
3. 出場（1~2 都測）：q135 = 實盤式 1/3/5R 各平 1/4、TP1 後保本、TP2 後移到 TP1、最後 1/4 追蹤 3 ATR；
   tp2 = 2R 全出；tp1 = 1R 全出。手續費每邊 0.05%。成交那根 K 棒同時碰到停損一律算停損（保守）。
   同一個幣同一個策略同時只持有一張。
4. AI 挑幣機器人（allencheng7575）：每天 00:00 UTC 在前 30 名（近 30 天成交額）裡挑最近 14 天漲最多、
   而且前 7 天和後 7 天都上漲的幣，全倉做多（1 倍）；-20% 全撤；BTC 24 小時跌超過 5% 就全部出場、
   當天不進場；被止損的幣 48 小時內不再選。持有天數測 1/3/7 天換一次和「抱到止損」。
   對照：同規則但隨機挑幣（200 次）、BTC 抱著不動、前 30 名等權重每天再平衡。
   注意：幣種名單是 2026-10 的市值前 100（倖存者偏差，做多會被高估），所以只看「和隨機挑幣比」。
"""

from __future__ import annotations

import sys
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from grid_trading import indicators as ind
from grid_trading.binance_history import load
from grid_trading.history import resample
from grid_trading.universe import TOP100
from research.stop_research import SPLIT, busy_filter, cluster_stats, sim_many

OUT = Path("output")
FEE = 0.0005
TFS = {"1D": ("1D", 24), "4H": ("4h", 4), "1H": (None, 1)}
EXITS = {  # 名稱: (TP 的 R 倍數, 平倉比例, 追蹤幾倍 ATR)
    "q135": ((1.0, 3.0, 5.0), (0.25, 0.25, 0.25), 3.0),
    "tp2": ((2.0,), (1.0,), 0.0),
    "tp1": ((1.0,), (1.0,), 0.0),
}
MAX_BARS = {"1D": 60, "4H": 240, "1H": 336}


def frame(coin: str, tf: str) -> pd.DataFrame | None:
    df = load(coin, "1h")
    if len(df) < 24 * 250:
        return None
    rule, _ = TFS[tf]
    if rule:
        df = resample(df, rule, origin="epoch")
    df = df.reset_index(drop=True)
    c = df["close"]
    df["atr"] = ind.atr(df)
    df["e8"], df["e20"] = ind.ema(c, 8), ind.ema(c, 20)
    # 日線 EMA200（大週期濾網），用收盤後才知道的值
    d = resample(load(coin, "1h"), "1D", origin="epoch")
    d["ema200"] = ind.ema(d["close"], 200).where(np.arange(len(d)) >= 200)
    d["avail"] = d["time"] + pd.Timedelta(days=1)
    bar_close = df["time"] + pd.Timedelta(hours=TFS[tf][1])
    m = pd.merge_asof(pd.DataFrame({"t": bar_close}), d[["avail", "close", "ema200"]], left_on="t",
                      right_on="avail", direction="backward")
    df["d_close"], df["d_ema200"] = m["close"].to_numpy(), m["ema200"].to_numpy()
    return df


# ---------- 進場訊號：每筆回傳 (訊號索引 i, 方向, 進場價, 停損價, 掛單起訖) ----------

def engulf_signals(df: pd.DataFrame, cross_recent: bool, window: int = 5) -> list[tuple]:
    o, c, h, l = (df[k].to_numpy() for k in ("open", "close", "high", "low"))
    e8, e20, atr = df["e8"].to_numpy(), df["e20"].to_numpy(), df["atr"].to_numpy()
    out = []
    below = e8 < e20
    for i in range(30, len(df)):
        for d in (1, -1):
            regime = (e8[i] > e20[i]) if d > 0 else below[i]
            if not regime:
                continue
            if cross_recent:
                prev = e8[i - window:i] - e20[i - window:i]
                if not ((prev * d) <= 0).any():
                    continue
            if d < 0:
                ok = c[i - 1] > o[i - 1] and c[i] < o[i] and o[i] >= c[i - 1] * 0.9995 and c[i] < o[i - 1]
            else:
                ok = c[i - 1] < o[i - 1] and c[i] > o[i] and o[i] <= c[i - 1] * 1.0005 and c[i] > o[i - 1]
            if ok:
                out.append((i, d))
    return out


def regime_random(df: pd.DataFrame, rng: np.random.Generator, cross_recent: bool = False) -> list[tuple]:
    """隨機基準：每 5 根抽一根，方向跟著 EMA8/20；cross_recent=True 只抽剛交叉 5 根內的 K 棒（不要求吞沒）。"""
    e8, e20 = df["e8"].to_numpy(), df["e20"].to_numpy()
    out = []
    for i in range(30, len(df)):
        if rng.random() < 0.2:
            d = 1 if e8[i] > e20[i] else -1
            if cross_recent and not ((d * (e8[i - 5:i] - e20[i - 5:i])) <= 0).any():
                continue
            out.append((i, d))
    return out


def zigzag(df: pd.DataFrame, k: float) -> list[tuple]:
    """ATR 門檻的 zigzag。回傳 (轉折索引, 價格, 方向 +1 高/-1 低, 確認的索引)。只用已收盤的 K 棒。"""
    h, l, atr = df["high"].to_numpy(), df["low"].to_numpy(), df["atr"].to_numpy()
    piv = []
    d = 0
    ext_i, ext_p = 0, h[0]
    lo_i, lo_p, hi_i, hi_p = 0, l[0], 0, h[0]
    for j in range(15, len(df)):
        th = k * atr[j]
        if d == 0:
            if h[j] > hi_p:
                hi_i, hi_p = j, h[j]
            if l[j] < lo_p:
                lo_i, lo_p = j, l[j]
            if hi_p - l[j] > th and hi_i < j:
                piv.append((hi_i, hi_p, 1, j))
                d, ext_i, ext_p = -1, j, l[j]
            elif h[j] - lo_p > th and lo_i < j:
                piv.append((lo_i, lo_p, -1, j))
                d, ext_i, ext_p = 1, j, h[j]
            continue
        if d > 0:
            if h[j] >= ext_p:
                ext_i, ext_p = j, h[j]
            elif ext_p - l[j] > th:
                piv.append((ext_i, ext_p, 1, j))
                d, ext_i, ext_p = -1, j, l[j]
        else:
            if l[j] <= ext_p:
                ext_i, ext_p = j, l[j]
            elif h[j] - ext_p > th:
                piv.append((ext_i, ext_p, -1, j))
                d, ext_i, ext_p = 1, j, h[j]
    return piv


def shark_orders(df: pd.DataFrame, k: float, strict: bool) -> list[tuple]:
    """回傳掛單：(確認索引, 方向, 掛單價, 停損價, 掛單截止索引)。"""
    piv = zigzag(df, k)
    atr = df["atr"].to_numpy()
    out = []
    for n in range(2, len(piv)):
        (ai, ap, ad, _), (bi, bp, bd, _), (ci, cp, cd, cconf) = piv[n - 2], piv[n - 1], piv[n]
        ext = abs(bp - cp) / abs(bp - ap)
        if (strict and not 1.618 <= ext <= 2.24) or (not strict and ext < 1.13):
            continue
        d = 1 if cd > 0 else -1          # C 是高點 → 回落到 0.886 做多；C 是低點 → 反彈到 0.886 做空
        bc = cp - bp
        limit = cp - 0.886 * bc
        stop = cp - 1.13 * bc - d * 0.1 * atr[cconf]
        end = min(piv[n + 1][3] if n + 1 < len(piv) else len(df), cconf + 100)
        out.append((cconf, d, limit, stop, end))
    return out


# ---------- 模擬 ----------

def fill_limit(df, start, end, d, limit) -> int:
    """從 start 起找第一根碰到掛單價的 K 棒；end 前沒有就回傳 -1。"""
    h, l = df["high"].to_numpy(), df["low"].to_numpy()
    for j in range(start, min(end, len(df))):
        if (d < 0 and h[j] >= limit) or (d > 0 and l[j] <= limit):
            return j
    return -1


def run(df: pd.DataFrame, orders: list[tuple], tf: str) -> pd.DataFrame:
    """orders: (成交後從哪根開始模擬, 方向, 進場價, 停損價, 成交那根已碰到停損, 訊號索引)。"""
    if not orders:
        return pd.DataFrame()
    op, hi, lo, cl = (df[k].to_numpy() for k in ("open", "high", "low", "close"))
    atr = df["atr"].to_numpy()
    starts = np.array([x[0] for x in orders], np.int64)
    ds = np.array([x[1] for x in orders], np.int64)
    entry = np.array([x[2] for x in orders], float)
    stop = np.array([x[3] for x in orders], float)
    risk = np.abs(entry - stop)
    isc = np.ones(len(hi), np.bool_)
    zero = np.zeros(len(orders))
    res = {"start": starts, "sig": np.array([x[5] for x in orders], np.int64), "side": ds, "entry": entry, "risk_pct": risk / entry,
           "time": df["time"].to_numpy()[np.minimum(starts, len(df) - 1)]}
    for name, (tp_r, portions, trail_k) in EXITS.items():
        tps = np.stack([entry + ds * kk * risk for kk in tp_r], axis=1)
        trails = trail_k * atr[np.minimum(starts, len(df) - 1) - 1]
        r, hit, j, kind, _ = sim_many(op, hi, lo, cl, isc, starts, ds, entry, risk, stop, tps, np.array(portions),
                                      0, trails, FEE, 0.0, MAX_BARS[tf], 0, 0, zero, 0.0, 0.0, zero, 0, cl, cl,
                                      False)
        # 成交那根已經碰到停損（掛單才會有）：算停損
        same = np.array([x[4] for x in orders], bool)
        r = np.where(same, (-risk - FEE * entry - FEE * stop) / risk, r)
        kind = np.where(same, 1, kind)
        j = np.where(same, starts - 1, j)
        res[f"r_{name}"], res[f"kind_{name}"], res[f"j_{name}"] = r, kind, j
    out = pd.DataFrame(res)
    return out[out["start"] < len(df)]


def build_orders(df, sigs, entry_mode: str, stop_mode: str, wait: int = 3) -> list[tuple]:
    o, c, h, l, atr = (df[k].to_numpy() for k in ("open", "close", "high", "low", "atr"))
    out = []
    for i, d in sigs:
        if np.isnan(atr[i]) or i + 1 >= len(df):
            continue
        if entry_mode == "mkt":
            px, j0, same = c[i], i + 1, False
        else:                                       # 掛在吞沒 K 實體一半
            px = (o[i] + c[i]) / 2
            j = fill_limit(df, i + 1, i + 1 + wait, d, px)
            if j < 0:
                continue
            j0 = j
        if stop_mode == "struct":
            ext = h[i] if d < 0 else l[i]
            sp = ext - d * 0.1 * atr[i]
            if d * (px - sp) < 0.5 * atr[i]:
                sp = px - d * 0.5 * atr[i]
        else:
            sp = px - d * 1.5 * atr[i]
        if entry_mode == "mkt":
            out.append((j0, d, px, sp, False, i))
        else:
            same = (h[j0] >= sp) if d < 0 else (l[j0] <= sp)
            out.append((j0 + 1, d, px, sp, bool(same), i))
    return out


def _coin(args):
    coin, tf = args
    df = frame(coin, tf)
    if df is None:
        return []
    rng = np.random.default_rng(zlib.crc32(f"{coin}{tf}".encode()))
    parts = []
    t = df["time"]

    def add(name, orders):
        r = run(df, orders, tf)
        if len(r):
            r["coin"], r["strategy"] = coin, name
            # 濾網用訊號那根收盤時已知的日線（掛單成交那根之後的資料不能用）
            sig = r["sig"].to_numpy()
            r["d_close"] = df["d_close"].to_numpy()[sig]
            r["d_ema200"] = df["d_ema200"].to_numpy()[sig]
            parts.append(r)

    if tf in ("1D", "4H"):
        for cross in (True, False):
            sigs = engulf_signals(df, cross)
            tag = "cross5" if cross else "below"
            for em in ("mkt", "half"):
                for sm in ("struct", "atr"):
                    add(f"engulf_{tf}_{tag}_{em}_{sm}", build_orders(df, sigs, em, sm))
        for cross in (False, True):
            sigs = regime_random(df, rng, cross)
            for sm in ("struct", "atr"):
                add(f"random_{tf}_{'cross5' if cross else 'regime'}_mkt_{sm}", build_orders(df, sigs, "mkt", sm))
    if tf == "1D":  # 敏感度：剛交叉的窗口、掛單等幾根
        for window in (3, 10):
            add(f"sens_1D_cross{window}_half_atr", build_orders(df, engulf_signals(df, True, window), "half", "atr"))
        for wait in (1, 5):
            add(f"sens_1D_cross5_half{wait}_atr", build_orders(df, engulf_signals(df, True), "half", "atr", wait))
    if tf in ("4H", "1H"):
        for k in (2.0, 3.0, 5.0):
            for strict in (False, True):
                orders = []
                for conf, d, limit, stop, end in shark_orders(df, k, strict):
                    c0 = df["close"].iat[conf]
                    if d * (c0 - limit) <= 0:   # 確認時已經越過掛單價：不追
                        continue
                    j = fill_limit(df, conf + 1, end, d, limit)
                    if j < 0:
                        continue
                    same = (df["high"].iat[j] >= stop) if d < 0 else (df["low"].iat[j] <= stop)
                    orders.append((j + 1, d, limit, stop, bool(same), conf))
                add(f"shark_{tf}_k{k:g}_{'strict' if strict else 'loose'}", orders)
    return parts


def stats(g: pd.DataFrame, ex: str) -> dict:
    r = g[f"r_{ex}"].to_numpy()
    key = (pd.to_datetime(g["time"]).dt.floor("D").astype(str) + g["side"].astype(str)).to_numpy()
    mean, se, cmean, cse, ncl = cluster_stats(r, key)
    return {"trades": len(r), "win": (r > 0).mean() if len(r) else np.nan, "exp_r": mean, "se_cl": se,
            "t_cl": mean / se if se else np.nan, "cluster_r": cmean, "clusters": ncl,
            "med_stop_pct": g["risk_pct"].median() * 100 if len(r) else np.nan}


def signals_main() -> None:
    jobs = [(c, tf) for tf in ("1D", "4H", "1H") for c in TOP100]
    with ProcessPoolExecutor() as ex:
        parts = [p for res in ex.map(_coin, jobs, chunksize=4) for p in res]
    allr = pd.concat(parts, ignore_index=True)
    allr["time"] = pd.to_datetime(allr["time"], utc=True)
    OUT.mkdir(exist_ok=True)
    rows = []
    for name, g in allr.groupby("strategy"):
        g = g.sort_values(["coin", "start"])
        for ex_name in EXITS:
            coin_id = pd.factorize(g["coin"])[0]
            take = busy_filter(coin_id, g["start"].to_numpy(np.int64) - 1, g[f"j_{ex_name}"].to_numpy(np.int64),
                               g[f"kind_{ex_name}"].to_numpy(np.int64))
            gg = g[take]
            filt = {"all": np.ones(len(gg), bool),
                    "ema200": (gg["side"] * (gg["d_close"] - gg["d_ema200"]) > 0).to_numpy()}
            for fn, fm in filt.items():
                for per, pm in (("全期", np.ones(len(gg), bool)), ("2020-23", (gg["time"] < SPLIT).to_numpy()),
                                ("2024+", (gg["time"] >= SPLIT).to_numpy())):
                    for side, sm in (("both", np.ones(len(gg), bool)), ("long", (gg["side"] > 0).to_numpy()),
                                     ("short", (gg["side"] < 0).to_numpy())):
                        sub = gg[fm & pm & sm]
                        rows.append({"strategy": name, "exit": ex_name, "filter": fn, "period": per, "side": side,
                                     **stats(sub, ex_name)})
                    # 每天同方向最多 3 筆（照時間先後）
                    sub = gg[fm & pm].sort_values("time")
                    day = sub["time"].dt.floor("D").astype(str) + sub["side"].astype(str)
                    capped = sub[sub.groupby(day).cumcount() < 3]
                    rows.append({"strategy": name, "exit": ex_name, "filter": fn, "period": per, "side": "cap3",
                                 **stats(capped, ex_name)})
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "threads_signals.csv", index=False)
    allr.to_pickle(OUT / "threads_trades.pkl")
    show = out[(out["side"] == "both") & (out["filter"] == "all")]
    with pd.option_context("display.width", 250, "display.max_rows", 500):
        print(show.pivot_table(index=["strategy", "exit"], columns="period", values=["exp_r", "t_cl", "win", "trades"])
              .round(3).to_string())


# ---------- AI 挑幣機器人 ----------

def daily_panel() -> dict[str, pd.DataFrame]:
    p = {}
    for c in TOP100:
        df = load(c, "1h")
        if len(df) < 24 * 60:
            continue
        d = resample(df, "1D", origin="epoch").set_index("time")
        d["qv"] = d["volume"] * d["close"]
        p[c] = d
    return p


def bot_arrays(panel) -> dict:
    close = pd.DataFrame({c: d["close"] for c, d in panel.items()})
    qv = pd.DataFrame({c: d["qv"] for c, d in panel.items()}).reindex(close.index)

    def get(k):
        return pd.DataFrame({c: d[k] for c, d in panel.items()}).reindex(close.index).to_numpy()

    return {"days": close.index, "open": get("open"), "low": get("low"),
            "vol30": qv.rolling(30, min_periods=30).mean().to_numpy(),
            "r14": (close / close.shift(14) - 1).to_numpy(), "r_a": (close.shift(7) / close.shift(14) - 1).to_numpy(),
            "r_b": (close / close.shift(7) - 1).to_numpy(),
            "btc24": (close["BTC"] / close["BTC"].shift(1) - 1).to_numpy()}


def bot(A: dict, hold: int | None, pick: str, rng=None, top_n=30, stop=0.20, btc_drop=0.05, cool=2) -> pd.DataFrame:
    """每天一步：t 這天收盤後決定，t+1 開盤成交；持有中的部位用 t+1 的日線低點檢查 -20% 止損。
    pick = best（14 天最強、前後 7 天都漲）/ worst（14 天最弱）/ random。回傳每筆交易。"""
    days, opn, low = A["days"], A["open"], A["low"]
    vol30, r14, r_a, r_b, btc24 = A["vol30"], A["r14"], A["r_a"], A["r_b"], A["btc24"]
    trades = []
    pos = None   # (幣的欄位, 進場價, 進場日索引, 上次換倉的日索引)
    cooldown = np.full(opn.shape[1], -1)
    for t in range(15, len(days) - 1):
        nxt = t + 1
        if pos is not None:
            c, e, t0, _ = pos
            lvl = e * (1 - stop)
            if not low[nxt, c] > lvl:   # 碰到止損（或資料斷了）
                px = min(opn[nxt, c], lvl) if not np.isnan(opn[nxt, c]) else lvl
                trades.append((c, days[t0], days[nxt], px / e - 1 - 2 * FEE, "stop"))
                cooldown[c] = nxt + cool
                pos = None
                continue
        market_dump = btc24[t] <= -btc_drop
        if pos is not None and market_dump:
            c, e, t0, _ = pos
            trades.append((c, days[t0], days[nxt], opn[nxt, c] / e - 1 - 2 * FEE, "dump"))
            pos = None
        if market_dump or (pos is not None and (hold is None or t - pos[3] + 1 < hold)):
            continue
        v = np.nan_to_num(vol30[t], nan=-1)
        top = np.argsort(-v)[:top_n]
        top = top[v[top] > 0]
        cand = top[(cooldown[top] <= nxt) & ~np.isnan(opn[nxt, top]) & ~np.isnan(r14[t, top])]
        if pick == "best":
            cand = cand[(r_a[t, cand] > 0) & (r_b[t, cand] > 0)]
        if not len(cand):
            if pos is not None:   # 到期但沒有合格的新標的：先抱著
                pos = pos[:3] + (nxt,)
            continue
        if pick == "best":
            c = cand[np.argmax(r14[t, cand])]
        elif pick == "worst":
            c = cand[np.argmin(r14[t, cand])]
        else:
            c = cand[rng.integers(len(cand))]
        if pos is not None:
            if pos[0] == c:       # 挑到同一個幣就續抱，不重複付手續費
                pos = pos[:3] + (nxt,)
                continue
            c0, e, t0, _ = pos
            trades.append((c0, days[t0], days[nxt], opn[nxt, c0] / e - 1 - 2 * FEE, "rotate"))
        pos = (c, opn[nxt, c], nxt, nxt)
    return pd.DataFrame(trades, columns=["coin", "entry_day", "exit_day", "ret", "why"])


def equity_stats(tr: pd.DataFrame, start, end) -> dict:
    if not len(tr):
        return {"trades": 0}
    tr = tr[(tr["entry_day"] >= start) & (tr["entry_day"] < end)]
    eq = np.cumprod(1 + tr["ret"].to_numpy())
    dd = (eq / np.maximum.accumulate(eq) - 1).min() if len(eq) else 0
    years = (end - start) / pd.Timedelta(days=365.25)
    return {"trades": len(tr), "win": (tr["ret"] > 0).mean(), "mean_ret": tr["ret"].mean(),
            "mean_R(20%)": tr["ret"].mean() / 0.20, "stops": (tr["why"] == "stop").mean(),
            "total": eq[-1] - 1 if len(eq) else 0, "cagr": (eq[-1] ** (1 / years) - 1) if len(eq) else 0,
            "max_dd": dd}


def bot_main() -> None:
    panel = daily_panel()
    A = bot_arrays(panel)
    idx = panel["BTC"].index
    pers = {"全期": (pd.Timestamp("2020-03-01", tz="UTC"), idx[-1]), "2020-23": (pd.Timestamp("2020-03-01", tz="UTC"), SPLIT),
            "2024+": (SPLIT, idx[-1])}
    rows = []
    for hold in (1, 3, 7, None):
        hn = "抱到止損" if hold is None else f"{hold}天換"
        tr = bot(A, hold, "best")
        tw = bot(A, hold, "worst")
        rng = np.random.default_rng(1)
        rand = [bot(A, hold, "random", rng) for _ in range(200)]
        for per, (a, b) in pers.items():
            s = equity_stats(tr, a, b)
            rs = pd.DataFrame([equity_stats(x, a, b) for x in rand])
            rows.append({"hold": hn, "period": per, "pick": "最強", **s,
                         "rand_cagr_median": rs["cagr"].median(), "rand_mean_ret_median": rs["mean_ret"].median(),
                         "pct_rand_beaten_cagr": (rs["cagr"] < s.get("cagr", 0)).mean()})
            rows.append({"hold": hn, "period": per, "pick": "最弱", **equity_stats(tw, a, b)})
    btc = panel["BTC"]["close"]
    for per, (a, b) in pers.items():
        s = btc[(btc.index >= a) & (btc.index < b)]
        years = (s.index[-1] - s.index[0]) / pd.Timedelta(days=365.25)
        rows.append({"hold": "BTC 抱著", "period": per, "pick": "-", "total": s.iat[-1] / s.iat[0] - 1,
                     "cagr": (s.iat[-1] / s.iat[0]) ** (1 / years) - 1, "max_dd": (s / s.cummax() - 1).min()})
    out = pd.DataFrame(rows)
    OUT.mkdir(exist_ok=True)
    out.to_csv(OUT / "threads_bot.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(out.round(3).to_string())


if __name__ == "__main__":
    {"signals": signals_main, "bot": bot_main}[sys.argv[1]]()
