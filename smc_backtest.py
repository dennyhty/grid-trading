"""SMC / ICT 策略回測：python smc_backtest.py [--timeframes 1H,2H,4H] [--out output]

用幣安 USDT 永續 1H 月檔（2020-01 起）合成 1H / 2H / 4H，84 個派網有合約的前 100 大幣。
結構定義見 grid_trading/smc.py。每個組合同時算兩種成本：手續費每邊 0.05%，以及再加滑價每邊 0.05%。

- 大週期（HTF）趨勢：1H 看 4H、2H 看 8H、4H 看日線，用同一套擺動點 / 結構突破判斷，只用已收盤的大週期 K 線。
- 限價單：突破那根收盤後掛單，最多掛 expiry 根；成交前價格先跑到 2R 就取消；有新的進場計畫就換成新的。
- 成交那根如果也碰到停損，整筆算停損（保守，不算同一根裡的停利）。
- 出場和維加斯趨勢完全相同（tracker）：TP 1R/2R/3R 各平 1/3，TP1 後保本，TP2 後停損移到 TP1，40 天時間出場。
- 同一個幣同時只有一張單。

輸出 output/smc_summary.csv（每個組合、每段期間）和 output/smc_trades.csv.gz。
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from grid_trading import indicators as ind
from grid_trading import smc
from grid_trading.binance_history import load as binance_load
from grid_trading.history import resample
from grid_trading.signals.strategy import LONG, SHORT, Signal
from grid_trading.signals.tracker import track
from grid_trading.universe import TOP100

TF_HOURS = {"1H": 1, "2H": 2, "4H": 4}
HTF_HOURS = {"1H": 4, "2H": 8, "4H": 24}
FEES = {"fee": 0.0005, "fee+slip": 0.0010}
SPLIT = pd.Timestamp("2024-01-01", tz="UTC")
TP_R = (1.0, 2.0, 3.0)
SETUPS = ["sweep", "choch", "bos"]
ENTRIES = ["market", "fvg", "fvg_ce", "ob"]


def configs():
    for setup, entry, htf in itertools.product(SETUPS, ENTRIES, (True, False)):
        for disc in ((False,) if entry == "market" else (True, False)):
            yield {"setup": setup, "entry": entry, "htf": htf, "discount": disc}


def config_name(c: dict) -> str:
    return f"{c['setup']}|{c['entry']}|{'htf' if c['htf'] else 'nohtf'}|{'disc' if c['discount'] else 'any'}"


def tf_bars(h1: pd.DataFrame, hours: int) -> pd.DataFrame:
    """1H 合成 hours 小時 K 線，丟掉最後一根還沒湊滿的。"""
    if hours == 1:
        return h1.copy()
    out = resample(h1, f"{hours}h", "start_day")
    last_close = h1["time"].iloc[-1] + pd.Timedelta(hours=1)
    return out[out["time"] + pd.Timedelta(hours=hours) <= last_close].reset_index(drop=True)


def htf_trend_for(df: pd.DataFrame, htf: pd.DataFrame, tf_h: int, htf_h: int, k: int) -> np.ndarray:
    """每根小週期 K 線收盤時，最後一根已收盤的大週期 K 線的結構趨勢。"""
    trend, _ = smc.structure(htf, k)
    htf_close = (htf["time"] + pd.Timedelta(hours=htf_h)).to_numpy()
    ltf_close = (df["time"] + pd.Timedelta(hours=tf_h)).to_numpy()
    idx = np.searchsorted(htf_close, ltf_close, side="right") - 1
    out = np.where(idx >= 0, trend[np.clip(idx, 0, None)], 0)
    return out


def simulate(df: pd.DataFrame, setups: list[smc.Setup], symbol: str, tf_h: int, fee: float,
             expiry: int = 20, cancel_r: float = 2.0, max_days: int = 40) -> list[dict]:
    times = df["time"]
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    bar = pd.Timedelta(hours=tf_h)
    max_bars = max_days * 24 // tf_h
    busy_until = -1
    out = []
    for n, s in enumerate(setups):
        if s.bar <= busy_until:
            continue
        d = s.direction
        risk = d * (s.entry - s.sl)
        fill = None
        if s.market:
            fill = s.bar
        else:
            nxt = setups[n + 1].bar if n + 1 < len(setups) else len(df) - 1
            end = min(s.bar + expiry, nxt, len(df) - 1)
            cancel = s.entry + d * cancel_r * risk
            for t in range(s.bar + 1, end + 1):
                if (d > 0 and low[t] <= s.entry) or (d < 0 and high[t] >= s.entry):
                    fill = t
                    break
                if (d > 0 and high[t] >= cancel) or (d < 0 and low[t] <= cancel):
                    break
        if fill is None:
            continue
        rec = {"symbol": symbol, "side": LONG if d > 0 else SHORT, "time": times.iat[fill] + bar,
               "setup_time": times.iat[s.bar] + bar, "entry": s.entry, "sl": s.sl,
               "risk_pct": risk / s.entry * 100, "kind": s.kind}
        if not s.market and ((d > 0 and low[fill] <= s.sl) or (d < 0 and high[fill] >= s.sl)):
            cost = fee * (s.entry + s.sl)
            rec.update(r=-(risk + cost) / risk, pct=-(risk + cost) / s.entry * 100, tps_hit=0, exit="SL_FILLBAR",
                       exit_time=times.iat[fill], bars=0)
            busy_until = fill
        else:
            sig = Signal(symbol, f"{tf_h}H", rec["side"], rec["time"], s.entry, s.sl,
                         [s.entry + d * k * risk for k in TP_R], "smc")
            res = track(sig, df, fee, max_bars)
            if res.status != "CLOSED":
                continue
            ex = res.events[-1].time
            rec.update(r=res.realized_r, pct=res.realized_pct, tps_hit=res.tps_hit, exit=res.events[-1].kind,
                       exit_time=ex, bars=res.bars)
            busy_until = int(times.searchsorted(ex))
        out.append(rec)
    return out


def run(job):
    coin, tf, k = job
    h1 = binance_load(coin, "1h")
    if h1.empty:
        return []
    tf_h, htf_h = TF_HOURS[tf], HTF_HOURS[tf]
    df = tf_bars(h1, tf_h)
    if len(df) < 300:
        return []
    df["atr"] = ind.atr(df)
    htf = tf_bars(h1, htf_h)
    htf_tr = htf_trend_for(df, htf, tf_h, htf_h, k)
    _, breaks = smc.structure(df, k)
    rows = []
    for c in configs():
        setups = smc.find_setups(df, breaks, htf_tr, **c)
        name = config_name(c)
        for fee_name, fee in FEES.items():
            for t in simulate(df, setups, f"{coin}_USDT_PERP", tf_h, fee):
                t.update(tf=tf, k=k, config=name, cost=fee_name)
                rows.append(t)
    return rows


def random_trades(trades: pd.DataFrame, fee: float, reps: int = 40, seed: int = 0, direction: str = "copy",
                  k: int = 3) -> pd.DataFrame:
    """隨機進場基準：每筆真實交易換成同一個幣、同一季、同停損百分比的隨機一根 K 線收盤進場，出場規則相同。

    direction：copy = 方向照抄真實那筆（同一季的方向等於偷看了整季的走勢，基準會偏高）；
    htf = 用隨機那根當時的大週期結構趨勢決定方向（沒有偷看）；coin = 擲銅板。
    """
    rng = np.random.default_rng(seed)
    rows = []
    for (sym, tf), grp in trades.groupby(["symbol", "tf"]):
        tf_h = TF_HOURS[tf]
        h1 = binance_load(sym.split("_")[0], "1h")
        df = tf_bars(h1, tf_h)
        close = df["time"].dt.tz_convert(None)
        quarter = (close + pd.Timedelta(hours=tf_h)).dt.to_period("Q").astype(str).to_numpy()
        close = df["close"].to_numpy()
        htf = htf_trend_for(df, tf_bars(h1, HTF_HOURS[tf]), tf_h, HTF_HOURS[tf], k) if direction == "htf" else None
        by_q = {q: np.flatnonzero(quarter == q) for q in np.unique(quarter)}
        max_bars = 40 * 24 // tf_h
        for rep in range(reps):
            for t in grp.itertuples():
                cand = by_q.get(str(pd.Timestamp(t.time).tz_convert(None).to_period("Q")))
                if direction == "htf" and cand is not None:
                    cand = cand[htf[cand] != 0]
                if cand is None or len(cand) == 0:
                    continue
                i = int(rng.choice(cand))
                if i >= len(df) - 2:
                    continue
                if direction == "copy":
                    d = 1 if t.side == LONG else -1
                elif direction == "htf":
                    d = int(htf[i])
                else:
                    d = 1 if rng.random() < 0.5 else -1
                side = LONG if d > 0 else SHORT
                px = close[i]
                risk = px * t.risk_pct / 100
                sig = Signal(sym, tf, side, df["time"].iat[i] + pd.Timedelta(hours=tf_h), px, px - d * risk,
                             [px + d * m * risk for m in TP_R], "random")
                res = track(sig, df, fee, max_bars)
                if res.status == "CLOSED":
                    rows.append({"rep": rep, "symbol": sym, "tf": tf, "side": side, "time": sig.time,
                                 "exit_time": res.events[-1].time, "r": res.realized_r, "risk_pct": t.risk_pct})
    return pd.DataFrame(rows)


def stats(t: pd.DataFrame, risk: float = 0.0025) -> dict:
    if t.empty:
        return {"n": 0}
    t = t.sort_values("exit_time")
    r = t["r"]
    eq = (1 + risk * r).cumprod()
    years = max((t["exit_time"].max() - t["time"].min()).days / 365.25, 0.1)
    return {"n": len(r), "win": (r > 0).mean(), "exp_r": r.mean(), "se": r.std() / np.sqrt(len(r)),
            "cagr": eq.iloc[-1] ** (1 / years) - 1, "dd": (eq / eq.cummax() - 1).min(),
            "risk_pct": t["risk_pct"].median(), "per_day": len(r) / (years * 365.25)}


def summarize(trades: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (tf, k, cfg, cost), t in trades.groupby(["tf", "k", "config", "cost"]):
        row = {"tf": tf, "k": k, "config": cfg, "cost": cost}
        for label, part in (("all", t), ("sel", t[t["time"] < SPLIT]), ("val", t[t["time"] >= SPLIT])):
            for key, v in stats(part).items():
                row[f"{label}_{key}"] = v
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> int:
    p = argparse.ArgumentParser(description="SMC / ICT 策略回測")
    p.add_argument("--timeframes", default="1H,2H,4H")
    p.add_argument("--k", default="3,5")
    p.add_argument("--coins", default=None)
    p.add_argument("--out", default="output")
    args = p.parse_args()
    coins = args.coins.split(",") if args.coins else TOP100
    jobs = [(c, tf, int(k)) for tf in args.timeframes.split(",") for k in args.k.split(",") for c in coins]
    rows = []
    with ProcessPoolExecutor(os.cpu_count() or 2) as ex:
        for r in ex.map(run, jobs):
            rows += r
    trades = pd.DataFrame(rows)
    Path(args.out).mkdir(exist_ok=True)
    trades.to_csv(Path(args.out) / "smc_trades.csv.gz", index=False)
    summary = summarize(trades)
    summary.to_csv(Path(args.out) / "smc_summary.csv", index=False)
    view = summary[summary["cost"] == "fee"].sort_values("sel_exp_r", ascending=False)
    cols = ["tf", "k", "config", "all_n", "sel_n", "sel_exp_r", "val_n", "val_exp_r", "val_dd", "all_risk_pct"]
    with pd.option_context("display.width", 250, "display.max_rows", 400):
        print(view[cols].round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
