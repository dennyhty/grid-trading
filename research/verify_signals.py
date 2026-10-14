"""逐筆驗證群組訊號的實際結果：python -m research.verify_signals

用派網 15 分 K 線看訊號發出後先碰到停損還是停利，並計算跟單結果（以 R = 停損距離為單位）：
- 「TP1 全出」：碰到 TP1 全部平倉，看純勝率
- 「TP1 半倉 + 保本」：TP1 平一半、停損移到進場價，剩下一半在 TP2 出場
同一根 K 線同時碰到停損和停利時，保守當作先停損。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from grid_trading.breakout import BreakoutConfig, add_indicators, signal_at
from grid_trading.history import fetch_history
from grid_trading.signal_log import SIGNALS

TPE = "Asia/Taipei"


def locate(df: pd.DataFrame, when: str, entry: float) -> pd.Timestamp:
    if len(when) > 10:
        return pd.Timestamp(when, tz=TPE).tz_convert("UTC")
    # 只有日期：取當天收盤價最接近進場價的 K 線
    day = df[df["time"].dt.tz_convert(TPE).dt.strftime("%Y-%m-%d") == when]
    i = (day["close"] - entry).abs().idxmin()
    return day.loc[i, "time"] + pd.Timedelta(minutes=15)


def follow(after: pd.DataFrame, entry: float, tps: list[float], sl: float) -> dict:
    risk = entry - sl
    res = {"first": "未定", "tp_hit": 0, "sl_hit": False, "mfe": after["high"].max() / entry - 1,
           "mae": after["low"].min() / entry - 1}
    tp_times = {}
    for _, b in after.iterrows():
        if b["low"] <= sl:
            res["sl_hit"] = True
            res.setdefault("sl_time", b["time"])
            if res["first"] == "未定":
                res["first"] = "停損"
            break
        for k, tp in enumerate(tps[:3]):
            if k not in tp_times and b["high"] >= tp:
                tp_times[k] = b["time"]
                if res["first"] == "未定" and k == 0:
                    res["first"] = "TP1"
    res["tp_hit"] = len(tp_times)
    res["tp1_time"] = tp_times.get(0)
    last = after["close"].iloc[-1]
    # TP1 全出
    if res["first"] == "TP1":
        res["r_tp1"] = (tps[0] - entry) / risk
    elif res["first"] == "停損":
        res["r_tp1"] = -1.0
    else:
        res["r_tp1"] = (last - entry) / risk
    # TP1 半倉 + 保本
    if not tps:
        res["r_half"] = res["r_tp1"]
    elif res["first"] == "停損":
        res["r_half"] = -1.0
    elif res["first"] == "TP1":
        part1 = 0.5 * (tps[0] - entry)
        if len(tps) > 1 and 1 in tp_times:
            # TP2 前有沒有回到保本價
            seg = after[(after["time"] > tp_times[0]) & (after["time"] < tp_times[1])]
            part2 = 0.0 if (seg["low"] <= entry).any() else 0.5 * (tps[1] - entry)
        else:
            seg = after[after["time"] > tp_times[0]]
            part2 = 0.0 if (seg["low"] <= entry).any() else 0.5 * (last - entry)
        res["r_half"] = (part1 + part2) / risk
    else:
        res["r_half"] = (last - entry) / risk
    return res


def main() -> int:
    rows = []
    for coin, sym, when, entry, tps, sl in SIGNALS:
        row = {"coin": coin, "time_tpe": when, "entry": entry, "sl": sl,
               "sl_pct": 100 * (sl / entry - 1),
               "tp1_r": round((tps[0] - entry) / (entry - sl), 2) if tps else None,
               "tp2_r": round((tps[1] - entry) / (entry - sl), 2) if len(tps) > 1 else None}
        if sym is None:
            rows.append({**row, "note": "派網沒有這個幣，無法驗證"})
            continue
        df = fetch_history(sym, "15M", 30, refresh=True)
        t = locate(df, when, entry)
        bar = df[df["time"] <= t - pd.Timedelta(minutes=15)].iloc[-1]
        after = df[df["time"] >= t.floor("15min")]
        row["price_gap_pct"] = 100 * (bar["close"] / entry - 1)
        row.update(follow(after, entry, tps, sl))
        row["days_tracked"] = (after["time"].iloc[-1] - after["time"].iloc[0]).total_seconds() / 86400
        # 推出的突破規則在訊號前最後兩根 1H 是否觸發
        h1 = add_indicators(fetch_history(sym, "60M", 120, refresh=True), BreakoutConfig())
        done = h1[h1["bar_close"] <= t].tail(2)
        r = done.iloc[-1]
        row["rule_fires"] = any(signal_at(x, BreakoutConfig()) is not None for _, x in done.iterrows())
        row["breakout"] = bool(r["close"] > r["prior_high"])
        row["above_vegas_1h"] = bool(r["close"] > r["ema144"] and r["close"] > r["ema169"])
        row["rsi_1h"] = round(r["rsi"], 1)
        row["h4_trend"] = bool(r["h4_trend"])
        row["h4_macd_up"] = bool(r["h4_macd"])
        rows.append(row)
    out = pd.DataFrame(rows)
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "output") / "signal_verification.csv"
    path.parent.mkdir(exist_ok=True)
    out.to_csv(path, index=False)
    cols = ["coin", "time_tpe", "sl_pct", "tp1_r", "tp2_r", "price_gap_pct", "first", "tp_hit", "mfe", "mae",
            "r_tp1", "r_half", "rule_fires", "breakout", "above_vegas_1h", "rsi_1h", "h4_trend", "h4_macd_up"]
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(out[[c for c in cols if c in out]].round(3).to_string(index=False))
    v = out.dropna(subset=["first"])
    print(f"\n可驗證 {len(v)} 筆；先到 TP1 {(v['first'] == 'TP1').sum()}、先停損 {(v['first'] == '停損').sum()}、未定 {(v['first'] == '未定').sum()}")
    print(f"TP1 全出 平均 {v['r_tp1'].mean():.2f}R 合計 {v['r_tp1'].sum():.1f}R；半倉+保本 平均 {v['r_half'].mean():.2f}R 合計 {v['r_half'].sum():.1f}R")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
