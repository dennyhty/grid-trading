"""BTC 長期定投（DCA）：在相對低點才買，和「每天固定買」比較。

資金模型（所有策略同樣的總資金）：
- 每天都有一筆錢進帳。比較基準有兩個：
  - usdt：每天固定 1 USDT 進帳（= 每天買固定金額的比較基準）。
  - btc：每天進帳「買固定數量 BTC 需要的錢」（= 每天買固定數量的比較基準）。
- 比較基準把當天進帳全部買掉；策略把錢存著，等訊號才買。沒買的錢算現金，不計利息。
- 訊號用前一天收盤算，隔天開盤價買（不偷看），手續費 0.05%。
- deadline=N：一筆錢存超過 N 天還沒等到訊號，就照樣買掉（先進先出），避免錢一直閒置。

用法：python dca_backtest.py  →  輸出 output/dca/*.csv 與終端機摘要
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from grid_trading.indicators import ema, macd, rsi
from grid_trading.spot_history import load_btc_long

FEE = 0.0005
GENESIS = pd.Timestamp("2009-01-03")


def weekly_rsi(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """週 RSI，只用已收完的週 K（週一到週日，UTC）；週日收盤後才更新。"""
    wk = df["close"].resample("W-SUN").last()
    r = rsi(wk, n)
    return r.reindex(df.index, method="ffill")


def features(df: pd.DataFrame) -> pd.DataFrame:
    c = df["close"]
    f = pd.DataFrame(index=df.index)
    f["close"] = c
    f["rsi"] = rsi(c, 14)
    f["wrsi"] = weekly_rsi(df)
    f["sma200"] = c.rolling(200).mean()
    f["mayer"] = c / f["sma200"]
    f["ema169"] = ema(c, 169)
    f["ema144"] = ema(c, 144)
    m = macd(c)
    f["dif"] = m["macd"]
    f["hist"] = m["hist"]
    f["dd"] = c / c.cummax() - 1
    # ahr999：華語圈常用的 BTC 定投指標（< 0.45 抄底、0.45~1.2 定投）
    gma200 = np.exp(np.log(c).rolling(200).mean())
    age = (df.index - GENESIS).days
    est = 10 ** (5.84 * np.log10(age) - 17.01)
    f["ahr999"] = (c / gma200) * (c / est)
    # 同一個指標，但冪次成長估值用「當時為止」的資料每月重新回歸（不偷看未來）
    f["ahr999_wf"] = (c / gma200) * (c / walk_forward_power_law(c))
    return f


def walk_forward_power_law(close: pd.Series, min_days: int = 730) -> pd.Series:
    """log10(價格) = a·log10(幣齡天數) + b，每個月底用到當天為止的全部資料重新回歸，套用到下個月。"""
    age = np.log10((close.index - GENESIS).days.to_numpy(dtype=float))
    lp = np.log10(close.to_numpy())
    est = np.full(len(close), np.nan)
    month_ends = close.index.to_series().groupby(close.index.to_period("M")).last()
    for m_end in month_ends:
        n = close.index.get_loc(m_end) + 1
        if n < min_days:
            continue
        a, b = np.polyfit(age[:n], lp[:n], 1)
        nxt = slice(n, min(n + 32, len(close)))
        est[nxt] = 10 ** (a * age[nxt] + b)
    est = pd.Series(est, index=close.index)
    # 每個月只覆蓋到下個月底前；重疊部分以較新的回歸為準（迴圈順序已經保證）
    return est


@dataclass
class Rule:
    name: str
    cond: str | None = None  # features 欄位組成的布林條件（Python 運算式）
    mode: str = "all"  # all：條件成立把存款全買；tranche：買 k 天的進帳；rate：每天買存款的 1/k；scaled：每天買 進帳×倍數
    k: float = 1.0
    base: float = 0.0  # 不管訊號，每天先買進帳的這個比例
    deadline: int | None = None
    mult: str | None = None  # scaled 用的倍數（Python 運算式）
    group: str = ""
    notes: str = field(default="", repr=False)


def simulate(f: pd.DataFrame, opens: np.ndarray, closes: np.ndarray, inflow: np.ndarray,
             rule: Rule | None, cond: np.ndarray | None, mult: np.ndarray | None) -> dict:
    """在 [0, n) 天模擬；cond / mult 是「前一天收盤」的訊號（已經位移好）。"""
    n = len(opens)
    cash = 0.0
    btc = 0.0
    spent = 0.0
    lots: deque = deque()
    buy_days = 0
    for i in range(n):
        x = inflow[i]
        cash += x
        lots.append([i, x])
        if rule is None:  # 比較基準：進帳全買
            want = cash
        else:
            want = rule.base * x
            if rule.mode == "scaled":
                want = max(want, mult[i] * x)
            elif cond[i]:
                if rule.mode == "all":
                    want = cash
                elif rule.mode == "tranche":
                    want = max(want, rule.k * x)
                elif rule.mode == "rate":
                    want = max(want, cash / rule.k, x)
            if rule.deadline is not None:
                forced = 0.0
                for d, a in lots:
                    if d > i - rule.deadline:
                        break
                    forced += a
                want = max(want, forced)
        want = min(want, cash)
        if want > 1e-12:
            btc += want * (1 - FEE) / opens[i]
            spent += want
            cash -= want
            buy_days += 1
            left = want
            while lots and left > 1e-12:
                if lots[0][1] <= left + 1e-12:
                    left -= lots[0][1]
                    lots.popleft()
                else:
                    lots[0][1] -= left
                    left = 0.0
    total_in = float(inflow.sum())
    value = btc * closes[-1] + cash
    return {
        "btc": btc,
        "spent": spent,
        "cash_left": cash,
        "deployed": spent / total_in,
        "avg_cost": spent / btc if btc > 0 else np.nan,
        "value": value,
        "total_in": total_in,
        "buy_days": buy_days,
    }


def rules() -> list[Rule]:
    out: list[Rule] = []
    # 1. 日 RSI 超賣
    for th in (25, 30, 35, 40):
        out.append(Rule(f"日RSI<{th} 存款全買", f"rsi < {th}", "all", group="RSI"))
    for th in (30, 35):
        for k in (5, 10, 30):
            out.append(Rule(f"日RSI<{th} 每次買{k}天份", f"rsi < {th}", "tranche", k, group="RSI"))
    for th in (30, 35):
        out.append(Rule(f"日RSI<{th} 存款全買 + 最多等90天", f"rsi < {th}", "all", deadline=90, group="RSI"))
        out.append(Rule(f"日RSI<{th} 存款全買 + 最多等180天", f"rsi < {th}", "all", deadline=180, group="RSI"))
        out.append(Rule(f"每天買一半 + 日RSI<{th} 存款全買", f"rsi < {th}", "all", base=0.5, group="RSI"))
    # 2. 週 RSI
    for th in (40, 45, 50):
        out.append(Rule(f"週RSI<{th} 每天買存款1/30", f"wrsi < {th}", "rate", 30, group="週RSI"))
        out.append(Rule(f"週RSI<{th} 存款全買", f"wrsi < {th}", "all", group="週RSI"))
    # 3. 均線
    out.append(Rule("收盤<200日均線 每天買存款1/30", "close < sma200", "rate", 30, group="均線"))
    out.append(Rule("收盤<200日均線 存款全買", "close < sma200", "all", group="均線"))
    out.append(Rule("收盤<200日均線 存款全買 + 最多等180天", "close < sma200", "all", deadline=180, group="均線"))
    out.append(Rule("收盤<200日均線×0.8 存款全買", "mayer < 0.8", "all", group="均線"))
    for k in (1, 2, 3):
        out.append(Rule(f"梅耶倍數縮放 (1/梅耶)^{k}", None, "scaled", mult=f"(1/mayer)**{k}", group="均線"))
    # 4. 維加斯通道（日線 EMA144/169）
    out.append(Rule("收盤<EMA169 每天買存款1/30", "close < ema169", "rate", 30, group="維加斯"))
    out.append(Rule("收盤<EMA169 存款全買", "close < ema169", "all", group="維加斯"))
    out.append(Rule("收盤<EMA169 存款全買 + 最多等180天", "close < ema169", "all", deadline=180, group="維加斯"))
    # 5. MACD 零軸下金叉
    out.append(Rule("MACD 零軸下金叉 存款全買", "(dif < 0) & (hist > 0) & (hist_prev <= 0)", "all", group="MACD"))
    out.append(Rule("MACD 零軸下柱狀體翻正 + 最多等180天", "(dif < 0) & (hist > 0) & (hist_prev <= 0)", "all",
                    deadline=180, group="MACD"))
    # 6. 從歷史高點回落
    for dd in (30, 50):
        out.append(Rule(f"距最高點跌>{dd}% 每天買存款1/30", f"dd < -{dd / 100}", "rate", 30, group="回撤"))
    # 7. ahr999
    out.append(Rule("ahr999<0.45 存款全買", "ahr999 < 0.45", "all", group="ahr999"))
    out.append(Rule("ahr999<1.2 每天買存款1/30", "ahr999 < 1.2", "rate", 30, group="ahr999"))
    out.append(Rule("ahr999<1.2 存款全買", "ahr999 < 1.2", "all", group="ahr999"))
    out.append(Rule("ahr999 縮放：<0.45 買3倍、<1.2 買1倍、其他不買", None, "scaled",
                    mult="where(ahr999 < 0.45, 3.0, where(ahr999 < 1.2, 1.0, 0.0))", group="ahr999"))
    # 門檻敏感度
    for lo in (0.35, 0.55, 0.7):
        out.append(Rule(f"ahr999<{lo} 存款全買", f"ahr999 < {lo}", "all", group="ahr999"))
    for hi in (0.9, 1.5, 2.0):
        out.append(Rule(f"ahr999<{hi} 存款全買", f"ahr999 < {hi}", "all", group="ahr999"))
    # 不偷看版本（估值線每月重新回歸）
    out.append(Rule("ahr999(滾動回歸)<0.45 存款全買", "ahr999_wf < 0.45", "all", group="ahr999滾動"))
    out.append(Rule("ahr999(滾動回歸)<1.2 存款全買", "ahr999_wf < 1.2", "all", group="ahr999滾動"))
    out.append(Rule("ahr999(滾動回歸) 縮放：<0.45 買3倍、<1.2 買1倍、其他不買", None, "scaled",
                    mult="where(ahr999_wf < 0.45, 3.0, where(ahr999_wf < 1.2, 1.0, 0.0))", group="ahr999滾動"))
    # 8. 過熱時暫停、平常照買（ahr999<1.2 的反向想法，但不用擬合的估值線）
    for th in (1.5, 2.0, 2.4):
        out.append(Rule(f"梅耶倍數<{th} 存款全買（過熱暫停）", f"mayer < {th}", "all", group="過熱暫停"))
    for th in (70, 80):
        out.append(Rule(f"週RSI<{th} 存款全買（過熱暫停）", f"wrsi < {th}", "all", group="過熱暫停"))
    # 9. RSI 縮放（不用存太久）
    out.append(Rule("日RSI 縮放：<30 買3倍、<50 買1.5倍、其他買0.5倍", None, "scaled",
                    mult="where(rsi < 30, 3.0, where(rsi < 50, 1.5, 0.5))", group="RSI"))
    return out


def evaluate(f: pd.DataFrame, opens: pd.Series, start, end, kind: str, rule_list: list[Rule],
             cond_cache: dict) -> list[dict]:
    sl = slice(start, end)
    o = opens.loc[sl].to_numpy()
    c = f["close"].loc[sl].to_numpy()
    if kind == "usdt":
        inflow = np.ones(len(o))
    else:  # 每天固定買 0.0001 BTC 需要的錢（數量不影響比例）
        inflow = 1e-4 * o
    base = simulate(f, o, c, inflow, None, None, None)
    rows = []
    for r in rule_list:
        cond = cond_cache[r.name][0].loc[sl].to_numpy() if r.cond else None
        mult = cond_cache[r.name][1].loc[sl].to_numpy() if r.mult else None
        res = simulate(f, o, c, inflow, r, cond, mult)
        rows.append({
            "rule": r.name,
            "group": r.group,
            "start": pd.Timestamp(start).date(),
            "end": pd.Timestamp(end).date(),
            "benchmark": kind,
            "btc_vs_bench": res["btc"] / base["btc"] - 1,
            "value_vs_bench": res["value"] / base["value"] - 1,
            "avg_cost": res["avg_cost"],
            "bench_avg_cost": base["avg_cost"],
            "cost_vs_bench": res["avg_cost"] / base["avg_cost"] - 1,
            "deployed": res["deployed"],
            "cash_left_pct": res["cash_left"] / res["total_in"],
            "buy_days": res["buy_days"],
            "days": len(o),
            "value_x": res["value"] / res["total_in"],
            "bench_value_x": base["value"] / base["total_in"],
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="output/dca")
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    df = load_btc_long(refresh=args.refresh)
    f = features(df)
    f["hist_prev"] = f["hist"].shift(1)
    first = pd.Timestamp("2018-03-06")  # 幣安現貨 2017-08-17 開始，加上 200 天暖機
    early = f["ahr999_wf"].first_valid_index() + pd.Timedelta(days=1)
    last = f.index[-1]
    rl = rules()
    env = {col: f[col] for col in f.columns} | {"where": np.where}
    cache = {}
    for r in rl:
        # NaN（指標還沒算出來）一律不算成立
        cond = pd.Series(eval(r.cond, {}, env), index=f.index).fillna(False).astype(bool).shift(1, fill_value=False) if r.cond else None
        mult = pd.Series(eval(r.mult, {}, env), index=f.index).clip(0, 10).shift(1).fillna(1.0) if r.mult else None
        cache[r.name] = (cond, mult)

    periods = {
        "早期 CMC 2015-01~2018-03": (max(early, pd.Timestamp("2015-01-01")), first - pd.Timedelta(days=1)),
        "全期間": (first, last),
        "樣本內 2018-03~2021-12": (first, pd.Timestamp("2021-12-31")),
        "樣本外 2022-01~2026-10": (pd.Timestamp("2022-01-01"), last),
        "週期1 2018-03~2020-05 減半": (first, pd.Timestamp("2020-05-10")),
        "週期2 2020-05~2024-04 減半": (pd.Timestamp("2020-05-11"), pd.Timestamp("2024-04-19")),
        "週期3 2024-04~2026-10": (pd.Timestamp("2024-04-20"), last),
    }
    rows = []
    for label, (s, e) in periods.items():
        for kind in ("usdt", "btc"):
            for row in evaluate(f, df["open"], s, e, kind, rl, cache):
                row["period"] = label
                rows.append(row)
    fixed = pd.DataFrame(rows)
    fixed.to_csv(out / "dca_periods.csv", index=False)

    # 滾動視窗：每月 1 日起算，2 年與 4 年。binance：2018-03 起（主要）；long：2015-01 起，前段是 CoinMarketCap
    roll = []
    for source, begin in (("binance", first), ("long", max(early, pd.Timestamp("2015-01-01")))):
        for years in (2, 4):
            starts = pd.date_range(begin, last - pd.DateOffset(years=years), freq="MS")
            for s in starts:
                e = s + pd.DateOffset(years=years) - pd.Timedelta(days=1)
                for row in evaluate(f, df["open"], s, e, "usdt", rl, cache):
                    row["source"] = source
                    row["years"] = years
                    roll.append(row)
    roll = pd.DataFrame(roll)
    roll.to_csv(out / "dca_rolling.csv", index=False)

    summ = roll.groupby(["source", "years", "rule"]).agg(
        windows=("value_vs_bench", "size"),
        win_rate=("value_vs_bench", lambda x: (x > 0).mean()),
        median=("value_vs_bench", "median"),
        p10=("value_vs_bench", lambda x: x.quantile(0.1)),
        worst=("value_vs_bench", "min"),
        best=("value_vs_bench", "max"),
        cost_median=("cost_vs_bench", "median"),
        deployed=("deployed", "median"),
    ).reset_index()
    summ.to_csv(out / "dca_rolling_summary.csv", index=False)

    pd.set_option("display.width", 250, "display.max_rows", 500, "display.max_colwidth", 60)
    piv = fixed[fixed.benchmark == "usdt"].pivot(index="rule", columns="period", values="value_vs_bench")
    print((piv * 100).round(1))
    print(summ.sort_values(["source", "years", "median"], ascending=[True, True, False]).round(3))


if __name__ == "__main__":
    main()
