"""倉位大小：每筆風險 = 帳戶權益 × risk（預設 0.25%），數量 = 風險金額 ÷ 止損距離。

幣安每個合約有最小數量、數量單位（stepSize）、價格單位（tickSize）和最小名目金額（多數 5U、BTC 約 100U）。
算出來太小時改用最小可下的量；這樣實際風險會變大，超過 max_risk（預設 1%）就不做。
數量至少要 4 個單位，TP1~TP3 和最後追蹤的 1/4 才分得開。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal


@dataclass(frozen=True)
class Filters:
    step: float          # 市價單數量單位
    min_qty: float
    max_qty: float
    tick: float          # 價格單位
    min_notional: float  # 開倉的最小名目金額（只減倉的單不受限制）


def parse_filters(info: dict) -> dict[str, Filters]:
    """exchangeInfo → {代號: Filters}，只留正在交易的 USDT 永續合約。"""
    out = {}
    for s in info.get("symbols", []):
        if s.get("status") != "TRADING" or s.get("contractType") != "PERPETUAL" or s.get("quoteAsset") != "USDT":
            continue
        f = {x["filterType"]: x for x in s.get("filters", [])}
        lot = f.get("MARKET_LOT_SIZE") or f.get("LOT_SIZE") or {}
        base_lot = f.get("LOT_SIZE") or lot
        out[s["symbol"]] = Filters(
            step=max(float(lot.get("stepSize") or 0), float(base_lot.get("stepSize") or 0)),
            min_qty=max(float(lot.get("minQty") or 0), float(base_lot.get("minQty") or 0)),
            max_qty=float(lot.get("maxQty") or 1e18),
            tick=float((f.get("PRICE_FILTER") or {}).get("tickSize") or 0),
            min_notional=float((f.get("MIN_NOTIONAL") or {}).get("notional") or 0))
    return out


@dataclass
class Size:
    steps: int            # 總數量 = steps × step
    slices: list[int]     # TP1、TP2、TP3、最後追蹤的部分，各幾個 step
    risk_frac: float      # 實際風險佔權益的比例
    bumped: bool          # 是否因為最小下單量而放大


def plan_size(equity: float, risk: float, price: float, stop: float, f: Filters,
              portions=(0.25, 0.25, 0.25), max_risk: float = 0.01) -> tuple[Size | None, str]:
    """回傳 (Size, "") 或 (None, 不做的原因)。"""
    dist = abs(price - stop)
    if equity <= 0 or dist <= 0 or price <= 0 or f.step <= 0:
        return None, "權益或止損距離不正常"
    want = int(math.floor(equity * risk / dist / f.step + 1e-9))
    floor_steps = max(math.ceil(f.min_qty / f.step - 1e-9),
                      math.ceil(f.min_notional / price / f.step - 1e-9),
                      len(portions) + 1)
    steps = max(want, floor_steps)
    if steps * f.step > f.max_qty:
        steps = int(f.max_qty / f.step)
    risk_frac = steps * f.step * dist / equity
    if risk_frac > max_risk + 1e-12:
        return None, f"最小下單量的風險 {risk_frac:.2%} 超過上限 {max_risk:.2%}"
    return Size(steps, split(steps, portions), risk_frac, steps > want), ""


def split(steps: int, portions=(0.25, 0.25, 0.25)) -> list[int]:
    """把總數量切成各 TP 的量 + 最後剩下的量（都是整數個 step，每段至少 1）。"""
    parts = [max(1, int(steps * p)) for p in portions]
    return parts + [steps - sum(parts)]


def fmt(x: float, unit: float) -> str:
    """依數量 / 價格單位格式化成幣安收的字串。"""
    d = Decimal(str(unit)).normalize()
    places = max(0, -d.as_tuple().exponent)
    return f"{round(x / unit) * unit:.{places}f}"
