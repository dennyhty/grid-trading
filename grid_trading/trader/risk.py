"""帳戶層級的風控：同時最多幾張、回撤到一定 R 數後風險減半（回到新高才恢復）。

R 都以「基本風險金額」計（開倉當下權益 × 每筆風險），減半期間的單一筆止損算 −0.5R。
兩個規則預設都關閉，由 binance_trader.py 的 --max-open、--dd-half-r 打開
（研究見 report_drawdown_2026_20261019.md：同時最多 15 張 + 回撤 20R 後減半）。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RiskRules:
    max_open: int | None = None    # 同時最多幾張；None 不限
    dd_half_r: float | None = None  # 從高點回撤幾 R 後每筆風險減半；None 不用


def equity_curve_r(ledger: list[dict]) -> tuple[float, float]:
    """已平倉紀錄 → (累計 R, 歷史最高累計 R)。"""
    cum = peak = 0.0
    for row in ledger:
        cum += float(row.get("r", 0.0))
        peak = max(peak, cum)
    return cum, peak


def risk_multiplier(ledger: list[dict], rules: RiskRules) -> float:
    """回撤達到 dd_half_r 後減半，直到累計 R 回到之前的最高點才恢復。"""
    if not rules.dd_half_r:
        return 1.0
    cum = peak = 0.0
    halved = False
    for row in ledger:
        cum += float(row.get("r", 0.0))
        if cum >= peak:
            peak, halved = cum, False
        elif peak - cum >= rules.dd_half_r:
            halved = True
    return 0.5 if halved else 1.0


def can_open(open_count: int, rules: RiskRules) -> bool:
    return rules.max_open is None or open_count < rules.max_open
