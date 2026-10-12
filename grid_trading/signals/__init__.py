"""交易訊號：策略（strategy）、訊號後續追蹤（tracker）、回測（evaluate）、即時掃描（engine）。"""

from .strategy import LONG, REGISTRY, SHORT, Signal, Strategy, get, register  # noqa: F401
from . import vegas_breakout, vegas_trend  # noqa: F401  載入時自動登記策略
