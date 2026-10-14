import numpy as np
import pandas as pd

from research.dca_backtest import Rule, simulate


def _run(rule, cond=None, mult=None, opens=None):
    opens = np.array(opens or [100.0, 50.0, 100.0, 200.0])
    n = len(opens)
    f = pd.DataFrame({"close": opens})
    return simulate(f, opens, opens, np.ones(n), rule, cond, mult)


def test_benchmark_spends_every_day():
    res = _run(None)
    assert res["deployed"] == 1.0
    assert res["buy_days"] == 4


def test_all_in_waits_for_signal():
    cond = np.array([False, True, False, False])
    res = _run(Rule("t", "x", "all"), cond=cond)
    assert res["buy_days"] == 1
    assert res["spent"] == 2.0  # 第 1、2 天的錢在第 2 天一起買
    assert res["cash_left"] == 2.0


def test_deadline_forces_old_cash():
    cond = np.zeros(4, dtype=bool)
    res = _run(Rule("t", "x", "all", deadline=2), cond=cond)
    # 第 0 天的錢在第 2 天到期，第 1 天的錢在第 3 天到期
    assert res["spent"] == 2.0
    assert res["cash_left"] == 2.0


def test_scaled_is_capped_by_savings():
    mult = np.array([0.0, 3.0, 3.0, 1.0])
    res = _run(Rule("t", None, "scaled", mult="m"), mult=mult)
    # 第 0 天存 1、第 1 天買 2（全部）、第 2 天只能買 1、第 3 天買 1
    assert res["spent"] == 4.0
    assert res["buy_days"] == 3
