"""實驗帳本與多重檢定校正的測試（帳本寫在 tmp 路徑）。"""
import csv
import json

import pytest

from research import experiments as ex
from research.experiments import (COLUMNS, deflated_sharpe, log_experiment, make_id, required_t, seed_ledger,
                                  trials)


def test_log_creates_file_and_counts(tmp_path):
    p = tmp_path / "ledger.csv"
    assert trials(path=p) == 0
    id1 = log_experiment(path=p, family="alpha", params={"b": 2, "a": 1}, period="2020-2023", trades=10,
                         verdict="fail", notes="configs=5")
    id2 = log_experiment(path=p, family="beta", hypothesis="h", verdict="info")
    assert id1.startswith("alpha-") and id2.startswith("beta-")
    assert trials(path=p) == 2
    assert trials("alpha", path=p) == 1
    assert trials("nope", path=p) == 0
    assert trials(path=p, count_configs=True) == 5 + 1
    with p.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert list(rows[0].keys()) == COLUMNS
    assert json.loads(rows[0]["params"]) == {"a": 1, "b": 2}
    assert rows[0]["ts"].endswith("Z")


def test_id_is_deterministic_and_param_sensitive():
    assert make_id("f", {"a": 1, "b": 2}, "p") == make_id("f", {"b": 2, "a": 1}, "p")
    assert make_id("f", {"a": 1}, "p") != make_id("f", {"a": 2}, "p")
    assert make_id("f", {"a": 1}, "p") != make_id("f", {"a": 1}, "q")


def test_log_validates(tmp_path):
    p = tmp_path / "l.csv"
    with pytest.raises(ValueError):
        log_experiment(path=p, family="x", bogus=1)
    with pytest.raises(ValueError):
        log_experiment(path=p, family="x", verdict="maybe")
    with pytest.raises(ValueError):
        log_experiment(path=p, hypothesis="no family")


def test_seed_ledger(tmp_path):
    p = tmp_path / "seed.csv"
    n = seed_ledger(p)
    assert n == len(ex.SEED_ROWS) and 10 <= n <= 20
    assert trials(path=p) == n
    assert trials(path=p, count_configs=True) > n
    assert seed_ledger(p) == 0  # 已存在不重複寫
    assert {r["verdict"] for r in ex.read_ledger(p)} <= set(ex.VERDICTS)


def test_committed_ledger_is_valid():
    rows = ex.read_ledger()
    assert rows and all(r["family"] and r["experiment_id"] for r in rows)


def test_deflated_sharpe_monotone_in_trials():
    vals = [deflated_sharpe(1.5, n, 6.0) for n in (1, 10, 100, 1000, 10000)]
    assert all(a > b for a, b in zip(vals, vals[1:]))
    assert all(0 <= v <= 1 for v in vals)


def test_deflated_sharpe_zero_sr_below_half():
    assert deflated_sharpe(0.0, 1, 6.0) == pytest.approx(0.5)
    assert deflated_sharpe(0.0, 50, 6.0) < 0.5
    assert deflated_sharpe(-0.5, 50, 6.0) < deflated_sharpe(0.0, 50, 6.0)


def test_deflated_sharpe_more_years_and_skew():
    assert deflated_sharpe(1.0, 20, 10.0) > deflated_sharpe(1.0, 20, 3.0)
    assert deflated_sharpe(1.0, 20, 6.0, skew=-1.0) < deflated_sharpe(1.0, 20, 6.0, skew=0.0)
    # 試驗之間夏普差異越大，最大值的期望越高，DSR 越低
    assert deflated_sharpe(1.0, 20, 6.0, sr_var_trials=1.0) < deflated_sharpe(1.0, 20, 6.0, sr_var_trials=0.05)


def test_required_t():
    assert required_t(1) == 2.0
    assert required_t(10) == 2.0 or required_t(10) > 2.0
    assert required_t(100) == pytest.approx(3.29, abs=0.02)
    assert required_t(1000) > required_t(100) > required_t(10)
    assert required_t(0) == 2.0
