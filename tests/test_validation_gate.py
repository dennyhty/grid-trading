"""策略驗證關卡的測試：全部用合成交易，不連網；帳本寫在 tmp 路徑。"""
import numpy as np
import pandas as pd
import pytest

from research.experiments import read_ledger
from research.validation_gate import GateConfig, evaluate, report_md, run_cli


def make_trades(n=2000, mean=0.2, seed=0, start="2021-01-01", end="2025-12-31", sd=1.0, n_symbols=40) -> pd.DataFrame:
    """合成交易：進場時間在區間內均勻分布，持有 4～96 小時，多空各半，散在 n_symbols 個幣。"""
    rng = np.random.default_rng(seed)
    lo, hi = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")
    secs = rng.uniform(0, (hi - lo).total_seconds(), n)
    t = (lo + pd.to_timedelta(secs, unit="s")).floor("h")
    hold = pd.to_timedelta(rng.integers(4, 97, n), unit="h")
    r = rng.normal(mean, sd, n)
    return pd.DataFrame({"symbol": [f"C{k}USDT" for k in rng.integers(0, n_symbols, n)],
                         "side": rng.choice([1, -1], n), "time": t, "exit_t": t + hold, "r": r, "r_net": r}
                        ).sort_values("exit_t").reset_index(drop=True)


CFG = GateConfig(n_trials=1)


def status(res, name):
    return res["checks"][name]["status"]


def test_pure_noise_fails():
    res = evaluate(make_trades(mean=0.0, seed=1), cfg=CFG)
    assert not res["passed"]
    assert res["verdict"].startswith("FAIL")
    assert status(res, "cluster_t") == "fail"
    assert status(res, "deflated_sharpe") == "fail"


def test_clear_edge_passes_basic_checks_and_is_incomplete():
    res = evaluate(make_trades(mean=0.2, seed=2), cfg=CFG)
    for name in ("n_trades", "mean_in_sample", "mean_oos", "cluster_t", "deflated_sharpe",
                 "robust_drop_coins", "robust_drop_year", "robust_drop_days"):
        assert res["checks"][name]["passed"], (name, res["checks"][name])
    assert res["passed"] and res["failed"] == []
    # 沒給隨機基準／高原／基準帳本：略過，不算通過，結論是 incomplete
    assert set(res["skipped"]) == {"beats_random", "plateau", "baseline_corr"}
    assert res["incomplete"] and res["verdict"].startswith("INCOMPLETE")
    assert res["checks"]["beats_random"]["passed"] is False
    c = res["checks"]["cluster_t"]
    assert set(c["per_clustering"]) == {"進場日+方向", "出場日+方向", "ISO週(進場)", "月(進場)"}
    assert c["value"] == min(c["per_clustering"].values())
    assert res["checks"]["sides"]["status"] == "info"


def test_multiple_testing_threshold_tightens_with_trials():
    tr = make_trades(mean=0.08, n=2000, seed=3)   # 最小 t 約 2.5～3.5：1 組時過 2.0，1,743 組時過不了
    small = evaluate(tr, cfg=GateConfig(n_trials=1))["checks"]["cluster_t"]
    assert 2.0 < small["value"] < 4.0 and small["passed"]
    big = evaluate(tr, cfg=GateConfig(n_trials=1743))["checks"]["cluster_t"]
    assert not big["passed"] and 3.9 < big["threshold"] < 4.1


def test_edge_only_before_oos_fails_oos():
    a = make_trades(n=1500, mean=0.35, seed=4, start="2021-01-01", end="2023-12-31")
    b = make_trades(n=500, mean=-0.15, seed=5, start="2024-01-01", end="2025-12-31")
    res = evaluate(pd.concat([a, b], ignore_index=True), cfg=CFG)
    assert status(res, "mean_in_sample") == "pass"
    assert status(res, "mean_oos") == "fail"
    assert not res["passed"]


def test_oos_needs_enough_trades():
    a = make_trades(n=1500, mean=0.3, seed=6, start="2021-01-01", end="2023-12-31")
    b = make_trades(n=50, mean=0.3, seed=7, start="2024-01-01", end="2025-12-31")
    res = evaluate(pd.concat([a, b], ignore_index=True), cfg=CFG)
    assert status(res, "mean_oos") == "fail"       # 平均為正但筆數 < min_trades/3


def test_edge_concentrated_in_one_crash_day_fails_robustness():
    base = make_trades(n=2000, mean=-0.05, seed=8)
    crash = make_trades(n=100, mean=3.0, seed=9, sd=0.3, start="2025-10-10", end="2025-10-10 06:00:00")
    crash["time"] = pd.Timestamp("2025-10-10 00:00", tz="UTC")
    crash["exit_t"] = pd.Timestamp("2025-10-10 12:00", tz="UTC")
    res = evaluate(pd.concat([base, crash], ignore_index=True), cfg=CFG)
    assert res["checks"]["n_trades"]["passed"]
    assert res["info"]["mean_r"] > 0
    assert status(res, "robust_drop_days") == "fail"
    assert res["checks"]["robust_drop_days"]["value"] < 0
    assert not res["passed"]


def test_dropping_top_coins_can_fail():
    base = make_trades(n=2000, mean=-0.05, seed=10, n_symbols=40)
    whale = make_trades(n=300, mean=1.0, seed=11, n_symbols=1).assign(symbol="WHALEUSDT")
    res = evaluate(pd.concat([base, whale], ignore_index=True), cfg=CFG)
    assert res["info"]["mean_r"] > 0
    assert status(res, "robust_drop_coins") == "fail"


def test_correlation_with_baseline_clone_fails():
    tr = make_trades(mean=0.2, seed=12)
    res = evaluate(tr, baseline=tr.copy(), cfg=CFG)
    c = res["checks"]["baseline_corr"]
    assert c["status"] == "fail" and c["value"] == pytest.approx(1.0)
    assert not res["passed"] and "baseline_corr" in res["failed"]
    # 獨立的基準帳本相關性低，通過
    other = evaluate(tr, baseline=make_trades(mean=0.2, seed=99), cfg=CFG)["checks"]["baseline_corr"]
    assert other["status"] == "pass" and abs(other["value"]) < 0.5


def test_random_baseline_paired_by_year():
    edge = make_trades(mean=0.2, seed=13)
    noise = make_trades(mean=0.0, seed=14)
    ok = evaluate(edge, random_baseline=noise, cfg=CFG)["checks"]["beats_random"]
    assert ok["status"] == "pass" and ok["diff"] == pytest.approx(0.2, abs=0.1)
    bad = evaluate(noise, random_baseline=make_trades(mean=0.0, seed=15), cfg=CFG)["checks"]["beats_random"]
    assert bad["status"] == "fail"


def test_plateau():
    main = make_trades(mean=0.2, seed=16)
    good = [make_trades(mean=0.18, seed=17), make_trades(mean=0.25, seed=18)]
    assert evaluate(main, neighbors=good, cfg=CFG)["checks"]["plateau"]["status"] == "pass"
    spike = good + [make_trades(mean=0.02, seed=19)]            # 鄰居掉到主設定的 50% 以下
    assert evaluate(main, neighbors=spike, cfg=CFG)["checks"]["plateau"]["status"] == "fail"
    neg = good + [make_trades(mean=-0.1, seed=20)]
    assert evaluate(main, neighbors=neg, cfg=CFG)["checks"]["plateau"]["status"] == "fail"


def test_all_optional_checks_given_gives_full_verdict():
    main = make_trades(mean=0.2, seed=21)
    res = evaluate(main, baseline=make_trades(mean=0.1, seed=22), random_baseline=make_trades(mean=0.0, seed=23),
                   neighbors=[make_trades(mean=0.2, seed=24)], cfg=CFG)
    assert res["skipped"] == [] and not res["incomplete"]
    assert res["passed"] and res["verdict"].startswith("PASS")


def test_too_few_trades_fails():
    res = evaluate(make_trades(n=100, mean=0.5, seed=25), cfg=CFG)
    assert status(res, "n_trades") == "fail"


def test_missing_columns_raise():
    with pytest.raises(ValueError):
        evaluate(make_trades().drop(columns=["exit_t"]), cfg=CFG)
    with pytest.raises(ValueError):
        evaluate(make_trades().drop(columns=["r_net"]), cfg=CFG)


def test_default_n_trials_reads_ledger():
    from research.experiments import trials
    assert GateConfig().n_trials == trials(count_configs=True)


def test_report_md_traditional_chinese_table():
    res = evaluate(make_trades(mean=0.2, seed=26), cfg=CFG)
    md = report_md(res, "demo")
    assert "策略驗證關卡：demo" in md and "| 結果 | 檢查 |" in md
    assert "✓" in md and "–" in md                 # 通過與略過都有
    fail_md = report_md(evaluate(make_trades(mean=0.0, seed=1), cfg=CFG), "noise")
    assert "✗" in fail_md


def test_cli_logs_to_tmp_ledger(tmp_path, capsys):
    p = tmp_path / "t.csv.gz"
    make_trades(mean=0.0, seed=27).to_csv(p, index=False)
    ledger = tmp_path / "ledger.csv"
    # 不帶 --log：不寫帳本
    res = run_cli(["--trades", str(p), "--name", "noise", "--pool", "none", "--n-trials", "5", "--ledger", str(ledger)])
    assert not res["passed"] and "策略驗證關卡：noise" in capsys.readouterr().out
    assert not ledger.exists()
    run_cli(["--trades", str(p), "--name", "noise", "--pool", "none", "--n-trials", "5", "--log", "--ledger", str(ledger)])
    rows = read_ledger(ledger)
    assert len(rows) == 1 and rows[0]["family"] == "noise" and rows[0]["verdict"] == "fail"
    assert rows[0]["trades"] == "2000"
