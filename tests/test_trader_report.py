import math

from research.trader_report import build, day_cluster_se, stats, to_md

DAY = 86_400_000
NOW = 100 * DAY


def row(closed_days_ago, r, side="LONG"):
    return {"id": f"x{closed_days_ago}{r}", "symbol": "BTCUSDT", "interval": "4H", "side": side,
            "opened": NOW - (closed_days_ago + 0.1) * DAY, "closed": NOW - closed_days_ago * DAY, "pnl": r, "r": r}


def feed(ledger, updated_ago_min=1, paused=False):
    return {"generated": NOW, "books": {"paper": {"updated": NOW - updated_ago_min * 60_000, "paused": paused,
                                                  "positions": [], "ledger": ledger, "processed": {}}}}


def test_stats_basic():
    s = stats([row(1, 1.0), row(1, -1.0, "SHORT"), row(1, 2.0)])
    assert s["trades"] == 3 and abs(s["mean_r"] - 2 / 3) < 1e-9
    assert s["long"]["trades"] == 2 and s["short"]["trades"] == 1
    assert stats([]) == {"trades": 0}


def test_window_filters_old_trades():
    rep = build(feed([row(0.5, 1.0), row(3, -1.0), row(20, 1.0)]), "day", NOW)
    assert rep["books"]["paper"]["window"]["trades"] == 1
    assert rep["books"]["paper"]["all_time"]["trades"] == 3
    assert build(feed([row(0.5, 1.0), row(3, -1.0), row(20, 1.0)]), "week", NOW)["books"]["paper"]["window"]["trades"] == 2


def test_stale_and_paused_notes():
    rep = build(feed([row(0.5, 1.0)], updated_ago_min=120, paused=True), "day", NOW)
    notes = rep["books"]["paper"]["notes"]
    assert any("沒更新" in n for n in notes) and any("暫停" in n for n in notes)


def spread(n_days, per_day, r_of):
    """n_days 個平倉日，每天 per_day 筆；r_of(day, i) 給每筆的 R。"""
    return [row(d + 1, r_of(d, i)) for d in range(n_days) for i in range(per_day)]


def notes_of(ledger):
    return build(feed(ledger), "all", NOW)["books"]["paper"]["notes"]


def test_day_cluster_se_counts_a_day_as_one_sample():
    one_day = [row(1, 1.0), row(1, -1.0), row(1, 2.0)]
    assert day_cluster_se(one_day)[1] == 1 and math.isnan(day_cluster_se(one_day)[0])
    # 每天內的單全同一個 R：分群標準誤遠大於逐筆標準誤
    rows = spread(10, 10, lambda d, i: 1.0 if d % 2 else -1.0)
    se_day, days = day_cluster_se(rows)
    assert days == 10 and se_day > stats(rows)["se"] * 2


def test_demotion_notice_needs_100_trades_30_days_and_low_mean():
    bad = spread(40, 3, lambda d, i: -1.0 if (d + i) % 3 else 0.5)  # 120 筆、40 天、平均遠低於基準
    assert any("降級" in n for n in notes_of(bad))
    assert not any("降級" in n for n in notes_of(bad[:50]))  # 筆數不夠
    assert not any("降級" in n for n in notes_of(spread(10, 12, lambda d, i: -1.0 if (d + i) % 3 else 0.5)))  # 天數不夠
    good = spread(40, 3, lambda d, i: 1.0 if (d + i) % 2 else -0.5)
    assert not any("降級" in n for n in notes_of(good))


def test_demotion_is_looser_when_days_are_correlated():
    # 同樣的平均 R、同樣 120 筆：一天內全贏或全輸（相關）時，分群標準誤大，不該被誤判為低於基準
    corr = spread(40, 3, lambda d, i: 1.0 if d % 2 else -1.4)
    assert not any("降級" in n for n in notes_of(corr))


def test_markdown_renders_empty_window():
    md = to_md(build(feed([]), "day", NOW))
    assert "沒有已平倉的單" in md and "## paper" in md
