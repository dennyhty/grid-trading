from research.trader_report import build, stats, to_md

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


def test_demotion_notice_needs_100_trades_and_low_mean():
    bad = [row(1, -1.0 if i % 3 else 0.5) for i in range(120)]
    assert any("降級" in n for n in build(feed(bad), "all", NOW)["books"]["paper"]["notes"])
    assert not any("降級" in n for n in build(feed(bad[:50]), "all", NOW)["books"]["paper"]["notes"])
    good = [row(1, 1.0 if i % 2 else -0.5) for i in range(120)]
    assert not any("降級" in n for n in build(feed(good), "all", NOW)["books"]["paper"]["notes"])


def test_markdown_renders_empty_window():
    md = to_md(build(feed([]), "day", NOW))
    assert "沒有已平倉的單" in md and "## paper" in md
