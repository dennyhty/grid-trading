import pandas as pd

from grid_trading.pit_universe import (
    coin_of, keep_symbol, membership, summarize_month, universe,
)


def test_keep_symbol_filters():
    syms = ["BTCUSDT", "1000PEPEUSDT", "LUNAUSDT", "BTCUSDT_230331", "ETHBUSD", "BTCUSDC", "USDCUSDT",
            "FDUSDUSDT", "EURUSDT", "BTCDOMUSDT", "DEFIUSDT", "FOOTBALLUSDT", "BLUEBIRDUSDT", "ETHDOMUSDT",
            "币安人生USDT", "USDT"]
    assert [s for s in syms if keep_symbol(s)] == ["BTCUSDT", "1000PEPEUSDT", "LUNAUSDT", "币安人生USDT"]


def test_coin_of():
    assert coin_of("BTCUSDT") == "BTC"
    assert coin_of("1000PEPEUSDT") == "1000PEPE"
    assert coin_of("1000000MOGUSDT") == "1000000MOG"


def make_table() -> pd.DataFrame:
    rows = [  # symbol, month, quote_volume, days
        ("AAAUSDT", "2021-01", 500, 31), ("BBBUSDT", "2021-01", 400, 31), ("CCCUSDT", "2021-01", 300, 31),
        ("NEWUSDT", "2021-01", 900, 10),  # 上市不滿 20 天：不入選
        ("AAAUSDT", "2021-02", 100, 28), ("BBBUSDT", "2021-02", 200, 28), ("NEWUSDT", "2021-02", 150, 28),
        ("LUNAUSDT", "2021-02", 300, 28),  # 之後下架
        ("BBBUSDT", "2021-03", 50, 31),
    ]
    return pd.DataFrame([{"symbol": s, "month": m, "quote_volume": float(v), "days": d,
                          "first_day": "", "last_day": ""} for s, m, v, d in rows])


def test_universe_uses_previous_month_and_min_days():
    t = make_table()
    assert universe("2021-02", n=2, table=t) == ["AAAUSDT", "BBBUSDT"]
    assert universe(pd.Period("2021-02", "M"), n=10, table=t) == ["AAAUSDT", "BBBUSDT", "CCCUSDT"]  # NEW 只有 10 天
    # 3 月看 2 月的成交額：LUNA 排第一，NEW 第二次才夠天數
    assert universe("2021-03", n=3, table=t) == ["LUNAUSDT", "BBBUSDT", "NEWUSDT"]
    assert universe("2021-03", n=1, table=t) == ["LUNAUSDT"]
    assert universe("2021-01", table=t) == []  # 沒有上個月的資料
    assert universe("2021-02", n=10, min_days=5, table=t)[0] == "NEWUSDT"


def test_membership_shape_and_consistency():
    t = make_table()
    m = membership(n=2, table=t)
    assert list(m.columns) == ["symbol", "month"]
    assert set(m["month"]) == {"2021-02", "2021-03", "2021-04"}  # 資料月的下一個月
    assert len(m) == 2 + 2 + 1 and not m.duplicated().any()
    for month, g in m.groupby("month"):
        assert sorted(g["symbol"]) == sorted(universe(month, n=2, table=t))
    # LUNA 下架後的 2021-04 仍可看到它在 2021-03 的成員資格（2 月成交額）
    assert ("LUNAUSDT", "2021-03") in set(zip(m["symbol"], m["month"]))
    assert ("LUNAUSDT", "2021-04") not in set(zip(m["symbol"], m["month"]))


def test_summarize_month_header_and_microseconds():
    raw = ("open_time,open,high,low,close,volume,close_time,quote_volume,count,tb,tbq,ignore\n"
           "1609459200000000,1,1,1,1,10,1,100,1,1,1,0\n"   # 2021-01-01，微秒
           "1609545600000000,1,1,1,1,0,1,0,0,0,0,0\n"      # 無量
           "1609632000000000,1,1,1,1,5,1,50,1,1,1,0\n")
    r = summarize_month(raw)
    assert r == {"quote_volume": 150.0, "days": 2, "first_day": "2021-01-01", "last_day": "2021-01-03"}
