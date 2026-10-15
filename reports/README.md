# 研究報告索引

依主題排，最新的在前。系統整體和策略現況見 [docs/system.md](../docs/system.md)。
報告裡提到的 CSV 明細在專案共用資料夾（signals/、backtests/、dca/），沒有放進 repo。

## 實盤策略：維加斯順勢（vegas_trend）

| 日期 | 報告 | 一句話結論 |
|---|---|---|
| 10-15 | [止損單共同特徵](signals/report_vegas_trend_stoploss_profile_20261015.md) | 近期止損單偏「大 K、爆量」，但這群歷史上最賺；沒有特徵能預測止損，規則不改，建議設總風險上限 |
| 10-13 | [進場條件與選幣](signals/report_vegas_trend_entry_20261013.md) | 濾網、模型、挑幣都過不了樣本外；只有 2H「ADX≥24 不做」留下（已上線） |
| 10-13 | [止損深入研究](signals/report_vegas_trend_stoploss_20261013.md) | 96 種止損都沒有比 1.5 ATR 好，維持不變 |
| 10-13 | [止損與止盈](signals/report_vegas_trend_exits_20261013.md) | 讓獲利跑比 1/2/3R 好；4H 1R 平一半 + 追蹤、2H 1/3/5R + 追蹤（已上線） |
| 10-12 | [勝率、獲利因子、期望值](signals/report_vegas_trend_metrics_20261012.md) | 沒有組合同時過期望值和獲利因子兩個門檻（舊出場） |
| 10-12 | [對照基準](signals/report_vegas_trend_benchmarks_20261012.md) | 2H 每段都贏隨機進場；4H 近期較弱 |
| 10-12 | [最近 30 天到 2 年](signals/report_vegas_trend_4H_periods_20261012.md) | 長期優勢是真的，近兩年變弱 |
| 10-12 | [2H RSI/MACD](signals/report_vegas_trend_2H_rsi_macd_20261012.md) | 收緊 RSI 在 2H 更有幫助 |
| 10-12 | [4H RSI/MACD](signals/report_vegas_trend_rsi_macd_20261012.md)、[對調驗證](signals/report_vegas_trend_rsi_macd_reversed_20261012.md) | 收緊 RSI（54–70 / 30–46）兩個方向驗證都贏（已上線） |
| 10-12 | [多週期（幣安）](signals/report_timeframes_binance_20261012.md)、[多週期（派網）](signals/report_timeframes_20261012.md) | 4H 最穩、2H 次之、1H 負的 |

## 其他策略

| 日期 | 報告 | 一句話結論 |
|---|---|---|
| 10-13 | [基礎指標（深入版）](signals/report_basic_indicators_opus_max_20261013.md) | 反轉、背離都輸；勝率由出場決定；2H 維加斯仍最好 |
| 10-13 | [基礎指標（普通版）](signals/report_basic_indicators_opus_medium_20261013.md) | 40 種指標都沒贏維加斯；RSI 背離 + EMA200 列候選 |
| 10-12 | [訂單流](signals/report_orderflow_20261012.md) | CVD 背離最差；價漲倉增當觀察組 |
| 10-12 | [SMC / ICT](signals/report_smc_ict_20261012.md) | 只有 4H 掃流動性 → 結構轉變站得住，當觀察組 |
| 10-12 | [強弱輪動、均值回歸、VWAP](signals/report_alt_strategies_20261012.md) | 輪動暫緩；均值回歸負的；VWAP 沒東西 |
| 10-12 | [維加斯 + 突破 1H](signals/report_combined_1h_20261012.md) | 1H 資料短、後半段變 0 |
| 10-12 | [外部訊號群回推](signals/report_group_signals_20261012.md)、[驗證與前 100 名回測](signals/report_group_signals_top100_20261012.md) | 照單跟每筆約 −0.10R |

## 網格（已放棄）

| 日期 | 報告 | 一句話結論 |
|---|---|---|
| 10-12 | [網格回測](grid/report_20261012.md)、[長週期](grid/report_cycle_20261012.md)、[改良第二輪](grid/report_round2_20261012.md) | 每個變形都虧：小賺很多次、大虧一次 |

## 長期投資

| 日期 | 報告 | 一句話結論 |
|---|---|---|
| 10-13 | [BTC 定投](dca/report_btc_dca_20261013.md) | 等低點才買輸給每天買；ahr999>1.2 暫停的版本 4 年視窗全贏 |
