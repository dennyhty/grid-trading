# grid-trading

派網 (Pionex) 合約的順勢交易訊號工具。只用公開行情資料，**不需要也不存放任何 API Key**；程式只給建議，開單由你自己在派網 App 操作。

## 交易訊號網頁

「順勢訊號」工具：程式找出進場點，給進場價、TP1/TP2/TP3、BE、SL，
之後自動追蹤 TP 觸及、保護止損移動和出場，也有通知中心。一樣只用公開行情、不會下單。

```bash
pip install -r requirements.txt
python signal_server.py            # 開 http://localhost:8000
python signal_server.py --intervals 4H 2H 60M --keep-days 14   # 加看 1H、顯示 14 天內的訊號
python -m pytest                   # 測試
python signal_backtest.py          # 重跑回測，更新 stats/signal_stats.json（網頁上的勝率/期望值）
```

手機要看：手機和電腦連同一個 Wi-Fi，開 `http://<電腦的區網 IP>:8000`（Mac 在「系統設定 → Wi-Fi → 詳細資訊」看得到 IP）。
瀏覽器通知在電腦上可用；iPhone 的 Safari 只有「加入主畫面」且是 HTTPS 網址時才會跳通知，區網 http 時請看頁面右上角的通知中心。

### 頁面功能

- 訊號列表：做多/做空、幣種、週期、觸發時間、訊號價、現價、盈虧（1 倍、含手續費）、TP3~SL 價位梯、觸及時間軸，分「進行中 / 已結束」。
- 訊號圖表：訊號時間（UTC+8）、RSI/MACD/ATR 與觸發原因、K 線＋均線（EMA12/144/169/576/676）＋訊號箭頭＋TP/SL 線、
  週期切換、回到訊號、最新價格即時更新（每 30 秒）、全螢幕。
- 通知中心：新訊號、TP 觸及、保護止損移動、止損/保本出場。
- 回測數字：每個策略、每個週期的勝率、期望值（R）、獲利因子、最大回撤，前後半段分開看。

### 出場規則（所有策略共用）

TP1、TP2、TP3 各平 1/3；TP1 觸及後剩餘倉位止損移到 BE（進場價＋來回手續費），TP2 觸及後移到 TP1。
同一根 K 線同時碰到止損和 TP 時保守算先止損。1R = 進場價到 SL 的距離。

### 第一個策略：維加斯通道順勢 + MACD/RSI（`vegas_trend`）

價格和 EMA12 都在 EMA144/169 通道上方、且 EMA144 > EMA169 時只找多單（反之只找空單）；MACD 柱翻正（翻負）、
RSI 在 54~70（30~46）、價格離通道不超過 3 ATR 時進場。SL = 1.5 ATR，TP = 1R/2R/3R。

訊號只掃派網的合約、用派網的 K 線；回測則以**幣安 USDT 永續的歷史為主**（2020 年起，比派網長，涵蓋 2022 熊市），
派網的回測為輔：`python signal_backtest.py --source binance`（月檔來自 data.binance.vision，只到上個月底；
CRO、MNT、OKB、PI 幣安沒有合約，所以是 83 個幣）。網頁以幣安的數字為主，派網的列在下面。

回測（CoinMarketCap 市值前 100 名中派網有合約的幣，含 0.05% 手續費，每筆止損賠總資金 0.25%）：

| 資料 | 週期 | 期間 | 幣數 | 筆數 | 勝率 | 期望值 | 總報酬 | 最大回撤 | 同時最多持倉 |
|---|---|---|---|---|---|---|---|---|---|
| 幣安 | 4H | 2020-03 ~ 2026-09 | 83 | 3397 | 55.1% | +0.111R | +152% | -21% | 25 |
| 幣安 | 2H | 2020-01 ~ 2026-09 | 83 | 6994 | 53.8% | +0.079R | +289% | -23% | 37 |
| 幣安 | 1H | 2020-01 ~ 2026-09 | 83 | 14464 | 50.9% | -0.015R | -46% | -67% | 41 |
| 派網 | 4H | 2023-04 ~ 2026-10 | 87 | 2130 | 55.5% | +0.127R | +95% | -16% | 24 |
| 派網 | 2H | 2025-09 ~ 2026-10 | 87 | 2052 | 52.6% | +0.036R | +19% | -22% | 37 |
| 派網 | 1H | 2025-08 ~ 2026-10 | 87 | 4062 | 49.1% | -0.057R | -45% | -58% | 40 |

（RSI 收緊前，幣安 4H 是 5316 筆、+0.084R、+200%、回撤 -28%。）

幣安 4H 按年份：2020 +0.03R、2021 +0.31R、2022 +0.19R、2023 +0.12R、2024 +0.14R、2025 +0.09R，**2026 年到 9 月約 0R**（795 筆），
優勢逐年變小。空單（+0.123R）比多單（+0.096R）好。名單是「現在」的前 100 名，過去表現差、掉出榜外的幣不在裡面（倖存者偏差），
實際會比回測差。收緊 RSI 前，每筆賠 1% 時幣安 4H 最大回撤是 -74%，所以網頁和回測預設每筆 0.25%（`--risk`）。1H 是負的，網頁只看 4H 和 2H。

**2H**（2026-10-12 加上）：派網沒有 2H K 線，掃描和回測都由 1H 合成（從 00:00 UTC 起每 2 小時一根，和幣安的 2h 一致）。
幣安 2H 按年份：2020 -0.03R、2021 +0.19R、2022 +0.07R、2023 +0.07R、2024 +0.08R、2025 +0.10R、2026 到 9 月 +0.04R；
2024 年起 4202 筆、+0.075R，平均每天約 4 筆。單數比 4H 多、每筆止損比較近，所以對滑價和手續費比 4H 敏感。
40 天逾時是 240 根 K 線，2H 就是 20 天。網頁上方可以切換「全部週期 / 4H / 2H」，回測數字和實盤紀錄兩個週期分開算；
Telegram 訊息裡會寫週期（例如「BTC/USDT 2H 做多訊號」）。新加一個週期時，補掃到 3 根 K 線以前的舊訊號只記錄、不推播。

加密貨幣名單在 `grid_trading/universe.py`（2026-10-12 的快照）；要改回用成交額門檻：`--crypto-list volume --min-volume 5e6`。
（舊版只看成交額 500 萬以上的 15 個幣：4H 661 筆 +0.019R。）

### RSI 範圍（2026-10-12 收緊）

RSI 原本是多單 50～70、空單 30～50，2026-10-12 改成多單 54～70、空單 30～46。幣安歷史回測（4H 和 2H，84 個幣）裡，
用 2020～2023 年挑設定、2024 年後驗證，或反過來，收緊後每筆期望值和 Sharpe 都比較好，回撤比較小；
被剔除的那段 RSI（多單 50～54、空單 46～50）平均約 0R。代價是單數少三到四成。

每個訊號的結果記在 `data/ledger.json`（`grid_trading/signals/ledger.py`）：GitHub Pages 每次執行會讀回網站上一份接著寫，
還沒結束、但已經超出 7 天顯示範圍的單，會用最新 K 線繼續追蹤到結束。讀不到上一份（不是 404）時不部署，免得紀錄被清空。
網頁回測區下方顯示實盤紀錄。收緊前那段觀察期留下的 `strict_rsi` 標記還在紀錄裡，舊規則才會進的單不算進實盤成績。

### 訂單流研究（還沒上線）

主動買賣差（delta、CVD）、持倉量、多空比的策略回測，只用幣安的歷史（派網沒有這些資料）。
策略在 `grid_trading/signals/orderflow.py`，沒有登記到網頁用的策略清單。

```bash
python -c "from grid_trading.binance_history import fetch_binance; from grid_trading.binance_metrics import fetch_metrics; \
  from grid_trading.universe import TOP100; [fetch_binance(c, '1h', flow=True) and fetch_metrics(c, '2026-10-11') for c in TOP100]"
python orderflow_backtest.py --tf 4h     # 112 組參數、正反向切分、隨機進場對照、過濾 vegas_trend
python orderflow_oi_check.py --tf 4h     # 持倉量 × 價格的擴大參數表和「只看價格」對照
```

結論（2026-10-12）：沒有一個穩定贏過 4H vegas_trend；CVD 背離反向做每筆約 -0.1R。

### 加新策略

在 `grid_trading/signals/` 新增一個檔案，繼承 `Strategy`、實作 `prepare()`（算指標）和 `check(df, i)`
（第 i 根收盤時是否進場，回傳方向、SL、三個 TP、資訊），加上 `@register`，再到 `grid_trading/signals/__init__.py` import。
之後 `python signal_backtest.py --strategy 新名稱` 回測、`python signal_server.py --strategies vegas_trend 新名稱` 上線。

圖表使用 [TradingView Lightweight Charts](https://github.com/tradingview/lightweight-charts)（Apache 2.0，放在 `web/vendor/`）。

### 放在 GitHub Pages（不用開電腦）

`.github/workflows/pages.yml` 每 5 分鐘在 GitHub 上跑一次 `signal_snapshot.py`：掃描訊號、追蹤 TP/SL、產生圖表資料，
連同網頁一起直接發佈到 GitHub Pages。網址是 `https://dennyhty.github.io/grid-trading/`。

- 第一次要做：到 repo 的 Settings → Pages，Source 選「GitHub Actions」。
- 訊號和 TP/SL 觸及最快約 5 分鐘更新一次（GitHub 排程最短 5 分鐘，而且常延遲幾分鐘）。
- 派網的行情 API 不允許網頁直接讀，所以網頁上的「現價」改由瀏覽器向 Binance U 本位合約抓公開即時價（每 30 秒），
  和派網價差通常很小；抓不到或差超過 5%（不是同一個商品）時，顯示掃描當下的派網價格。
- 本機版 `python signal_server.py` 照常可用，網頁會自動判斷是哪一種模式。
- Telegram 推播：新訊號、TP 觸及、移動止損、出場都會傳到 Telegram（`signal_telegram.py`），網站部署成功後才送，傳過的不重送。
  設定方式：在 Telegram 找 @BotFather 用 `/newbot` 建一個 Bot 拿到 token，打開自己的 Bot 按「開始」，
  再找 @userinfobot 拿到自己的 Id；然後到 repo 的 Settings → Secrets and variables → Actions 新增
  `TELEGRAM_BOT_TOKEN` 和 `TELEGRAM_CHAT_ID` 兩個 secret。沒設定就不推。token 不要寫進任何檔案。
- GitHub 內建的定時排程在這個 repo 一直沒有觸發，所以另外有 `.github/workflows/ticker.yml`（訊號定時器）：
  一個工作每 5 分鐘觸發一次訊號網頁，跑約 5 小時 40 分後自己接力開下一個。中斷時到 Actions → 訊號定時器 → Run workflow 重新啟動；
  要停就在同一頁 Disable workflow。這種一直開著的用法在 GitHub 規範裡算灰色地帶，有可能被 GitHub 停掉。
- 公開 repo 的 GitHub Actions 免費。

### 美股、商品（RWA）

派網的永續合約除了加密貨幣，還有代幣化美股/ETF（代號是股票代號加 X，例如 NVDAX、QQQX、MUX＝美光）和商品
（XAU 黃金、XAG 白銀、XPT、XPD、WTI、BRENTOIL、NATGAS、COPPER）。分類在 `grid_trading/markets.py`。

- 網頁上分成「加密貨幣 / 美股 / 商品」三個分頁。美股和商品的成交額比較小，門檻另外用 `--min-volume-rwa`（預設 30 萬 USDT）。
- 回測：`python signal_backtest.py --market stock`（或 `commodity`），結果存在 stats 的 `stock|4H` 這種鍵。
  還沒回測過的類別，網頁會標「未回測」。
- 美股和商品的現價用掃描時的派網價格（Binance 大多沒有這些商品，價格也不一定一樣）。

維加斯順勢策略在 RWA 上的回測（含 0.05% 手續費；RWA 大多 2025 下半年才上市，資料短）：

| 類別 | 週期 | 標的數 | 筆數 | 勝率 | 期望值 | 同時最多持倉 |
|---|---|---|---|---|---|---|
| 美股/ETF | 4H | 55 | 220 | 47.3% | -0.054R | 6 |
| 美股/ETF | 1H | 60 | 878 | 48.7% | -0.072R | 14 |
| 商品 | 4H | 7 | 48 | 52.1% | -0.081R | 4 |
| 商品 | 1H | 7 | 137 | 56.2% | -0.046R | 3 |

都看不出優勢（RSI 收緊後重跑，2026-10-12）。很多標的會同時出訊號、一起虧，
回測的回撤已經把同時持倉的單都算進去（每個訊號都跟）。
