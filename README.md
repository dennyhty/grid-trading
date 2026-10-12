# grid-trading

派網 (Pionex) 合約的順勢交易訊號工具。只用公開行情資料，**不需要也不存放任何 API Key**；程式只給建議，開單由你自己在派網 App 操作。

## 交易訊號網頁

「順勢訊號」工具：程式找出進場點，給進場價、TP1/TP2/TP3、BE、SL，
之後自動追蹤 TP 觸及、保護止損移動和出場，也有通知中心。一樣只用公開行情、不會下單。

```bash
pip install -r requirements.txt
python signal_server.py            # 開 http://localhost:8000
python signal_server.py --intervals 4H 60M --keep-days 14   # 加看 1H、顯示 14 天內的訊號
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
RSI 在 50~70（30~50）、價格離通道不超過 3 ATR 時進場。SL = 1.5 ATR，TP = 1R/2R/3R。

回測（CoinMarketCap 市值前 100 名中、派網有合約的 87 個幣，含 0.05% 手續費，每筆止損賠總資金 0.25%）：

| 週期 | 期間 | 筆數 | 勝率 | 期望值 | 總報酬 | 最大回撤 | 同時最多持倉 |
|---|---|---|---|---|---|---|---|
| 4H | 2023-05 ~ 2026-10 | 3421 | 54.0% | +0.085R | +105% | -26% | 29 |
| 1H | 2025-08 ~ 2026-10 | 6399 | 49.7% | -0.048R | -55% | -64% | 43 |

4H 前半段 +0.157R、後半段 +0.047R；按年份 2023 +0.22R、2024 +0.13R、2025 +0.11R，**2026 年到目前約 0R**（1223 筆），
優勢在變小。空單（+0.134R）比多單（+0.023R）好。名單是「現在」的前 100 名，過去表現差、掉出榜外的幣不在裡面（倖存者偏差），
實際會比回測差。每筆賠 1% 時同一段回測最大回撤是 -71%，所以網頁和回測預設每筆 0.25%（`--risk`）。1H 是負的，網頁只看 4H。

加密貨幣名單在 `grid_trading/universe.py`（2026-10-12 的快照）；要改回用成交額門檻：`--crypto-list volume --min-volume 5e6`。
（舊版只看成交額 500 萬以上的 15 個幣：4H 661 筆 +0.019R。）

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
- 公開 repo 的 GitHub Actions 免費；repo 超過 60 天沒有任何 commit，GitHub 會暫停排程，到 Actions 頁面重新啟用即可。

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
| 美股/ETF | 4H | 56 | 341 | 49.3% | -0.017R | 10 |
| 美股/ETF | 1H | 58 | 1362 | 46.6% | -0.131R | 15 |
| 商品 | 4H | 7 | 72 | 54.2% | -0.037R | 4 |
| 商品 | 1H | 7 | 271 | 57.6% | +0.057R | 6 |

都看不出優勢（商品 1H 的 +0.057R 筆數太少，誤差約 ±0.07R）。很多標的會同時出訊號、一起虧，
回測的回撤已經把同時持倉的單都算進去（每個訊號都跟）。
