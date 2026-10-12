# grid-trading

派網 (Pionex) USDT 本位合約網格的選幣與參數工具。只用公開行情資料，**不需要也不存放任何 API Key**；程式只給建議，開單由你自己在派網 App 操作。

## 計畫

1. **選幣篩選器**（本次）：排出目前最適合開網格的幣種。
2. **回測**：測試區間、格數、方向、停損組合，含手續費。
3. **開單建議卡**：幣種、方向、上下限、格數、停損、倉位。
4. **檢討調整**：拿實際結果和回測比對，修正參數。

## 使用

```bash
pip install -r requirements.txt
python screen.py                  # 全部 USDT 永續合約，4H K 線
python screen.py --top 20 --min-volume 1e7
python screen.py --symbols BTC_USDT_PERP ETH_USDT_PERP
python -m pytest                  # 測試
```

完整結果另存在 `output/screen_*.csv`。

## 分數怎麼算

網格靠來回震盪賺錢，單邊趨勢是主要虧損來源，所以分數（0~100）偏好：

| 權重 | 指標 | 偏好 |
|---|---|---|
| 30 | 效率比（淨漲跌 ÷ 路徑總長） | 越低越震盪 |
| 25 | ADX | 越低趨勢越弱 |
| 20 | 收盤穿越 20 均線次數 | 越多越容易成交 |
| 15 | 4H ATR% | 0.8%~3% 最佳 |
| 10 | 24h 成交額 | 越高越好 |

價格已經跑出建議區間的幣（可能正在突破），或區間寬度超過 50%（期間內暴漲暴跌過）的幣，分數各打 6 折。

## 建議欄位

- **方向**：維加斯通道（EMA144/169）。價格在通道上方且通道明顯向上 → 做多；下方且明顯向下 → 做空；其他 → 中性。
- **上下限**：最近 120 根 K 線（4H 約 20 天）高低點的 5% / 95% 分位，排除插針。
- **格數 / 格距**：格距約半根 K 線的 ATR，最低 0.3%（低於這個手續費會吃掉利潤）。
- **停損**：區間外 1 倍 ATR。做多網格只設下方停損，做空網格只設上方停損，中性兩邊都設。

這些是第 1 步的初步參數，第 2 步回測後會再調整。

## 回測

```bash
python backtest.py                # 抓約 14 個月 1H K 線（快取在 data/），跑 24 種參數組合
```

每隔 3/7/14 天，只用當時以前的資料跑篩選器取得參數，再用之後的 K 線模擬網格（含手續費、停損），
前半段當訓練期、後半段當驗證期。結果在 `output/backtest_summary.csv` 和 `output/backtest_windows.csv`。

## 交易訊號網頁（第二個方向）

網格回測沒有正期望值，所以另外做了一個「順勢訊號」工具：程式找出進場點，給進場價、TP1/TP2/TP3、BE、SL，
之後自動追蹤 TP 觸及、保護止損移動和出場，也有通知中心。一樣只用公開行情、不會下單。

```bash
pip install -r requirements.txt
python signal_server.py            # 開 http://localhost:8000
python signal_server.py --intervals 4H 60M --keep-days 14   # 加看 1H、顯示 14 天內的訊號
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

回測（成交額 500 萬 USDT 以上的 15 個加密貨幣，含 0.05% 手續費）：

| 週期 | 期間 | 筆數 | 勝率 | 期望值 | 每筆賠 1% 的總報酬 | 最大回撤 |
|---|---|---|---|---|---|---|
| 4H | 2023-05 ~ 2026-10 | 661 | 51.1% | +0.019R | +8.5% | -25.4% |
| 1H | 2025-09 ~ 2026-10 | 1175 | 49.3% | -0.061R | -55% | -61% |

4H 前後半段都是小正數（+0.033R / +0.010R），但每筆 R 的標準差約 1.2，誤差約 ±0.05R，
所以**看不出真正的優勢**；而且只測了「現在」流動性高的幣（倖存者偏差）。1H 是負的，網頁預設只看 4H。
（第一版數字 +0.041R 混進了 6 檔代幣化美股，現在美股和商品分開統計。）

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

都看不出優勢（商品 1H 的 +0.057R 筆數太少，誤差約 ±0.07R）。另外很多標的會同時出訊號，
加密貨幣 4H 回測期間最多同時 7 筆持倉；每筆都冒 1% 的話虧損會疊加，網頁上的回撤是一筆接一筆算的，實際會更大。
