# 網站：訊號、績效、策略研究

網址：https://dennyhty.github.io/grid-trading/ 。上方三個分頁：

| 分頁 | 內容 | 資料從哪來 |
| --- | --- | --- |
| 訊號 | 幣安 USDT 永續的實盤維加斯 4H / 2H 訊號列表和圖表；上方一行回測摘要（`web/perf/summary.json`）和網站訊號紀錄 | `signal_snapshot.py` 每 15 分鐘用幣安 K 線掃描，`data/state.json` |
| 績效 → 幣安自動交易 | VPS 上自動交易的實際紀錄：勝率、平均 R、累計 R、已實現損益、回撤、持倉、最近平倉 | VPS 的 `deploy/trader_feed.py` → GitHub secret `TRADER_FEED_URL` → `data/trader.json` |
| 績效 → 維加斯回測 | 實盤規則回測：CAGR、單利年化、最大回撤、Sharpe、Sortino、Calmar、勝率、獲利因子、資金曲線、回撤、滾動期望值、逐年、每月、多空拆開、R 分布 | `web/perf/vegas_backtest.json`（`research/site_performance.py` 產生，固定資料） |
| 策略研究 | 所有研究報告的索引（已上線 / 觀察組 / 待決定 / 不採用）和全文 | `web/perf/research.json` + `reports/**/*.md` |

## 讓網站看到幣安自動交易的紀錄（VPS 上做一次）

自動交易的紀錄只存在 VPS 的 `trader_state/*.json`。`deploy/trader_feed.py` 把它整理成網站要的 JSON（持倉、已平倉損益、R），
用 HTTP 提供出來，網站每 15 分鐘更新時去抓。它只讀 `trader_state`，**不讀 `trader.env`，也不需要任何金鑰**；
未平倉的浮動損益用幣安公開的標記價格估算。

1. 在 VPS 上拿到這兩個檔（還沒合併到你 VPS 上的分支時用這個方法拿）：

   ```bash
   su - trader
   cd grid-trading
   git fetch origin
   git checkout origin/claude/project-thread-t012rc -- deploy/trader_feed.py deploy/trader-feed.service
   ```

2. 產生一串隨機字當網址密碼，寫進 `/home/trader/feed.env`（和放金鑰的 `trader.env` 分開）：

   ```bash
   echo "FEED_TOKEN=$(openssl rand -hex 16)" > /home/trader/feed.env
   chmod 600 /home/trader/feed.env
   cat /home/trader/feed.env          # 記下 = 後面那串字
   python3 deploy/trader_feed.py --print | head -30   # 先看輸出對不對
   exit
   ```

3. 用 root 裝成服務、打開 8787 port：

   ```bash
   cp /home/trader/grid-trading/deploy/trader-feed.service /etc/systemd/system/
   systemctl daemon-reload
   systemctl enable --now trader-feed
   ufw allow 8787/tcp                  # 有開 ufw 才需要；VPS 商的防火牆面板也要放行 8787
   curl http://127.0.0.1:8787/那串字/trader.json | head -c 300   # 有 JSON 就成功
   ```

4. 到 GitHub repo → Settings → Secrets and variables → Actions → New repository secret：
   名稱 `TRADER_FEED_URL`，值 `http://VPS的IP:8787/那串字/trader.json`。

下一次網站更新（最多 15 分鐘）後，「績效 → 幣安自動交易」就會出現紀錄。VPS 連不上時網站沿用上一份，訊號照常更新。

- 網址裡的那串字就是密碼，不對一律 404；不要貼到聊天或 GitHub 檔案裡，只放在 secret。
- 網站本身是公開的，所以持倉和損益任何人都看得到（金鑰、帳戶餘額、訂單編號不會出現）。不想公開就不要設 secret。
- 改了程式後：`systemctl restart trader-feed`。

## 更新回測數字

規則改了（出場、濾網、同幣規則）之後重算：

```bash
pip install numba
python -m research.exit_unify --source binance_ext     # 下載幣安 K 線 + 模擬，約 30~60 分鐘
python -m research.site_performance                    # → web/perf/vegas_backtest.json
```

## 新增研究報告

報告放在 `reports/` 底下，再在 `web/perf/research.json` 對應的分組加一筆（日期、標題、檔名、狀態、一句話結論）。
狀態：`live` 已上線、`observe` 觀察組、`pending` 待決定、`keep` 維持現狀、`rejected` 不採用、`info` 參考。
