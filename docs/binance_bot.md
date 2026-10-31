# 幣安自動交易：VPS 安裝與使用

程式：`binance_trader.py`（`grid_trading/trader/`）。照實盤維加斯 4H + 2H 的規則自動開倉，止損和三個止盈掛在幣安上，
程式只負責移動止損。網站和 Telegram 推播照舊在 GitHub 上跑，這個程式是另外一個服務，兩邊互不影響。

上線順序：**紙上交易（不用金鑰）2 週 → 測試網 2 週 → 小額實盤**。每一段沒問題才進下一段。

## 0. 先把 VPS 弄安全

- 改 root 密碼：`passwd`
- 改用 SSH 金鑰登入（在自己電腦上 `ssh-keygen -t ed25519`，再 `ssh-copy-id root@你的IP`），確定能用金鑰登入後，
  在 `/etc/ssh/sshd_config` 設 `PasswordAuthentication no`，然後 `systemctl restart ssh`。
- 密碼、API 金鑰都不要貼到聊天、GitHub 或任何文件裡。

## 1. 安裝（一次就好）

```bash
apt update && apt install -y git python3 python3-venv
useradd -m -s /bin/bash trader          # 用一般帳號跑，不用 root
su - trader
git clone https://github.com/dennyhty/grid-trading.git
cd grid-trading
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
curl https://fapi.binance.com/fapi/v1/time   # 有回時間數字才繼續
```

## 2. 紙上交易（不用金鑰）

```bash
.venv/bin/python binance_trader.py --mode paper --once   # 先跑一輪看有沒有錯
```

沒錯的話裝成背景服務（回到 root 執行）：

```bash
cp /home/trader/grid-trading/deploy/binance-trader.service /etc/systemd/system/binance-trader@.service
systemctl daemon-reload
systemctl enable --now binance-trader@paper
journalctl -u binance-trader@paper -f      # 看即時紀錄，Ctrl+C 離開
```

要推 Telegram 的話，把 Bot Token 和 Chat ID 寫進 `/home/trader/trader.env`（見第 3 步，同一個檔案）。
紀錄在 `/home/trader/grid-trading/trader_state/paper.json`。

## 3. 測試網

1. 到幣安合約測試網（testnet.binancefuture.com）登入，在 API Key 頁面建一組測試金鑰。這是假錢，和正式帳戶無關。
2. 在 VPS 上寫環境變數檔（只有 trader 帳號讀得到）：

```bash
cat > /home/trader/trader.env <<'EOF'
BINANCE_API_KEY=貼上測試網金鑰
BINANCE_API_SECRET=貼上測試網密鑰
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=
EOF
chmod 600 /home/trader/trader.env && chown trader /home/trader/trader.env
systemctl stop binance-trader@paper
systemctl enable --now binance-trader@testnet
```

程式第一次下單時會自動把帳戶設成雙向持倉，每個幣設成逐倉 5 倍。

## 4. 正式帳戶（測試網跑過再做）

- 在幣安正式帳戶建 API 金鑰：只勾「讀取」和「合約交易」，**不要勾提幣**，IP 限制填這台 VPS 的 IP。
- 把 `trader.env` 換成正式金鑰，加一行 `TRADER_ALLOW_LIVE=1`，然後 `systemctl enable --now binance-trader@live`。
- 一開始只放 100–300U 在合約帳戶。

## 日常操作

| 要做的事 | 指令 |
| --- | --- |
| 暫停開新倉（已開的照常管理） | `touch /home/trader/grid-trading/trader_state/pause` |
| 恢復 | `rm /home/trader/grid-trading/trader_state/pause` |
| 看紀錄 | `journalctl -u binance-trader@testnet -f` |
| 更新程式 | `su - trader -c 'cd grid-trading && git pull'`，再 `systemctl restart binance-trader@testnet` |
| 檢查某個幣在程式和幣安上對不對得上（唯讀） | `su - trader -c 'cd grid-trading && set -a && . /home/trader/trader.env && set +a && .venv/bin/python deploy/trader_diag.py APTUSDT'` |
| 完全停止 | `systemctl stop binance-trader@testnet`（交易所上的止損止盈還在，倉位不會沒保護） |

## 規則

- 進場：只做剛收盤那根的訊號，停機期間錯過的不補。同一個幣同一個週期一次一張；2H、4H 可以同時各一張（同方向也可以），
  在幣安上會合成同一個倉位，但每張單的止損、止盈各用自己的數量掛、各自管理。
- 每筆風險 0.25%（`--risk`）。最小下單量讓實際風險超過 1%（`--max-risk`）的訊號不做並推播原因。
- 出場：止損 1.5 ATR；TP 1R / 3R / 5R 各平 1/4；TP1 後移到保本、TP2 後移到 TP1；TP3 後最後 1/4 用 3 ATR 追蹤，
  每根 K 線收盤更新；240 根還沒結束就市價平倉。價位以實際成交價為準，止損距離照訊號的比例。
- 回撤控制（預設關，Denny 2026-10-20 決定不限張數）：`--max-open 15` 同時最多 15 張；`--dd-half-r 20` 回撤 20R 後風險減半，回到新高才恢復。
  要開就改 `deploy/binance-trader.service` 的 `ExecStart` 加上這兩個參數。
- 止損掛不上就立刻市價平倉；止損單不見了會自動補掛；不是這個程式開的倉位只提醒、不碰。
- 同幣同向兩張單時有個小縫：某張的 TP 成交到程式把止損數量改小之間（最多約 1 分鐘），若剛好打到止損，
  可能多平到另一張的一點點數量。
