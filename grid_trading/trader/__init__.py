"""幣安 USDT 本位合約自動交易：照實盤維加斯訊號（4H + 2H）開倉，止損和止盈掛在交易所上。

三種模式（binance_trader.py --mode）：
- paper：不用金鑰，用幣安即時 K 線算訊號、用回測同一套 tracker 模擬出場，只記帳。
- testnet：幣安合約測試網（假錢），真的下單、掛止損止盈、移動止損。
- live：正式帳戶。要另外設環境變數 TRADER_ALLOW_LIVE=1 才會啟動。

金鑰只從環境變數 BINANCE_API_KEY / BINANCE_API_SECRET 讀，不寫進程式、檔案或 git。
"""
