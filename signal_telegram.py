"""把新的訊號通知推到 Telegram。GitHub Actions 每次掃描後執行，分兩步：

  python signal_telegram.py outbox site/data previous_notifications.json outbox.json
      比對這次和上一次的通知，把新的寫進 outbox.json（網站部署前）。
  python signal_telegram.py send outbox.json
      網站部署成功後才送出，避免部署失敗、下一次又重送同樣的通知。

Bot Token 和 Chat ID 只從環境變數 TELEGRAM_BOT_TOKEN、TELEGRAM_CHAT_ID 讀
（放在 GitHub 的 repository secrets），不寫進程式或檔案。沒設定就什麼都不送。
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

SITE_URL = "https://dennyhty.github.io/grid-trading/"
MAX_MESSAGES = 30  # 一次最多送幾則，避免洗版；其餘請看網站


def _load(path: Path):
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _fmt(x: float) -> str:
    return f"{x:.6g}"


def build_outbox(data_dir: Path, previous_path: Path) -> list[str]:
    """這次新增、而且不是第一次啟動補上的通知，轉成 Telegram 訊息文字。觀察組（策略有 tag）的通知不送。"""
    previous = _load(previous_path)
    if not previous:
        return []  # 拿不到上一次的紀錄就不送，免得把舊通知全部重送一遍
    seen = {n["id"] for n in previous}
    notes = _load(data_dir / "notifications.json") or []
    state = _load(data_dir / "state.json") or {}
    signals = {s["id"]: s for s in state.get("signals", [])}
    observe = {name for name, st in (state.get("strategies") or {}).items() if st.get("tag")}
    out = []
    for n in notes:
        if n["id"] in seen or n.get("initial"):
            continue
        strategy = (n["signal_id"].split("|") + ["", "", ""])[2]  # id 是「幣|週期|策略|時間」
        if strategy in observe:
            continue  # 觀察組（SMC、訂單流…）只記錄在網站上，不推播
        text = n["text"]
        s = signals.get(n["signal_id"])
        if n["kind"] == "NEW" and s:
            tps = " / ".join(f"TP{i} {_fmt(p)}" for i, p in enumerate(s["tps"], 1))
            text += f"\n止損 {_fmt(s['sl'])}\n{tps}"
        out.append(text)
    return out


def send(messages: list[str], token: str, chat_id: str) -> tuple[int, int]:
    """回傳（送出則數, 失敗則數）。"""
    if len(messages) > MAX_MESSAGES:
        extra = len(messages) - MAX_MESSAGES + 1
        messages = messages[:MAX_MESSAGES - 1] + [f"還有 {extra} 則通知，請到網站查看"]
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    failed = 0
    for i, text in enumerate(messages):
        body = urllib.parse.urlencode({"chat_id": chat_id, "text": f"{text}\n{SITE_URL}",
                                       "disable_web_page_preview": "true"}).encode()
        try:
            urllib.request.urlopen(url, body, timeout=20).read()
        except Exception as e:  # 錯誤訊息不含 token（只印類型和狀態碼）
            failed += 1
            print(f"第 {i + 1} 則送出失敗：{type(e).__name__} {getattr(e, 'code', '')}", file=sys.stderr)
        time.sleep(1.1)  # Telegram 對同一個聊天大約每秒 1 則
    return len(messages) - failed, failed


def main(argv: list[str]) -> int:
    if len(argv) == 5 and argv[1] == "outbox":
        msgs = build_outbox(Path(argv[2]), Path(argv[3]))
        Path(argv[4]).write_text(json.dumps(msgs, ensure_ascii=False))
        print(f"{len(msgs)} 則新通知待送出", file=sys.stderr)
        return 0
    if len(argv) == 3 and argv[1] == "send":
        token, chat_id = os.environ.get("TELEGRAM_BOT_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID")
        msgs = _load(Path(argv[2])) or []
        if os.environ.get("TELEGRAM_TEST") == "true":  # 手動執行 workflow 時勾選，確認設定正確
            msgs.insert(0, "測試訊息：Telegram 推播設定成功，之後有新訊號、TP、止損變動會傳到這裡。")
        if not token or not chat_id:
            print("沒有設定 TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID，略過 Telegram", file=sys.stderr)
            return 0
        if not msgs:
            return 0
        sent, failed = send(msgs, token, chat_id)
        print(f"送出 {sent} 則，失敗 {failed} 則", file=sys.stderr)
        return 1 if not sent else 0
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
