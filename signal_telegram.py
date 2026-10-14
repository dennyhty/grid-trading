"""把新的訊號通知推到 Telegram。GitHub Actions 每次掃描後執行，分兩步：

  python signal_telegram.py outbox site/data previous_notifications.json outbox.json [early_sent.json]
      比對這次和上一次的通知，把新的寫進 outbox.json（網站部署前）。early_sent.json 是掃描時
      已經搶先送出的通知（signal_snapshot.py --early-push），不再送。
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


def build_outbox(data_dir: Path, previous_path: Path, skip: set[str] | None = None) -> list[str]:
    """這次新增、而且不是第一次啟動補上的通知，轉成 Telegram 訊息文字。觀察組（策略有 tag）的通知不送。
    skip：收盤後搶先送過的通知 id（signal_snapshot.py --early-push），不再送一次。"""
    previous = _load(previous_path)
    if not previous:
        return []  # 拿不到上一次的紀錄就不送，免得把舊通知全部重送一遍
    seen = {n["id"] for n in previous} | (skip or set())
    notes = _load(data_dir / "notifications.json") or []
    state = _load(data_dir / "state.json") or {}
    return [m for _, m in messages(notes, seen, state.get("signals", []), state.get("strategies") or {})]


def messages(notes: list[dict], seen: set[str], signals: list[dict], strategies: dict) -> list[tuple[str, str]]:
    """（通知 id, 訊息文字）：seen 裡沒有、不是第一次啟動補上的、不是觀察組的通知。"""
    signals = {s["id"]: s for s in signals}
    observe = {name for name, st in strategies.items() if st.get("tag")}
    out = []
    for n in notes:
        if n["id"] in seen or n.get("initial"):
            continue
        strategy = (n["signal_id"].split("|") + ["", "", ""])[2]  # id 是「幣|週期|策略|時間」
        if n.get("observe") or strategy in observe:
            continue  # 觀察組（SMC、訂單流、美股/商品的維加斯…）只記錄在網站上，不推播
        text = n["text"]
        s = signals.get(n["signal_id"])
        if n["kind"] == "NEW" and s:
            tps = " / ".join(f"TP{i} {_fmt(p)}" for i, p in enumerate(s["tps"], 1))
            text += f"\n止損 {_fmt(s['sl'])}\n{tps}"
        out.append((n["id"], text))
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
        for attempt in range(3):  # 這一輪沒送到，下一輪也不會再送，所以失敗時等一下重試
            try:
                urllib.request.urlopen(url, body, timeout=20).read()
                break
            except Exception as e:  # 錯誤訊息不含 token（只印類型和狀態碼）
                code = getattr(e, "code", "")
                print(f"第 {i + 1} 則送出失敗：{type(e).__name__} {code}", file=sys.stderr)
                if attempt == 2 or (isinstance(code, int) and 400 <= code < 500 and code != 429):
                    failed += 1  # 400/401/403 之類重試也沒用
                    break
                time.sleep(_retry_after(e) if code == 429 else 3)
        time.sleep(1.1)  # Telegram 對同一個聊天大約每秒 1 則（群組每分鐘 20 則，超過會回 429）
    return len(messages) - failed, failed


def _retry_after(e: Exception) -> float:
    """Telegram 回 429 時，回應裡的 parameters.retry_after 是要等幾秒。"""
    try:
        return min(60.0, float(json.loads(e.read())["parameters"]["retry_after"]) + 1)
    except Exception:
        return 5.0


def main(argv: list[str]) -> int:
    if len(argv) in (5, 6) and argv[1] == "outbox":
        skip = set(_load(Path(argv[5])) or []) if len(argv) == 6 else set()
        msgs = build_outbox(Path(argv[2]), Path(argv[3]), skip)
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
