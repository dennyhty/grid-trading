/* 交易訊號網頁。資料都來自本機 signal_server.py 的 /api/*，伺服器再去抓派網公開行情。 */
"use strict";

const TZ = "Asia/Taipei";
const TZ_SHIFT = 8 * 3600; // 圖表用 UTC+8 顯示
const INTERVAL_SEC = { "15M": 900, "30M": 1800, "60M": 3600, "2H": 7200, "4H": 14400, "8H": 28800, "12H": 43200, "1D": 86400 };
const LABEL = { "60M": "1H" };
const KIND_TEXT = { TP1: "TP1 觸及", TP2: "TP2 觸及", TP3: "TP3 觸及", STOP_MOVED: "保護止損移動",
  SL: "止損出場", BE_EXIT: "保本出場", PROTECT_EXIT: "保護止損出場", TRAIL_MOVED: "追蹤止損上移", TRAIL_EXIT: "追蹤止損出場", TIMEOUT: "逾時平倉", NEW: "新訊號" };

/** 出場規則的文字。訊號的 exit 沒寫就是舊的預設：TP1/2/3 各平 1/3。 */
function exitRuleText(s) {
  const e = s.exit || {};
  const n = s.tps.length;
  const parts = (e.portions || [1 / 3, 1 / 3, 1 / 3]).slice(0, n);
  const fr = (p) => Math.abs(p - 0.5) < 1e-6 ? "一半" : Math.abs(p - 1 / 3) < 1e-6 ? "1/3" : Math.abs(p - 0.25) < 1e-6 ? "1/4" : `${Math.round(p * 100)}%`;
  const risk = Math.abs(s.price - s.sl);
  const rs = s.tps.map((t) => `${+(Math.abs(t - s.price) / risk).toFixed(2)}R`);
  let text = s.tps.map((_, k) => `TP${k + 1}（${rs[k]}）平${fr(parts[k])}`).join("、") + "；TP1 後止損移到 BE";
  if (n >= 3) text += "，TP2 後移到 TP1";
  if (e.trail) text += `；剩下的倉位用追蹤止損：進場後最${dir(s) > 0 ? "高" : "低"}價往回 ${e.trail_atr || ""} 倍 ATR（${fp(e.trail, s.price)}，約 ${(e.trail / s.price * 100).toFixed(2)}%）`;
  return text;
}

let state = null;
let tab = "OPEN";
let market = (() => { try { return localStorage.getItem("market") || "crypto"; } catch { return "crypto"; } })();
// 週期篩選：ALL 或 state.intervals 裡的一個（"4H"、"2H"）
let ivFilter = (() => { try { return localStorage.getItem("iv") || "ALL"; } catch { return "ALL"; } })();
// 策略 / 觀察組：策略有 tag（例如「SMC觀察組」「訂單流觀察組」）就算觀察組，沒有 tag 的是正式策略
let group = (() => { try { return localStorage.getItem("group") || "strategy"; } catch { return "strategy"; } })();
const isObserve = (name) => !!(state.strategies[name] || {}).tag;
const inGroup = (name) => isObserve(name) === (group === "observe");
const ivOf = (label) => (label === "1H" ? "60M" : label);
/** 週期篩選列。瀏覽器快取了舊版 index.html 時沒有這個元素，就自己補在市場分頁下面，免得整頁的程式停住。 */
/** 策略 / 觀察組 切換列（放在市場分頁下面；舊版 index.html 沒有，所以由程式建立）。 */
function groupTabsEl() {
  let el = document.getElementById("grouptabs");
  if (!el) {
    el = document.createElement("div");
    el.id = "grouptabs";
    el.className = "tabs ivtabs hidden";
    const row = document.querySelector(".mtabs-row");
    if (row) row.after(el); else document.body.prepend(el);
  }
  return el;
}
function ivTabsEl() {
  let el = document.getElementById("ivtabs");
  if (!el) {
    el = document.createElement("div");
    el.id = "ivtabs";
    el.className = "tabs ivtabs hidden";
    const row = document.getElementById("grouptabs") || document.querySelector(".mtabs-row");
    if (row) row.after(el); else document.body.prepend(el);
  }
  return el;
}
const MARKET_LABEL = { crypto: "加密貨幣", stock: "美股", commodity: "商品" };
let current = null;          // 目前打開的訊號
let chart = null, candleSeries = null, emaSeries = {}, priceLines = [];
let ind = null; // RSI / MACD 副圖的 series
let chartData = null;
let live = false;
let notified = null;         // 已經跳過瀏覽器通知的 id
const $ = (id) => document.getElementById(id);

// ---------- 格式 ----------
const pair = (sym) => sym.replace("_USDT_PERP", "/USDT");
const lbl = (iv) => LABEL[iv] || iv;
function fmtTime(ms, withSec = true) {
  const d = new Date(ms);
  const p = new Intl.DateTimeFormat("zh-TW", { timeZone: TZ, year: "numeric", month: "2-digit", day: "2-digit",
    hour: "2-digit", minute: "2-digit", second: withSec ? "2-digit" : undefined, hour12: false }).formatToParts(d);
  const g = (t) => (p.find((x) => x.type === t) || {}).value;
  return `${g("year")}/${g("month")}/${g("day")} ${g("hour")}:${g("minute")}${withSec ? ":" + g("second") : ""}`;
}
function fmtShort(ms) { return fmtTime(ms, false).slice(5); }
// 訊號的 time 已經是訊號 K 線的「收盤」時間（= 確認、推播的時間），直接顯示；
// TP / 止損事件的 time 是事件那根 K 線的「開盤」時間，事件發生在那根 K 線裡某個時刻，顯示成時間區間。
const barMs = (iv) => (INTERVAL_SEC[iv] || 0) * 1000;
const ivOfId = (id) => String(id).split("|")[1];
const closeTime = (ms, iv) => ms + barMs(iv);
function hhmm(ms) { return fmtTime(ms, false).slice(11); }
/** 事件發生在哪根 K 線：「10/13 10:00–12:00」。 */
function barRange(ms, iv) { return `${fmtShort(ms)}–${hhmm(closeTime(ms, iv))}`; }
/** 事件實際被偵測到的時間（台北時間）。網站每 15 分鐘掃一次，所以和真正觸價最多差 15 分鐘左右；
 * 第一次上線補記的舊事件沒有偵測時間，改顯示它發生在哪根 K 線。 */
function eventTime(sid, kind, t) {
  const iv = ivOfId(sid);
  const n = ((state && state.notifications) || []).find((x) => x.id === `${sid}|${kind}|${t}`);
  if (n && !n.initial && n.created) return fmtTime(n.created, false);
  return kind === "NEW" ? `${fmtTime(t, false)} 收盤確認` : `${barRange(t, iv)} 之間`;
}
function decimals(p) {
  if (!p) return 2;
  const a = Math.abs(p);
  return a >= 1000 ? 2 : a >= 10 ? 3 : a >= 1 ? 4 : Math.min(8, Math.ceil(-Math.log10(a)) + 3);
}
const fp = (p, ref) => (p == null ? "—" : Number(p).toFixed(decimals(ref ?? p)));
const pct = (x) => (x == null ? "—" : `${x >= 0 ? "+" : ""}${x.toFixed(2)}%`);
function ago(ms) {
  const m = (Date.now() - ms) / 60000;
  if (m < 60) return `${Math.max(1, Math.round(m))} 分鐘前觸發`;
  if (m < 1440) return `${Math.round(m / 60)} 小時前觸發`;
  return `${Math.round(m / 1440)} 天前觸發`;
}
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

const mk = (s) => s.market || "crypto";
/** 顯示名稱：加密貨幣是 BTC/USDT，美股是 NVDA，商品是 XAU（黃金）。 */
function nameHtml(s, withMarket = false) {
  if (mk(s) === "crypto") return `${pair(s.symbol).split("/")[0]}<small>/USDT</small>`;
  return `${esc(s.name || s.symbol)}${withMarket ? `<small> ${MARKET_LABEL[mk(s)]}</small>` : ""}`;
}
/** 回測數字：加密貨幣存在 stats[策略][週期]，美股/商品存在 stats[策略]["類別|週期"]，
 *  幣安資料的再加「@binance」。沒指定 source 時以幣安為主，沒有才用派網。 */
function statsFor(strategy, mkt, interval, source) {
  const st = state.stats[strategy] || {};
  const key = mkt === "crypto" ? interval : `${mkt}|${interval}`;
  if (source) return st[source === "pionex" ? key : `${key}@${source}`];
  return st[`${key}@binance`] || st[key];
}
const SOURCE_LABEL = { binance: "幣安資料", pionex: "派網資料" };
const period = (s) => `${fmtTime(s.start, false).slice(0, 7)} ~ ${fmtTime(s.end, false).slice(0, 7)}`;
const tested = (s) => !!(statsFor(s.strategy, mk(s), s.interval) || {}).all?.trades;

// ---------- 計算 ----------
function dir(s) { return s.side === "LONG" ? 1 : -1; }
function priceMove(s, price) { return price == null ? null : (dir(s) * (price - s.price) / s.price) * 100; }
/** 1 倍槓桿下的倉位盈虧 %：已平倉部分（扣手續費）+ 剩餘倉位的浮動。 */
function positionPnl(s, price) {
  const t = s.track;
  if (t.status === "CLOSED" || price == null) return t.realized_pct;
  return t.realized_pct + t.remaining * priceMove(s, price);
}

// ---------- 資料 ----------
async function api(path) {
  const r = await fetch(path, { cache: "no-store" });
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).error || r.status);
  return r.json();
}
// 兩種模式：本機 signal_server.py（有 /api），或 GitHub Pages 靜態網站（讀 data/*.json，由排程更新）
let STATIC = null;
let livePrices = {};   // 靜態模式下，瀏覽器直接向 Binance 合約抓的即時價
let livePricesAt = 0;

async function loadState() {
  if (STATIC !== true) {
    try { const st = await api("api/state"); STATIC = false; return st; } catch (e) { if (STATIC === false) throw e; }
  }
  STATIC = true;
  return api(`data/state.json?t=${Date.now()}`);
}

/** 派網行情不允許瀏覽器直接讀，靜態網站改用 Binance U 本位合約的公開即時價（同一個幣價差通常 < 0.1%）。 */
async function loadLivePrices() {
  try {
    const rows = await (await fetch("https://fapi.binance.com/fapi/v1/ticker/price", { cache: "no-store" })).json();
    livePrices = Object.fromEntries(rows.map((r) => [r.symbol, Number(r.price)]));
    livePricesAt = Date.now();
  } catch { /* 抓不到就用快照價 */ }
}

function applyLivePrices() {
  for (const s of state.signals) {
    s.liveSource = "snapshot";
    if (mk(s) !== "crypto") continue; // Binance 大多沒有美股/商品，用掃描時的派網價格
    const p = livePrices[s.symbol.replace("_USDT_PERP", "USDT")];
    // 名稱相同但不是同一個商品（例如代幣化股票）時價差會很大，這種就不用
    if (p && s.live && Math.abs(p / s.live - 1) < 0.05) { s.live = p; s.liveSource = "binance"; }
  }
}

async function refresh() {
  try {
    state = await loadState();
  } catch (e) {
    $("foot").textContent = `無法讀取資料：${e.message}`;
    return;
  }
  if (STATIC) { await loadLivePrices(); applyLivePrices(); }
  handleNotifications();
  renderList();
  if (current) {
    const s = state.signals.find((x) => x.id === current.id);
    if (s) { current = s; renderDetailHead(); renderLevels(); }
  }
}

// ---------- 列表 ----------
const fmtR = (x) => (x == null ? "—" : `${x >= 0 ? "+" : ""}${x.toFixed(3)}R`);
/** 實盤紀錄：網站上出現過的每個訊號照規則跟單的結果。
 * 2026-10-12 RSI 收緊成 54～70 / 30～46 之前，紀錄裡有幾筆舊規則才會進的單（標記為 excluded），不算進來。 */
function liveHtml(name, iv, bt) {
  const ledger = state.ledger || {};
  const g = (ledger.groups || {})[`${name}|${market}|${iv}`];
  if (!g || !g.all) return "";
  const ex = g.excluded || { trades: 0, wins: 0, sum_r: 0, open: 0 };
  const n = g.all.trades - ex.trades, open = g.all.open - ex.open;
  const r = n ? (g.all.sum_r - ex.sum_r) / n : null;
  const win = n ? (g.all.wins - ex.wins) / n : null;
  return `<div class="groups"><h3>實盤紀錄${ledger.since ? `（${fmtTime(ledger.since, false).slice(0, 10)} 起）` : ""}</h3>
    <div class="muted">網站上出現過的每個訊號照規則跟單的結果：已結束 ${n} 筆${n ? `，勝率 ${(win * 100).toFixed(0)}%，平均 ${fmtR(r)}` : ""}；持倉中 ${open} 筆。
    回測每筆約 ${fmtR(bt)}，要累積幾百筆才看得出差別。</div></div>`;
}

/** 「全部週期」時每個週期只列一行摘要，讓下面的持倉清單不會被擠到很下面；點 4H / 2H 看完整數字。 */
function compactStats() {
  const rows = [];
  for (const [name, st] of Object.entries(state.strategies)) {
    if (!inGroup(name)) continue;
    if (st.markets && !st.markets.includes(market)) continue;
    for (const iv of state.intervals.map(ivOf)) {
      if (st.intervals && !st.intervals.map(ivOf).includes(iv)) continue;
      const s = statsFor(name, market, iv);
      const a = (s && s.all) || {};
      const g = ((state.ledger || {}).groups || {})[`${name}|${market}|${iv}`];
      const ex = (g && g.excluded) || { trades: 0, sum_r: 0, open: 0 };
      const n = g && g.all ? g.all.trades - ex.trades : 0;
      const live = g && g.all ? `實盤 ${n} 筆${n ? ` ${fmtR((g.all.sum_r - ex.sum_r) / n)}` : ""}，持倉中 ${g.all.open - ex.open}` : "";
      const bt = a.trades ? `回測 ${fmtR(a.expectancy_r)}，勝率 ${(a.win_rate * 100).toFixed(1)}%，${a.trades} 筆`
        : `還沒有在${MARKET_LABEL[market]}回測過`;
      rows.push(`<div><b>${esc(st.title)} · ${lbl(iv)}</b><div class="muted">${bt}${live ? "；" + live : ""}</div></div>`);
    }
  }
  return `${rows.join("")}<div class="muted">點上方 ${state.intervals.join(" / ")} 看完整回測和風險說明。</div>`;
}

function renderStats() {
  if (ivFilter === "ALL" && state.intervals.length > 1) { $("stats").innerHTML = compactStats(); return; }
  const parts = [];
  for (const [name, st] of Object.entries(state.strategies)) {
    if (!inGroup(name)) continue;
    if (st.markets && !st.markets.includes(market)) continue;
    for (const iv of state.intervals.map(ivOf).filter((x) => ivFilter === "ALL" || x === ivFilter)) {
      if (st.intervals && !st.intervals.map(ivOf).includes(iv)) continue;
      const s = statsFor(name, market, iv);
      if (!s) {
        parts.push(`<h3>${esc(st.title)} · ${lbl(iv)} · ${MARKET_LABEL[market]}</h3>
          <p class="warn">這個策略還沒有在${MARKET_LABEL[market]}上回測過，勝率和期望值未知。下面的訊號只能參考，不建議照著下單。</p>`);
        continue;
      }
      const a = s.all || {};
      if (!a.trades) continue;
      const h1 = s.first_half || {}, h2 = s.second_half || {};
      const ok = a.expectancy_r > 0;
      const src = s.source || "pionex";
      const px = src === "binance" ? statsFor(name, market, iv, "pionex") : null;
      const pa = (px && px.all) || {};
      parts.push(`<h3>${esc(st.title)} · ${lbl(iv)} · ${MARKET_LABEL[market]} 回測（${SOURCE_LABEL[src]}）</h3>
        <div class="muted">${period(s)}，${s.symbols} 個標的，含手續費 ${(s.fee * 100).toFixed(2)}%/次${src === "binance" ? "。訊號和下單都在派網，回測用幣安較長的歷史" : ""}</div>
        <div class="grid">
          <div><span class="muted">勝率</span><b>${(a.win_rate * 100).toFixed(1)}%</b></div>
          <div><span class="muted">期望值 / 筆</span><b class="${ok ? "up" : "down"}">${a.expectancy_r >= 0 ? "+" : ""}${a.expectancy_r.toFixed(3)}R</b></div>
          <div><span class="muted">交易筆數</span><b>${a.trades}</b></div>
          <div><span class="muted">獲利因子</span><b>${a.profit_factor.toFixed(2)}</b></div>
          <div><span class="muted">每筆賠 ${a.risk_per_trade_pct}% 總報酬（複利）</span><b class="${a.total_return_pct >= 0 ? "up" : "down"}">${pct(a.total_return_pct)}</b></div>
          <div><span class="muted">最大回撤</span><b class="down">${a.max_drawdown_pct.toFixed(1)}%</b></div>
        </div>
        <div class="muted">前半段 ${h1.trades || 0} 筆 期望值 ${h1.trades ? h1.expectancy_r.toFixed(3) + "R" : "—"}；後半段 ${h2.trades || 0} 筆 期望值 ${h2.trades ? h2.expectancy_r.toFixed(3) + "R" : "—"}</div>
        ${pa.trades ? `<div class="muted">派網資料（${period(px)}，${px.symbols} 個標的）：${pa.trades} 筆，勝率 ${(pa.win_rate * 100).toFixed(1)}%，期望值 ${pa.expectancy_r >= 0 ? "+" : ""}${pa.expectancy_r.toFixed(3)}R，最大回撤 ${pa.max_drawdown_pct.toFixed(1)}%</div>` : ""}
        ${a.expectancy_r <= 0 ? `<p class="warn">期望值是負的：照這個策略在${MARKET_LABEL[market]}上交易，長期是虧錢的。</p>`
          : a.expectancy_r < 0.1 ? `<p class="warn">期望值接近 0，優勢很小、可能只是運氣。先用小資金或只觀察。</p>` : ""}
        <p class="warn">上面的總報酬和回撤，是每個訊號都照單跟、每筆止損最多賠「當下」總資金 ${a.risk_per_trade_pct}% 算出來的，賺了下一筆就下得大一點，所以是複利，不等於 筆數 × 期望值 × ${a.risk_per_trade_pct}%（每筆固定賠起始資金 ${a.risk_per_trade_pct}%、不複利的話是 ${pct(a.trades * a.expectancy_r * a.risk_per_trade_pct)}）${a.max_concurrent ? `；回測期間最多同時有 ${a.max_concurrent} 筆單在持倉，很多標的會一起出訊號、一起虧，所以回撤已經包含這些一起虧的單` : ""}。每筆賠得越多，回撤就成倍放大，建議每筆止損不超過總資金的 ${a.risk_per_trade_pct}%。</p>
        ${st.tag ? `<p class="warn">${esc(st.tag)}：只記錄實盤結果、不推播，不建議照單下。把同一天、同方向的單算成一次賭注時，2024 年後的優勢接近 0，還不能當成已驗證的策略。</p>` : ""}
        ${liveHtml(name, iv, st.tag ? a.expectancy_r : 0.07)}`);
    }
  }
  $("stats").innerHTML = parts.join("") ||
    `<span class="muted">還沒有回測數字。執行 <code>python signal_backtest.py</code> 後這裡會顯示勝率與期望值。</span>`;
}

function ladderHtml(s, price) {
  const t = s.track, d = dir(s);
  const kinds = new Set(t.events.map((e) => e.kind));
  const rows = s.tps.map((p, k) => [`TP${k + 1}`, p, kinds.has(`TP${k + 1}`), "tp"]).reverse();
  if (t.trailing) rows.unshift(["追蹤止損", t.stop, kinds.has("TRAIL_EXIT"), "be"]);
  rows.push(["BE", t.be, kinds.has("BE_EXIT"), "be"], ["進場", s.price, true, "entry"], ["SL", s.sl, kinds.has("SL"), "sl"]);
  return `<div class="ladder">${rows.map(([name, p, hit, cls]) => {
    const dist = d * (p - s.price) / s.price * 100;
    return `<div class="lv ${cls} ${hit && cls !== "entry" ? "hit" : ""}">
      <span>${name}</span><span class="bar"></span><span class="num">${fp(p, s.price)}</span>
      <span class="pct num">${cls === "entry" ? "" : pct(dist)}</span></div>`;
  }).join("")}</div>`;
}

function timelineHtml(s) {
  const ev = s.track.events;
  if (!ev.length) return "";
  return `<ul class="timeline">${ev.map((e) => {
    const bad = e.kind === "SL";
    return `<li class="${bad ? "bad" : ""}"><div>${esc(e.text)}</div>
      <div class="t num">${eventTime(s.id, e.kind, e.time)} · ${fp(e.price, s.price)} · ${KIND_TEXT[e.kind] || e.kind}</div></li>`;
  }).join("")}</ul>`;
}

function statusText(s) {
  const t = s.track;
  if (t.status === "OPEN") return { cls: "", text: "● 持倉中" };
  const last = t.events[t.events.length - 1];
  return { cls: "closed", text: `已結束 · ${KIND_TEXT[last.kind] || last.kind}` };
}

function tagChip(s) {
  const tag = (state.strategies[s.strategy] || {}).tag;
  return tag ? `<span class="chip tag-chip">${esc(tag)}</span>` : "";
}

function cardHtml(s) {
  const price = s.live;
  const pnl = positionPnl(s, price);
  const mv = priceMove(s, price);
  const st = statusText(s);
  const title = (state.strategies[s.strategy] || {}).title || s.strategy;
  return `<div class="panel card" data-id="${esc(s.id)}">
    <div class="row1"><span class="side ${s.side}">${s.side === "LONG" ? "做多" : "做空"}</span>
      <span class="pair">${nameHtml(s)}</span>
      <span class="chip">${s.label}</span>${tagChip(s)}${tested(s) ? "" : `<span class="chip warn-chip">未回測</span>`}
      <span class="status ${st.cls}">${st.text}</span></div>
    <div class="meta">${esc(title)} · ${ago(s.time)} · 訊號價 ${fp(s.price)}</div>
    <div class="pnl"><div class="big num ${pnl >= 0 ? "up" : "down"}">${pct(pnl)}<small>${s.track.status === "OPEN" ? "未實現盈虧" : "已實現盈虧"}（1x）</small></div>
      <div class="right num">價格 ${pct(mv)}<br>現價 ${fp(price, s.price)}</div></div>
    ${ladderHtml(s, price)}
    ${timelineHtml(s)}
  </div>`;
}

/** 選了單一週期而且清單是空的時候，提醒其他週期還有幾筆。 */
function hiddenNote(inMarket) {
  if (ivFilter === "ALL") return "";
  const n = inMarket.filter((s) => s.interval !== ivFilter && (tab === "OPEN") === (s.track.status === "OPEN")).length;
  return n ? `<br><span class="muted">其他週期還有 ${n} 筆，點上方「全部週期」查看</span>` : "";
}

function renderList() {
  if (ivFilter !== "ALL" && !state.intervals.map(ivOf).includes(ivFilter)) ivFilter = "ALL";
  const hasObserve = Object.keys(state.strategies).some(isObserve);
  if (!hasObserve) group = "strategy";
  const openIn = (g, m) => state.signals.filter((s) => isObserve(s.strategy) === (g === "observe") && mk(s) === m
    && s.track.status === "OPEN").length;
  groupTabsEl().classList.toggle("hidden", !hasObserve);
  groupTabsEl().innerHTML = [["strategy", "策略"], ["observe", "觀察組"]].map(([g, name]) =>
    `<button class="ivtab grouptab ${g === group ? "active" : ""}" data-group="${g}">${name} <span>${openIn(g, market) || ""}</span></button>`).join("");
  renderStats();
  document.querySelectorAll(".mtab").forEach((b) => {
    const m = b.dataset.market;
    b.classList.toggle("active", m === market);
    b.querySelector("span").textContent = openIn(group, m) || "";
  });
  const inMarket = state.signals.filter((s) => mk(s) === market && inGroup(s.strategy));
  const ivs = state.intervals.map(ivOf);
  ivTabsEl().classList.toggle("hidden", ivs.length < 2);
  ivTabsEl().innerHTML = ["ALL", ...ivs].map((iv) => {
    const n = inMarket.filter((s) => (iv === "ALL" || s.interval === iv) && s.track.status === "OPEN").length;
    return `<button class="ivtab ${iv === ivFilter ? "active" : ""}" data-iv="${iv}">${iv === "ALL" ? "全部週期" : lbl(iv)} <span>${n || ""}</span></button>`;
  }).join("");
  const mine = inMarket.filter((s) => ivFilter === "ALL" || s.interval === ivFilter);
  const open = mine.filter((s) => s.track.status === "OPEN");
  const closed = mine.filter((s) => s.track.status !== "OPEN");
  $("n-open").textContent = open.length;
  $("n-closed").textContent = closed.length;
  const shown = tab === "OPEN" ? open : closed;
  $("cards").innerHTML = shown.map(cardHtml).join("") ||
    `<div class="empty">${ivFilter === "ALL" ? "" : lbl(ivFilter) + " "}${tab === "OPEN" ? "目前沒有進行中的訊號" : "最近沒有已結束的訊號"}${
      tab === "OPEN" && closed.length ? `<br><small>最近 ${state.keep_days || 7} 天有 ${closed.length} 筆已經結束，點「已結束」查看</small>` : ""}${hiddenNote(inMarket)}</div>`;
  const counts = state.markets ? Object.entries(state.markets).map(([m, v]) => `${v.label} ${v.symbols}`).join("、") : `${state.symbols} 個`;
  $("foot").textContent = `監看 ${counts} · 週期 ${state.intervals.join("/")} · ` +
    `最後掃描 ${state.updated ? fmtTime(state.updated) : "—"}` +
    (STATIC ? ` · 訊號每 15 分鐘由 GitHub 排程更新（整點收盤後 2 分鐘就掃到新訊號） · ${market === "crypto" && livePricesAt ? "現價來自 Binance 合約即時價" : "現價為掃描時的派網價格"}`
      : " · 每 30 秒更新頁面");
}

// ---------- 單一訊號 ----------
function renderDetailHead() {
  const s = current;
  const price = s.live;
  const i = s.info || {};
  const st = statsFor(s.strategy, mk(s), s.interval);
  const a = st && st.all;
  $("detail-head").innerHTML = `<div class="dh">
    <div class="row1"><span class="pair">${mk(s) === "crypto" ? pair(s.symbol) : nameHtml(s, true)}</span><span class="chip">${s.label}</span>${tagChip(s)}
      <span class="dir ${s.side === "LONG" ? "up" : "down"}">${s.side}</span></div>
    <div class="muted">訊號時間 ${fmtTime(s.time)}（${s.label} K 線收盤確認）· UTC+8</div>
    <div class="prices"><div><span class="muted">訊號價格</span><b class="num">${fp(s.price)}</b></div>
      <div><span class="muted">現價</span><b class="num ${priceMove(s, price) >= 0 ? "up" : "down"}">${fp(price, s.price)}</b></div></div>
    <details><summary>RSI 與策略資訊</summary><div class="kv num">
      <span>策略</span><span>${esc((state.strategies[s.strategy] || {}).title || s.strategy)}</span>
      <span>觸發原因</span><span>${esc(i.reason || "—")}</span>
      ${i.details ? i.details.map(([k, v]) => `<span>${esc(k)}</span><span>${esc(v)}</span>`).join("") + `<span>ATR</span><span>${fp(i.atr, s.price)}</span>`
        : `<span>訊號 RSI</span><span>${i.rsi ?? "—"}</span>
      <span>MACD DIF / DEA</span><span>${fp(i.macd, s.price)} / ${fp(i.macd_signal, s.price)}</span>
      <span>MACD 柱</span><span>${fp(i.hist, s.price)}</span>
      <span>ATR</span><span>${fp(i.atr, s.price)}</span>
      ${i.adx != null ? `<span>ADX(14)</span><span>${i.adx}</span>` : ""}
      <span>EMA144 / 169</span><span>${fp(i.ema144)} / ${fp(i.ema169)}</span>`}
      ${a ? `<span>回測勝率</span><span>${(a.win_rate * 100).toFixed(1)}%（${a.trades} 筆）</span>
      <span>回測期望值</span><span>${a.expectancy_r.toFixed(3)}R / 筆</span>`
        : `<span>回測</span><span>這類標的尚未回測，結果未知</span>`}
      <span>出場規則</span><span>${esc(exitRuleText(s))}</span>
      ${s.track.trailing && s.track.status === "OPEN" ? `<span>目前追蹤止損</span><span>${fp(s.track.stop, s.price)}（派網的止損單要跟著改到這裡）</span>` : ""}
    </div></details>
    <div class="muted">現價更新 ${fmtTime(Date.now())} · 資訊每 30 秒更新</div></div>`;
}

function renderLevels() {
  const s = current;
  const pnl = positionPnl(s, s.live);
  $("detail-levels").innerHTML = `<div class="pnl"><div class="big num ${pnl >= 0 ? "up" : "down"}">${pct(pnl)}
      <small>${s.track.status === "OPEN" ? "未實現" : "已實現"}（1x 含費）</small></div>
      <div class="right">${statusText(s).text}</div></div>
    ${ladderHtml(s, s.live)}${timelineHtml(s)}`;
}

function openDetail(id) {
  const s = state && state.signals.find((x) => x.id === id);
  if (!s) return;
  current = s;
  if (chart) setLive(false);
  $("list-view").classList.add("hidden");
  $("detail-view").classList.remove("hidden");
  $("back").classList.remove("hidden");
  $("subtitle").textContent = "訊號圖表";
  const sel = $("interval");
  const ivs = [s.interval, ...["15M", "60M", "4H", "1D"].filter((x) => x !== s.interval)];
  sel.innerHTML = ivs.map((iv, k) => `<option value="${iv}">${lbl(iv)}${k === 0 ? " · 原訊號" : ""}</option>`).join("");
  renderDetailHead();
  renderLevels();
  loadChart(s.interval, true);
  window.scrollTo(0, 0);
}

function closeDetail() {
  current = null;
  setLive(false);
  $("chart-panel").classList.remove("fs");
  $("detail-view").classList.add("hidden");
  $("list-view").classList.remove("hidden");
  $("back").classList.add("hidden");
  $("subtitle").textContent = "訊號列表";
  if (location.hash) history.replaceState(null, "", location.pathname);
}

// ---------- 圖表 ----------
function ensureChart() {
  if (chart) return;
  const el = $("chart");
  chart = LightweightCharts.createChart(el, {
    autoSize: true,
    layout: { background: { color: "#171a21" }, textColor: "#8a91a0" },
    grid: { vertLines: { color: "#1f2430" }, horzLines: { color: "#1f2430" } },
    rightPriceScale: { borderColor: "#262b36" },
    timeScale: { borderColor: "#262b36", timeVisible: true, secondsVisible: false },
    crosshair: { mode: 0 },
  });
  candleSeries = chart.addCandlestickSeries({
    upColor: "#22c58b", downColor: "#f0697a", borderVisible: false, wickUpColor: "#22c58b", wickDownColor: "#f0697a",
  });
  // RSI 與 MACD 各一張小圖，放在 K 線圖下方，左右捲動和縮放跟著 K 線圖同步
  const sub = (id) => LightweightCharts.createChart($(id), {
    autoSize: true,
    layout: { background: { color: "#171a21" }, textColor: "#8a91a0" },
    grid: { vertLines: { color: "#1f2430" }, horzLines: { color: "#1f2430" } },
    rightPriceScale: { borderColor: "#262b36", minimumWidth: 80, scaleMargins: { top: 0.12, bottom: 0.08 } },
    timeScale: { borderColor: "#262b36", visible: false },
    crosshair: { mode: 0 },
    handleScroll: false, handleScale: false,
  });
  chart.priceScale("right").applyOptions({ minimumWidth: 80 });
  if (!$("panes")) {
    $("chart").insertAdjacentHTML("afterend", `<div id="panes" class="panes">
      <div class="pane-tag">RSI(14)</div><div id="rsi-chart" class="sub-chart" style="height:130px"></div>
      <div class="pane-tag">MACD(12,26,9)</div><div id="macd-chart" class="sub-chart" style="height:120px"></div></div>`);
  }
  const rsiChart = sub("rsi-chart"), macdChart = sub("macd-chart");
  const line = (c, color, w = 1) => c.addLineSeries({ color, lineWidth: w, priceLineVisible: false,
    lastValueVisible: true, crosshairMarkerVisible: false, priceFormat: { type: "price", precision: 2, minMove: 0.01 } });
  ind = {
    charts: [rsiChart, macdChart],
    rsi: rsiChart.addLineSeries({ color: "#b07cff", lineWidth: 1.5, priceLineVisible: false, crosshairMarkerVisible: false,
      priceFormat: { type: "price", precision: 1, minMove: 0.1 },
      autoscaleInfoProvider: () => ({ priceRange: { minValue: 15, maxValue: 85 } }) }),
    hist: macdChart.addHistogramSeries({ priceLineVisible: false, lastValueVisible: false }),
    dif: line(macdChart, "#4fb3d9"),
    dea: line(macdChart, "#e6c35c"),
  };
  // vegas_trend 的 RSI 區間：做多 54–70、做空 30–46
  for (const [v, c] of [[70, "#f0697a"], [54, "#22c58b"], [46, "#f0697a"], [30, "#22c58b"]]) {
    ind.rsi.createPriceLine({ price: v, color: c, lineWidth: 1, lineStyle: 2, axisLabelVisible: true, title: "" });
  }
  chart.timeScale().subscribeVisibleLogicalRangeChange((r) => {
    if (r) for (const c of ind.charts) c.timeScale().setVisibleLogicalRange(r);
  });
  layoutPanes();
}

/** 「顯示 RSI / MACD」開關。舊版 index.html 快取裡沒有這些元素時不做事。 */
function layoutPanes() {
  const opt = $("opt-ind"), panes = $("panes");
  if (panes) panes.classList.toggle("hidden", !(opt ? opt.checked : true));
  if (ind && chart) {
    const r = chart.timeScale().getVisibleLogicalRange();
    if (r) for (const c of ind.charts) c.timeScale().setVisibleLogicalRange(r);
  }
}

/** Wilder RSI(14)，和後端 grid_trading/indicators.py 同算法。 */
function calcRsi(closes, n = 14) {
  const out = new Array(closes.length).fill(null);
  let g = 0, l = 0;
  for (let k = 1; k < closes.length; k++) {
    const d = closes[k] - closes[k - 1];
    const up = Math.max(d, 0), dn = Math.max(-d, 0);
    if (k === 1) { g = up; l = dn; } else { g += (up - g) / n; l += (dn - l) / n; }
    if (k >= n) out[k] = l === 0 ? 100 : 100 - 100 / (1 + g / l);
  }
  return out;
}

function calcEma(xs, n) {
  const a = 2 / (n + 1), out = [];
  xs.forEach((x, k) => out.push(k === 0 ? x : out[k - 1] + a * (x - out[k - 1])));
  return out;
}

/** MACD(12,26,9)：DIF、DEA、柱狀體。 */
function calcMacd(closes) {
  const f = calcEma(closes, 12), sl = calcEma(closes, 26);
  const dif = closes.map((_, k) => f[k] - sl[k]);
  const dea = calcEma(dif, 9);
  return { dif, dea, hist: dif.map((v, k) => v - dea[k]) };
}

function drawIndicators(candles) {
  const closes = candles.map((c) => c.close);
  const t = candles.map((c) => c.time + TZ_SHIFT);
  const r = calcRsi(closes), m = calcMacd(closes);
  const skip = 30; // 前幾根還在暖機，不畫
  // 每張圖都放滿同樣的 K 線時間（暖機段只放時間不放值），三張圖的邏輯位置才會一一對齊
  const pts = (arr) => arr.map((v, k) => (k < skip || v == null ? { time: t[k] } : { time: t[k], value: v }));
  ind.rsi.setData(pts(r));
  ind.dif.setData(pts(m.dif));
  ind.dea.setData(pts(m.dea));
  ind.hist.setData(m.hist.map((v, k) => ({ time: t[k], value: v,
    color: v >= 0 ? (v >= (m.hist[k - 1] ?? 0) ? "#22c58b" : "#1d7a5a") : (v <= (m.hist[k - 1] ?? 0) ? "#f0697a" : "#93404b") }))
    .map((p, k) => (k < skip ? { time: p.time } : p)));
  const r0 = chart.timeScale().getVisibleLogicalRange();
  if (r0) for (const c of ind.charts) c.timeScale().setVisibleLogicalRange(r0);
}

const EMA_COLORS = ["#e6c35c", "#4fb3d9", "#6f8cff", "#b07cff", "#d27cc0", "#9aa3b2"];

async function loadChart(interval, focusSignal) {
  ensureChart();
  $("interval-note").textContent = interval === current.interval ? "原訊號週期" : `原訊號為 ${current.label}`;
  let data;
  try {
    data = STATIC
      ? await api(`data/chart/${current.symbol}_${interval}.json?t=${state.updated}`)
      : await api(`api/chart?symbol=${encodeURIComponent(current.symbol)}&interval=${interval}&strategy=${current.strategy}`);
  } catch (e) {
    $("kinfo").textContent = `圖表載入失敗：${e.message}`;
    return;
  }
  chartData = data;
  const prec = decimals(current.price);
  candleSeries.applyOptions({ priceFormat: { type: "price", precision: prec, minMove: Math.pow(10, -prec) } });
  candleSeries.setData(data.candles.map((c) => ({ ...c, time: c.time + TZ_SHIFT })));
  const lastC = data.candles[data.candles.length - 1];
  if (STATIC && lastC && current.liveSource === "binance") {
    // 快照之後的價格變化：用即時價更新最後一根 K 線
    const p = current.live;
    candleSeries.update({ ...lastC, time: lastC.time + TZ_SHIFT, close: p, high: Math.max(lastC.high, p), low: Math.min(lastC.low, p) });
  }
  drawIndicators(STATIC && lastC && current.liveSource === "binance"
    ? [...data.candles.slice(0, -1), { ...lastC, close: current.live }] : data.candles);

  for (const s of Object.values(emaSeries)) chart.removeSeries(s);
  emaSeries = {};
  const showEmas = $("opt-emas").checked;
  Object.entries(data.emas).forEach(([name, pts], k) => {
    const s = chart.addLineSeries({ color: EMA_COLORS[k % EMA_COLORS.length], lineWidth: 1, priceLineVisible: false,
      lastValueVisible: false, crosshairMarkerVisible: false, visible: showEmas });
    s.setData(pts.map((p) => ({ time: p.time + TZ_SHIFT, value: p.value })));
    emaSeries[name] = s;
  });
  $("legend").innerHTML = Object.keys(data.emas).map((n, k) =>
    `<span style="--c:${EMA_COLORS[k % EMA_COLORS.length]}">${n}</span>`).join("");

  // 訊號標記：放在包含訊號收盤時間（前一秒）的那根 K 線
  const sigSec = current.time / 1000 - 1;
  let idx = -1;
  for (let k = 0; k < data.candles.length; k++) if (data.candles[k].time <= sigSec) idx = k;
  if (idx >= 0) {
    const long = current.side === "LONG";
    candleSeries.setMarkers([{ time: data.candles[idx].time + TZ_SHIFT, position: long ? "belowBar" : "aboveBar",
      color: long ? "#22c58b" : "#f0697a", shape: long ? "arrowUp" : "arrowDown",
      text: `${current.side} ${fp(current.price)}` }]);
  } else {
    candleSeries.setMarkers([]);
  }
  drawLevels();

  const l = data.last;
  if (l && l.time) {
    $("kinfo").innerHTML = `訊號 RSI ${current.info.rsi ?? "—"} · RSI・MACD<br>最新 K 棒 · ${fmtTime(l.time)}　開 ${fp(l.open)}　高 ${fp(l.high)}　低 ${fp(l.low)}　收 ${fp(l.close)}　RSI ${l.rsi.toFixed(2)}　MACD 柱 ${fp(l.hist, l.close)}`;
  }
  if (focusSignal) focusOnSignal(idx);
  else if (live) chart.timeScale().scrollToRealTime();
}

function drawLevels() {
  for (const pl of priceLines) candleSeries.removePriceLine(pl);
  priceLines = [];
  if (!$("opt-levels").checked) return;
  const s = current;
  const lines = [["進場", s.price, "#e6e8ee", 0], ["SL", s.sl, "#f0697a", 2],
    ...s.tps.map((p, k) => [`TP${k + 1}`, p, "#22c58b", 2])];
  if (s.track.stop !== s.sl) lines.push([s.track.trailing ? "追蹤止損" : "保護止損", s.track.stop, "#d9b45a", 1]);
  for (const [title, price, color, style] of lines) {
    priceLines.push(candleSeries.createPriceLine({ price, color, lineWidth: 1, lineStyle: style, axisLabelVisible: true, title }));
  }
}

function focusOnSignal(idx) {
  if (idx < 0) return chart.timeScale().fitContent();
  chart.timeScale().setVisibleLogicalRange({ from: idx - 90, to: idx + 30 });
}

let liveTimer = null;
function setLive(on) {
  live = on;
  $("btn-live").classList.toggle("on", on);
  $("live-dot").classList.toggle("on", on);
  $("live-hint").textContent = on ? "每 30 秒更新 K 線" : "點「最新價格」開啟即時更新";
  clearInterval(liveTimer);
  if (on && chart) {
    chart.timeScale().scrollToRealTime();
    liveTimer = setInterval(() => current && loadChart($("interval").value, false), 30000);
  }
}

// ---------- 通知 ----------
function lastSeen() { try { return Number(localStorage.getItem("lastSeenNote") || 0); } catch { return 0; } }
function setLastSeen(v) { try { localStorage.setItem("lastSeenNote", String(v)); } catch { /* 私密模式 */ } }

const isObserveNote = (n) => isObserve(String(n.signal_id).split("|")[2]);

function handleNotifications() {
  const notes = state.notifications;
  const seen = lastSeen();
  // 觀察組（SMC、訂單流…）不推播，也不算進鈴鐺的未讀數
  const unread = notes.filter((n) => !n.initial && !isObserveNote(n) && n.created > seen).length;
  $("badge").textContent = unread;
  $("badge").classList.toggle("hidden", unread === 0);
  if (notified === null) { notified = new Set(notes.map((n) => n.id)); return; }
  for (const n of notes) {
    if (notified.has(n.id)) continue;
    notified.add(n.id);
    // 觀察組（SMC、訂單流…）只記錄在通知中心，不跳出通知
    if (!n.initial && !isObserveNote(n) && "Notification" in window && Notification.permission === "granted") {
      try { new Notification(KIND_TEXT[n.kind] || "訊號", { body: n.text, tag: n.id }); } catch { /* iOS 非安裝模式不支援 */ }
    }
  }
  if (!$("notif").classList.contains("hidden")) renderNotes();
}

function renderNotes() {
  const seen = lastSeen();
  $("notif-list").innerHTML = state.notifications.map((n) => `
    <div class="note ${!n.initial && n.created > seen ? "new" : ""}" data-id="${esc(n.signal_id)}">
      <div>${esc(n.text)}</div><div class="t num">${eventTime(n.signal_id, n.kind, n.time)} · ${KIND_TEXT[n.kind] || n.kind}</div></div>`).join("") ||
    `<div class="empty">還沒有通知</div>`;
  $("notif-list").insertAdjacentHTML("afterbegin", `<div class="muted" style="padding:4px 0 8px">時間是台北時間（UTC+8），網站偵測到事件的時間；每 15 分鐘掃一次（整點後 2 分鐘起），止盈止損和實際觸價最多差 15 分鐘左右；新訊號在 K 線收盤後約 2 分鐘就會出現。</div>`);
  const perm = "Notification" in window ? Notification.permission : "unsupported";
  $("notif-perm").classList.toggle("hidden", perm === "granted" || perm === "unsupported");
}

function openNotes() {
  renderNotes();
  $("notif").classList.remove("hidden");
  setLastSeen(Date.now());
  $("badge").classList.add("hidden");
}

// ---------- 事件 ----------
$("cards").addEventListener("click", (e) => {
  const c = e.target.closest(".card");
  if (c) { location.hash = "s=" + encodeURIComponent(c.dataset.id); }
});
document.querySelectorAll(".mtab").forEach((b) => b.addEventListener("click", () => {
  market = b.dataset.market;
  try { localStorage.setItem("market", market); } catch { /* 私密模式 */ }
  renderList();
}));
document.querySelectorAll(".tab").forEach((b) => b.addEventListener("click", () => {
  tab = b.dataset.tab;
  document.querySelectorAll(".tab").forEach((x) => x.classList.toggle("active", x === b));
  renderList();
}));
groupTabsEl().addEventListener("click", (e) => {
  const b = e.target.closest(".grouptab");
  if (!b) return;
  group = b.dataset.group;
  try { localStorage.setItem("group", group); } catch { /* 私密模式 */ }
  renderList();
});
ivTabsEl().addEventListener("click", (e) => {
  const b = e.target.closest(".ivtab");
  if (!b) return;
  ivFilter = b.dataset.iv;
  try { localStorage.setItem("iv", ivFilter); } catch { /* 私密模式 */ }
  renderList();
});
$("back").addEventListener("click", closeDetail);
$("bell").addEventListener("click", openNotes);
$("notif-close").addEventListener("click", () => $("notif").classList.add("hidden"));
$("notif-list").addEventListener("click", (e) => {
  const n = e.target.closest(".note");
  if (!n) return;
  $("notif").classList.add("hidden");
  location.hash = "s=" + encodeURIComponent(n.dataset.id);
});
$("notif-perm").addEventListener("click", async () => {
  if ("Notification" in window) await Notification.requestPermission();
  renderNotes();
});
$("interval").addEventListener("change", (e) => loadChart(e.target.value, true));
$("btn-signal").addEventListener("click", () => {
  setLive(false);
  const sel = $("interval");
  if (sel.value !== current.interval) { sel.value = current.interval; loadChart(current.interval, true); }
  else {
    const sigSec = current.time / 1000 - 1;
    let idx = -1;
    chartData.candles.forEach((c, k) => { if (c.time <= sigSec) idx = k; });
    focusOnSignal(idx);
  }
});
$("btn-live").addEventListener("click", () => setLive(!live));
$("btn-full").addEventListener("click", () => {
  const el = $("chart-panel");
  if (document.fullscreenElement) document.exitFullscreen();
  else if (el.requestFullscreen) el.requestFullscreen();
  else el.classList.toggle("fs");
});
$("btn-more").addEventListener("click", () => $("more").classList.toggle("hidden"));
$("opt-levels").addEventListener("change", drawLevels);
$("opt-ind")?.addEventListener("change", () => chart && layoutPanes());
$("opt-emas").addEventListener("change", (e) => Object.values(emaSeries).forEach((s) => s.applyOptions({ visible: e.target.checked })));

function route() {
  const m = location.hash.match(/^#s=(.+)$/);
  if (m && state) openDetail(decodeURIComponent(m[1]));
  else if (!m && current) closeDetail();
}
window.addEventListener("hashchange", route);

(async function init() {
  await refresh();
  route();
  setInterval(refresh, 30000);
})();
