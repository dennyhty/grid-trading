/* 「績效」和「策略研究」兩個分頁。訊號分頁在 app.js，這裡只負責切換分頁和這兩頁的內容。
 * 績效：幣安自動交易的實際紀錄（data/trader.json，VPS 每 15 分鐘經由網站更新帶過來）
 *       + 維加斯實盤規則回測（perf/vegas_backtest.json，research/site_performance.py 產生）。
 * 研究：perf/research.json 的報告索引，點進去讀 reports/*.md 原文。 */
"use strict";

(function () {
  const $id = (id) => document.getElementById(id);
  const h = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const num = (x, d = 2) => (x == null || !isFinite(x) ? "—" : Number(x).toFixed(d));
  const sgn = (x, d = 2, unit = "") => (x == null || !isFinite(x) ? "—" : `${x >= 0 ? "+" : ""}${Number(x).toFixed(d)}${unit}`);
  const pctOf = (x, d = 1) => (x == null ? "—" : `${(x * 100).toFixed(d)}%`);
  const cls = (x) => (x == null ? "" : x > 0 ? "up" : x < 0 ? "down" : "");
  const tw = (ms) => new Intl.DateTimeFormat("zh-TW", { timeZone: "Asia/Taipei", month: "2-digit", day: "2-digit",
    hour: "2-digit", minute: "2-digit", hour12: false }).format(new Date(ms));
  const twDate = (ms) => new Intl.DateTimeFormat("sv-SE", { timeZone: "Asia/Taipei" }).format(new Date(ms));
  const coin = (sym) => String(sym).replace(/USDT$/, "");
  const sideTag = (s) => `<span class="side ${s}">${s === "LONG" ? "做多" : "做空"}</span>`;
  const store = (k, v) => { try { if (v === undefined) return localStorage.getItem(k); localStorage.setItem(k, v); } catch { return null; } };

  const MODE_LABEL = { testnet: "測試網", paper: "紙上交易", live: "實盤" };
  const STATUS_CLASS = { live: "st-live", observe: "st-observe", keep: "st-keep", pending: "st-pending", rejected: "st-rejected", info: "st-info" };

  let page = "signals";
  let bt = null, trader = null, research = null;
  let book = store("perf-book") || "4H+2H";
  let curveMode = store("perf-curve") || "simple";
  let mode = store("perf-mode") || null;
  let charts = [];

  async function getJson(path) {
    const r = await fetch(`${path}${path.includes("?") ? "&" : "?"}t=${Date.now()}`, { cache: "no-store" });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    return r.json();
  }

  // ---------- 分頁 ----------
  function route() {
    const hs = location.hash;
    const next = hs.startsWith("#perf") ? "perf" : hs.startsWith("#research") ? "research" : "signals";
    page = next;
    document.body.classList.toggle("page-other", next !== "signals");
    $id("perf-view").classList.toggle("hidden", next !== "perf");
    $id("research-view").classList.toggle("hidden", next !== "research");
    document.querySelectorAll("#nav a").forEach((a) => a.classList.toggle("active", a.dataset.page === next));
    if (next === "perf") { $id("subtitle").textContent = "績效"; renderPerf(); }
    else if (next === "research") {
      const m = hs.match(/^#research\/(.+)$/);
      $id("subtitle").textContent = "策略研究";
      if (m) renderReport(decodeURIComponent(m[1])); else renderResearch();
    } else if (!document.getElementById("detail-view").classList.contains("hidden")) {
      $id("subtitle").textContent = "訊號圖表";
    } else {
      $id("subtitle").textContent = "訊號列表";
    }
  }

  // ---------- 圖表工具 ----------
  function dropCharts() { charts.forEach((c) => c.remove()); charts = []; }
  function lineChart(el, opts = {}) {
    const c = LightweightCharts.createChart(el, {
      height: el.clientHeight || opts.height || 240,
      layout: { background: { color: "#171a21" }, textColor: "#8a91a0", fontSize: 11 },
      grid: { vertLines: { color: "#1e222b" }, horzLines: { color: "#1e222b" } },
      rightPriceScale: { borderColor: "#262b36", mode: opts.log ? LightweightCharts.PriceScaleMode.Logarithmic : 0 },
      timeScale: { borderColor: "#262b36", minBarSpacing: 0.01 },
      crosshair: { mode: 0 },
      handleScroll: { vertTouchDrag: false }, handleScale: { axisPressedMouseMove: { price: false } },
      localization: { locale: "zh-TW", priceFormatter: opts.fmt || ((v) => v.toFixed(1)) },
    });
    charts.push(c);
    return c;
  }
  const fit = (c) => c.timeScale().fitContent();
  window.addEventListener("resize", () => charts.forEach((c) => {
    const el = c.chartElement().parentElement;
    if (el) c.applyOptions({ width: el.clientWidth });
  }));

  function tiles(list) {
    return `<div class="kpis">${list.map(([k, v, c, sub]) =>
      `<div><span>${h(k)}</span><b class="num ${c || ""}">${v}</b>${sub ? `<i>${sub}</i>` : ""}</div>`).join("")}</div>`;
  }

  // ---------- 績效 ----------
  async function renderPerf() {
    const el = $id("perf-view");
    if (!bt || !trader) {
      el.innerHTML = `<div class="empty">讀取中…</div>`;
      const [b, t] = await Promise.allSettled([getJson("perf/vegas_backtest.json"), getJson("data/trader.json")]);
      bt = b.status === "fulfilled" ? b.value : { error: b.reason?.message };
      trader = t.status === "fulfilled" ? t.value : { books: {} };
      if (page !== "perf") return;
    }
    dropCharts();
    el.innerHTML = `
      <div class="seg" id="perf-sec">
        <a href="#perf" class="${location.hash === "#perf/bt" ? "" : "on"}">幣安自動交易</a>
        <a href="#perf/bt" class="${location.hash === "#perf/bt" ? "on" : ""}">維加斯回測</a>
      </div>
      <div id="perf-body"></div>`;
    if (location.hash === "#perf/bt") renderBacktest($id("perf-body"));
    else renderTrader($id("perf-body"));
  }

  // 幣安自動交易
  function traderStats(ledger) {
    const r = ledger.map((x) => x.r || 0);
    const pnl = ledger.map((x) => x.pnl || 0);
    const wins = r.filter((x) => x > 0), losses = r.filter((x) => x <= 0);
    let cum = 0, peak = 0, dd = 0;
    for (const x of r) { cum += x; peak = Math.max(peak, cum); dd = Math.min(dd, cum - peak); }
    const sl = losses.reduce((a, b) => a + b, 0);
    return {
      n: r.length, win: r.length ? wins.length / r.length : null, avg: r.length ? cum / r.length : null, total: cum,
      pnl: pnl.reduce((a, b) => a + b, 0), pf: sl < 0 ? wins.reduce((a, b) => a + b, 0) / -sl : null, dd,
    };
  }

  function renderTrader(el) {
    const books = trader.books || {};
    const modes = ["live", "testnet", "paper"].filter((m) => books[m]);
    if (!modes.length) {
      el.innerHTML = `<div class="panel">
        <h3>還沒有自動交易紀錄</h3>
        <p class="muted">VPS 上的自動交易程式（binance-trader@testnet / @paper）把紀錄存在 VPS 上，網站要經由
        <code>deploy/trader_feed.py</code> 才拿得到。照 <code>docs/website.md</code> 在 VPS 開好 feed、把網址存成
        GitHub secret <code>TRADER_FEED_URL</code> 後，下一次網站更新（每 15 分鐘）這裡就會出現實際成交、持倉和資金曲線。</p>
      </div>`;
      return;
    }
    if (!modes.includes(mode)) mode = modes[0];
    const b = books[mode];
    const ledger = [...(b.ledger || [])].sort((x, y) => x.closed - y.closed);
    const st = traderStats(ledger);
    const open = b.positions || [];
    const openR = open.reduce((a, p) => a + (p.r_est || 0), 0);
    const ref = bt && bt.books ? bt.books["4H+2H"].metrics : null;
    const lastScan = Math.max(0, ...Object.values(b.processed || {}));
    const stale = Date.now() - (b.updated || 0) > 3 * 3600e3;
    el.innerHTML = `
      <div class="seg small">${modes.map((m) => `<button data-mode="${m}" class="${m === mode ? "on" : ""}">${MODE_LABEL[m]}</button>`).join("")}</div>
      <div class="panel">
        <div class="row-between"><h3>幣安${MODE_LABEL[mode]}</h3>
          <span class="chip ${stale ? "warn-chip" : ""}">${stale ? "紀錄超過 3 小時沒更新" : "運作中"}${b.paused ? "・已暫停開新倉" : ""}</span></div>
        <p class="muted">紀錄更新 ${b.updated ? tw(b.updated) : "—"}・最後掃描的 K 線收盤 ${lastScan ? tw(lastScan) : "—"}（台北時間）
        ${mode === "paper" ? "・紙上交易用訊號收盤價成交、照回測規則出場，不是真的下單" : mode === "testnet" ? "・測試網是假資金，成交價和流動性跟正式站不完全一樣" : ""}</p>
        ${tiles([
          ["已平倉", `${st.n} 筆`],
          ["勝率", pctOf(st.win), "", ref ? `回測 ${pctOf(ref.win_rate, 0)}` : ""],
          ["平均每筆", sgn(st.avg, 3, "R"), cls(st.avg), ref ? `回測 ${sgn(ref.avg_r, 3, "R")}` : ""],
          ["累計", sgn(st.total, 2, "R"), cls(st.total)],
          ["已實現損益", sgn(st.pnl, 2, " U"), cls(st.pnl)],
          ["獲利因子", num(st.pf), "", ref ? `回測 ${num(ref.profit_factor)}` : ""],
          ["最大回撤", sgn(st.dd, 2, "R"), cls(st.dd)],
          ["持倉", `${open.length} 張`],
          ["持倉浮動（估）", sgn(openR, 2, "R"), cls(openR)],
        ])}
        ${st.n < 30 ? `<p class="warn">只有 ${st.n} 筆，勝率和平均 R 的誤差很大（100 筆時平均 R 的誤差約 ±0.15R），先看有沒有照規則執行，不要急著和回測比高低。</p>` : ""}
      </div>
      <div class="panel"><h3>累計 R（已平倉）</h3><div id="tr-curve" class="chart-box"></div>
        ${ledger.length ? "" : `<p class="muted">還沒有平倉的單。</p>`}</div>
      <div class="panel"><h3>目前持倉</h3>${openTable(open)}</div>
      <div class="panel"><h3>最近平倉</h3>${closedTable(ledger.slice(-30).reverse())}</div>`;
    el.querySelectorAll("[data-mode]").forEach((x) => x.addEventListener("click", () => { mode = x.dataset.mode; store("perf-mode", mode); renderPerf(); }));
    if (ledger.length) {
      const c = lineChart($id("tr-curve"), { height: 200, fmt: (v) => v.toFixed(2) });
      let cum = 0;
      const pts = [];
      for (const x of ledger) {
        cum += x.r || 0;
        const t = Math.floor(x.closed / 1000);
        if (pts.length && pts[pts.length - 1].time >= t) pts[pts.length - 1].value = cum; else pts.push({ time: t, value: cum });
      }
      c.addBaselineSeries({ baseValue: { type: "price", price: 0 }, lineWidth: 2, priceLineVisible: false,
        topLineColor: "#22c58b", topFillColor1: "rgba(34,197,139,.25)", topFillColor2: "rgba(34,197,139,0)",
        bottomLineColor: "#f0697a", bottomFillColor1: "rgba(240,105,122,0)", bottomFillColor2: "rgba(240,105,122,.25)" }).setData(pts);
      c.applyOptions({ timeScale: { timeVisible: true } });
      fit(c);
    }
  }

  function openTable(open) {
    if (!open.length) return `<p class="muted">沒有持倉。</p>`;
    const rows = [...open].sort((a, b) => (b.opened || 0) - (a.opened || 0)).map((p) => `<tr>
      <td><b>${h(coin(p.symbol))}</b> <span class="chip">${h(p.interval)}</span></td><td>${sideTag(p.side)}</td>
      <td class="num">${num(p.entry, 6).replace(/\.?0+$/, "")}</td><td class="num">${num(p.stop, 6).replace(/\.?0+$/, "")}</td>
      <td>${p.tps_hit ? `TP${p.tps_hit}` : "—"}</td>
      <td class="num ${cls(p.r_est)}">${sgn(p.r_est, 2, "R")}</td><td class="muted">${p.opened ? tw(p.opened) : ""}</td></tr>`).join("");
    return `<div class="tbl"><table><thead><tr><th>幣</th><th>方向</th><th>進場</th><th>止損</th><th>已到</th><th>浮動</th><th>開倉</th></tr></thead>
      <tbody>${rows}</tbody></table></div>`;
  }

  function closedTable(list) {
    if (!list.length) return `<p class="muted">還沒有平倉的單。</p>`;
    const rows = list.map((x) => `<tr><td><b>${h(coin(x.symbol))}</b> <span class="chip">${h(x.interval)}</span></td><td>${sideTag(x.side)}</td>
      <td class="num ${cls(x.r)}">${sgn(x.r, 2, "R")}</td><td class="num ${cls(x.pnl)}">${sgn(x.pnl, 3)}</td>
      <td class="muted">${x.closed ? tw(x.closed) : ""}</td></tr>`).join("");
    return `<div class="tbl"><table><thead><tr><th>幣</th><th>方向</th><th>R</th><th>USDT</th><th>平倉</th></tr></thead>
      <tbody>${rows}</tbody></table></div>`;
  }

  // 維加斯回測
  function renderBacktest(el) {
    if (!bt || bt.error || !bt.books) { el.innerHTML = `<div class="panel">回測資料讀不到：${h(bt && bt.error)}</div>`; return; }
    const B = bt.books[book] || bt.books["4H+2H"];
    const m = B.metrics;
    el.innerHTML = `
      <div class="panel">
        <div class="row-between"><h3>維加斯順勢 <span class="chip tag-chip">實盤規則回測</span></h3>
          <div class="seg small">${Object.keys(bt.books).map((k) => `<button data-book="${k}" class="${k === book ? "on" : ""}">${k === "4H+2H" ? "4H+2H 實盤" : k}</button>`).join("")}</div></div>
        <p class="muted">${m.start} ~ ${m.end}，${m.coins} 個幣，幣安合約（上市前接現貨）。每筆風險 ${(bt.risk * 100).toFixed(2)}%，
          含手續費每邊 0.05%，<b>沒算滑價和資金費率</b>。這是用現在的規則回頭算，不是實際成交。</p>
        ${tiles([
          ["年化（複利 CAGR）", sgn(m.cagr_pct, 1, "%"), cls(m.cagr_pct)],
          ["年化（單利）", sgn(m.simple_per_year_pct, 1, "%"), cls(m.simple_per_year_pct), "每筆都用起始資金的 0.25%"],
          ["最大回撤", `${num(m.max_dd_pct, 1)}%`, "down", `已平倉；${num(m.max_dd_r, 0)}R`],
          ["Sharpe", num(m.sharpe), "", "日報酬 × √365"],
          ["Sortino", num(m.sortino)],
          ["Calmar", num(m.calmar), "", "CAGR ÷ 最大回撤"],
          ["勝率", pctOf(m.win_rate)],
          ["平均每筆", sgn(m.avg_r, 3, "R"), cls(m.avg_r), `t ${num(m.t_cluster, 1)}（同日同向算一群）`],
          ["獲利因子", num(m.profit_factor)],
          ["平均賺 / 賠", `${num(m.avg_win_r, 2)} / ${num(-m.avg_loss_r, 2)}`, "", "R，賺的單 / 賠的單"],
          ["交易筆數", `${m.trades}`, "", `每年約 ${num(m.trades_per_year, 0)} 筆`],
          ["最長連虧", `${m.max_losing_streak} 筆`, "", `最久回撤 ${m.longest_dd_days} 天`],
        ])}
      </div>
      <div class="panel">
        <div class="row-between"><h3>資金曲線</h3>
          <div class="seg small"><button data-curve="simple" class="${curveMode === "simple" ? "on" : ""}">單利 %</button>
          <button data-curve="compound" class="${curveMode === "compound" ? "on" : ""}">複利（對數）</button></div></div>
        <div id="bt-curve" class="chart-box tall"></div>
        <div class="legend-row"><span style="--c:#d9b45a">維加斯 ${h(book)}</span>${curveMode === "compound" ? `<span style="--c:#6b7385">BTC 持有</span>` : ""}</div>
        <h3 class="mt">回撤（複利資金，已平倉）</h3>
        <div id="bt-dd" class="chart-box short"></div>
        <p class="muted">這裡只看已平倉的單。把未平倉的浮動損益也算進去，回撤會再深幾個百分點（舊的「同幣只做一張」規則下，最深一次已平倉 −35.7%、含浮動約 −39%）。</p>
      </div>
      <div class="panel">
        <h3>滾動期望值（最近 500 筆的平均 R）</h3>
        <div id="bt-roll" class="chart-box"></div>
        <p class="muted">低於 0 代表最近 500 筆（約 5 個月）是虧的。線越往下，優勢越弱；2026 年明顯變弱。</p>
      </div>
      <div class="panel"><h3>各期間</h3>${recentTable(B.recent)}</div>
      <div class="panel"><h3>逐年</h3>${yearTable(B.yearly)}</div>
      <div class="panel"><h3>每月報酬（單利 %）</h3>${monthGrid(B.monthly)}</div>
      <div class="panel"><h3>多空、週期拆開</h3>${splitTable(B)}</div>
      <div class="panel"><h3>每筆 R 的分布</h3>${histSvg(B.histogram)}
        <p class="muted">−1R 附近是止損，0 附近是保本出場，右邊的長尾是追蹤止損抓到的大行情。勝率主要由出場規則決定。</p></div>
      <div class="panel caveat"><h3>要打折看的地方</h3><ul>
        <li>沒算滑價和資金費率；每邊多 0.05% 滑價每筆約少 0.04R，2026 年的優勢會接近 0。</li>
        <li>同一天很多幣同方向出訊號，其實是同一個賭注；t 值已經用「同日同向算一群」算過。</li>
        <li>規則是看過 2020 ~ 2026 的資料後定的（RSI 範圍、出場、ADX），全期數字偏樂觀；2024 年後和最近一年比較接近實際。</li>
        <li>2017 ~ 2019 年用現貨補，當時能交易的幣只有 10 ~ 20 個。</li>
      </ul></div>`;
    el.querySelectorAll("[data-book]").forEach((x) => x.addEventListener("click", () => { book = x.dataset.book; store("perf-book", book); renderPerf(); }));
    el.querySelectorAll("[data-curve]").forEach((x) => x.addEventListener("click", () => { curveMode = x.dataset.curve; store("perf-curve", curveMode); renderPerf(); }));
    drawBacktestCharts(B);
  }

  function drawBacktestCharts(B) {
    const dates = bt.dates;
    const series = (arr, f = (v) => v) => {
      const out = [];
      arr.forEach((v, i) => { if (v != null) out.push({ time: dates[i], value: f(v) }); });
      return out;
    };
    const log = curveMode === "compound";
    const c1 = lineChart($id("bt-curve"), { height: 300, log, fmt: log ? (v) => `${v >= 10 ? v.toFixed(0) : v.toFixed(2)}x` : (v) => `${v.toFixed(0)}%` });
    if (log) {
      c1.addLineSeries({ color: "#6b7385", lineWidth: 1, priceLineVisible: false, lastValueVisible: false })
        .setData(series(bt.btc_pct, (v) => 1 + v / 100));
      c1.addLineSeries({ color: "#d9b45a", lineWidth: 2, priceLineVisible: false }).setData(series(B.curve.compound_pct, (v) => 1 + v / 100));
    } else {
      c1.addLineSeries({ color: "#d9b45a", lineWidth: 2, priceLineVisible: false }).setData(series(B.curve.simple_pct));
    }
    fit(c1);
    const c2 = lineChart($id("bt-dd"), { height: 150, fmt: (v) => `${v.toFixed(0)}%` });
    c2.addBaselineSeries({ baseValue: { type: "price", price: 0 }, lineWidth: 1, priceLineVisible: false,
      bottomLineColor: "#f0697a", bottomFillColor1: "rgba(240,105,122,.05)", bottomFillColor2: "rgba(240,105,122,.4)",
      topLineColor: "#f0697a", topFillColor1: "rgba(0,0,0,0)", topFillColor2: "rgba(0,0,0,0)" }).setData(series(B.curve.dd_pct));
    fit(c2);
    const c3 = lineChart($id("bt-roll"), { height: 170, fmt: (v) => `${v.toFixed(2)}R` });
    const roll = B.rolling_avg_r.map(([t, v]) => ({ time: t, value: v }));
    const dedup = roll.filter((p, i) => i === roll.length - 1 || roll[i + 1].time !== p.time);
    c3.addBaselineSeries({ baseValue: { type: "price", price: 0 }, lineWidth: 2, priceLineVisible: false,
      topLineColor: "#22c58b", topFillColor1: "rgba(34,197,139,.2)", topFillColor2: "rgba(34,197,139,0)",
      bottomLineColor: "#f0697a", bottomFillColor1: "rgba(240,105,122,0)", bottomFillColor2: "rgba(240,105,122,.25)" }).setData(dedup);
    fit(c3);
  }

  function recentTable(rows) {
    return `<div class="tbl"><table><thead><tr><th>期間</th><th>筆數</th><th>勝率</th><th>每筆</th><th>單利/年</th><th>回撤</th><th>Sharpe</th><th>t</th></tr></thead><tbody>
      ${rows.map((r) => `<tr><td>${h(r.name)}</td><td class="num">${r.trades}</td><td class="num">${pctOf(r.win_rate, 0)}</td>
        <td class="num ${cls(r.avg_r)}">${sgn(r.avg_r, 3)}</td><td class="num ${cls(r.simple_per_year_pct)}">${sgn(r.simple_per_year_pct, 1, "%")}</td>
        <td class="num">${num(r.max_dd_pct, 1)}%</td><td class="num">${num(r.sharpe)}</td><td class="num">${num(r.t_cluster, 1)}</td></tr>`).join("")}
      </tbody></table></div><p class="muted">t 值大於 2 才算統計上分得出和 0 不一樣；「最近 6 個月」「今年」樣本少，只能看方向。年化用該期間天數換算。</p>`;
  }

  function yearTable(rows) {
    const max = Math.max(...rows.map((r) => Math.abs(r.simple_pct || 0)), 1);
    return `<div class="tbl"><table><thead><tr><th>年</th><th>筆數</th><th>勝率</th><th>每筆</th><th>單利</th><th></th><th>複利</th><th>當年回撤</th><th>BTC</th></tr></thead><tbody>
      ${rows.map((r) => `<tr><td>${r.year}</td><td class="num">${r.trades}</td><td class="num">${pctOf(r.win_rate, 0)}</td>
        <td class="num ${cls(r.avg_r)}">${sgn(r.avg_r, 3)}</td><td class="num ${cls(r.simple_pct)}">${sgn(r.simple_pct, 1, "%")}</td>
        <td class="barcell"><i class="${r.simple_pct >= 0 ? "pos" : "neg"}" style="width:${Math.abs(r.simple_pct) / max * 100}%"></i></td>
        <td class="num ${cls(r.compound_pct)}">${sgn(r.compound_pct, 1, "%")}</td><td class="num">${num(r.max_dd_pct, 1)}%</td>
        <td class="num muted">${sgn(r.btc_pct, 0, "%")}</td></tr>`).join("")}
      </tbody></table></div><p class="muted">損益記在出場那年。第一年和今年都不滿 12 個月。</p>`;
  }

  function monthGrid(rows) {
    const byYear = {};
    for (const [ym, v] of rows) { const [y, mo] = ym.split("-"); (byYear[y] = byYear[y] || {})[+mo] = v; }
    const color = (v) => {
      if (v == null) return "";
      const a = Math.min(1, Math.abs(v) / 15);
      return v >= 0 ? `background:rgba(34,197,139,${0.12 + a * 0.6})` : `background:rgba(240,105,122,${0.12 + a * 0.6})`;
    };
    const years = Object.keys(byYear).sort();
    return `<div class="tbl"><table class="heat"><thead><tr><th></th>${[...Array(12)].map((_, i) => `<th>${i + 1}</th>`).join("")}<th>全年</th></tr></thead><tbody>
      ${years.map((y) => {
        const tot = Object.values(byYear[y]).reduce((a, b) => a + b, 0);
        return `<tr><td>${y}</td>${[...Array(12)].map((_, i) => { const v = byYear[y][i + 1];
          return `<td class="num" style="${color(v)}">${v == null ? "" : (Math.round(v) || 0)}</td>`; }).join("")}
          <td class="num ${cls(tot)}"><b>${sgn(tot, 0)}</b></td></tr>`; }).join("")}
      </tbody></table></div>`;
  }

  function splitTable(B) {
    const m = B.metrics;
    const rows = [["做多", m.long], ["做空", m.short]];
    if (B.by_interval) for (const [k, v] of Object.entries(B.by_interval)) rows.push([`${k} 訊號`, v]);
    return `<div class="tbl"><table><thead><tr><th></th><th>筆數</th><th>勝率</th><th>每筆</th><th>合計</th></tr></thead><tbody>
      ${rows.map(([k, v]) => `<tr><td>${k}</td><td class="num">${v.trades}</td><td class="num">${pctOf(v.win_rate, 0)}</td>
        <td class="num ${cls(v.avg_r)}">${sgn(v.avg_r, 3)}</td><td class="num ${cls(v.total_r)}">${sgn(v.total_r, 0, "R")}</td></tr>`).join("")}
      </tbody></table></div>`;
  }

  function histSvg(bins) {
    const W = 640, H = 160, pad = 18;
    const max = Math.max(...bins.map((b) => b[1]), 1);
    const bw = (W - pad) / bins.length;
    const bars = bins.map(([x, n], i) => {
      const bh = (n / max) * (H - 30);
      const color = x < -0.01 ? "#f0697a" : x < 0.24 ? "#8a91a0" : "#22c58b";
      return `<rect x="${pad + i * bw + 1}" y="${H - 18 - bh}" width="${bw - 2}" height="${bh}" fill="${color}" opacity=".8"><title>${x}R ~ ${(x + 0.25).toFixed(2)}R：${n} 筆</title></rect>`;
    }).join("");
    const labels = bins.map(([x], i) => (Math.abs(x % 1) < 1e-6 ? `<text x="${pad + i * bw}" y="${H - 4}" class="ax">${x}R</text>` : "")).join("");
    return `<svg viewBox="0 0 ${W} ${H}" class="hist" role="img" aria-label="每筆 R 分布">${bars}${labels}</svg>`;
  }

  // ---------- 策略研究 ----------
  let filter = "all", query = "";
  async function loadResearch() {
    if (!research) research = await getJson("perf/research.json");
    return research;
  }

  async function renderResearch() {
    dropCharts();
    const el = $id("research-view");
    let R;
    try { R = await loadResearch(); } catch (e) { el.innerHTML = `<div class="panel">報告索引讀不到：${h(e.message)}</div>`; return; }
    const all = R.groups.flatMap((g) => g.items);
    const count = (s) => all.filter((x) => s === "all" || x.status === s).length;
    const stats = ["all", "live", "observe", "pending", "keep", "rejected", "info"].filter((s) => s === "all" || count(s));
    el.innerHTML = `
      <div class="panel">
        <h3>策略研究</h3>
        <p class="muted">每個想法都用幣安 2020 年起（部分 2017 年起）的資料回測，前後兩段互相驗證，再用派網資料複核。
        上線條件：兩段時期都賺、同日同向合併後 t ≥ 2、贏過隨機進場。共 ${all.length} 份報告。</p>
        <div class="funnel">${["live", "observe", "pending", "rejected"].map((s) => `<div class="${STATUS_CLASS[s]}"><b>${count(s)}</b><span>${R.status[s]}</span></div>`).join("")}</div>
        <input id="rs-q" class="search" placeholder="搜尋報告（例如 止損、SMC、回撤）" value="${h(query)}">
        <div class="chips">${stats.map((s) => `<button data-f="${s}" class="${filter === s ? "on" : ""}">${s === "all" ? "全部" : R.status[s]} ${count(s)}</button>`).join("")}</div>
      </div>
      <div id="rs-list"></div>`;
    const draw = () => {
      const q = query.trim().toLowerCase();
      const html = R.groups.map((g) => {
        const items = g.items.filter((x) => (filter === "all" || x.status === filter) &&
          (!q || `${x.title} ${x.summary}`.toLowerCase().includes(q)));
        if (!items.length) return "";
        return `<h4 class="grp">${h(g.name)}</h4>${items.map((x) => `
          <a class="panel rcard" href="#research/${encodeURIComponent(x.file)}">
            <div class="row-between"><b>${h(x.title)}</b><span class="st ${STATUS_CLASS[x.status]}">${h(R.status[x.status])}</span></div>
            <p>${h(x.summary)}</p><span class="muted">${h(x.date)}</span></a>`).join("")}`;
      }).join("");
      $id("rs-list").innerHTML = html || `<div class="empty">沒有符合的報告</div>`;
    };
    draw();
    $id("rs-q").addEventListener("input", (e) => { query = e.target.value; draw(); });
    el.querySelectorAll("[data-f]").forEach((b) => b.addEventListener("click", () => {
      filter = b.dataset.f; el.querySelectorAll("[data-f]").forEach((x) => x.classList.toggle("on", x === b)); draw();
    }));
  }

  async function renderReport(file) {
    dropCharts();
    const el = $id("research-view");
    if (!/^[\w\-/]+\.md$/.test(file) || file.includes("..")) { el.innerHTML = `<div class="panel">找不到這份報告</div>`; return; }
    el.innerHTML = `<div class="empty">讀取中…</div>`;
    let meta = null;
    try { meta = (await loadResearch()).groups.flatMap((g) => g.items).find((x) => x.file === file); } catch { /* 沒有索引也能讀全文 */ }
    let text;
    try {
      const r = await fetch(`reports/${file}`, { cache: "no-store" });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      text = await r.text();
    } catch (e) { el.innerHTML = `<div class="panel"><a href="#research">‹ 所有報告</a><p>報告讀不到：${h(e.message)}</p></div>`; return; }
    const dir = file.includes("/") ? file.slice(0, file.lastIndexOf("/") + 1) : "";
    el.innerHTML = `<div class="panel report">
      <div class="row-between"><a href="#research">‹ 所有報告</a>${meta ? `<span class="st ${STATUS_CLASS[meta.status]}">${h(research.status[meta.status])}</span>` : ""}</div>
      <article class="md">${markdown(text, dir)}</article></div>`;
    window.scrollTo(0, 0);
  }

  // 精簡版 Markdown：標題、段落、清單（含縮排）、表格、程式碼、引用、粗體、連結
  function inline(s, dir) {
    return h(s)
      .replace(/`([^`]+)`/g, "<code>$1</code>")
      .replace(/\*\*([^*]+)\*\*/g, "<b>$1</b>")
      .replace(/(^|[^*])\*([^*\s][^*]*)\*/g, "$1<i>$2</i>")
      .replace(/\[([^\]]+)\]\(([^)\s]+)\)/g, (_, t, u) => {
        const url = u.replace(/&amp;/g, "&");
        if (/^https?:\/\//.test(url)) return `<a href="${h(url)}" target="_blank" rel="noopener">${t}</a>`;
        if (/\.md(#.*)?$/.test(url)) {
          const parts = (dir + url.replace(/#.*$/, "")).split("/");
          const out = [];
          for (const p of parts) { if (p === "..") out.pop(); else if (p && p !== ".") out.push(p); }
          return `<a href="#research/${encodeURIComponent(out.join("/"))}">${t}</a>`;
        }
        return t;
      });
  }

  function markdown(src, dir) {
    const lines = src.replace(/\r/g, "").split("\n");
    const out = [];
    let i = 0;
    const isTable = (k) => /^\s*\|/.test(lines[k] || "") && /^\s*\|?\s*:?-{2,}/.test(lines[k + 1] || "");
    const cells = (l) => l.trim().replace(/^\|/, "").replace(/\|$/, "").split("|").map((c) => c.trim());
    while (i < lines.length) {
      const l = lines[i];
      if (/^```/.test(l)) {
        const buf = [];
        for (i++; i < lines.length && !/^```/.test(lines[i]); i++) buf.push(lines[i]);
        i++;
        out.push(`<pre><code>${h(buf.join("\n"))}</code></pre>`);
      } else if (/^#{1,6}\s/.test(l)) {
        const n = l.match(/^#+/)[0].length;
        out.push(`<h${n}>${inline(l.replace(/^#+\s*/, ""), dir)}</h${n}>`);
        i++;
      } else if (isTable(i)) {
        const head = cells(l);
        i += 2;
        const body = [];
        for (; i < lines.length && /^\s*\|/.test(lines[i]); i++) body.push(cells(lines[i]));
        out.push(`<div class="tbl"><table><thead><tr>${head.map((c) => `<th>${inline(c, dir)}</th>`).join("")}</tr></thead><tbody>
          ${body.map((r) => `<tr>${r.map((c) => `<td>${inline(c, dir)}</td>`).join("")}</tr>`).join("")}</tbody></table></div>`);
      } else if (/^\s*([-*+]|\d+\.)\s/.test(l)) {
        // 清單：用縮排決定層級
        const stack = [];
        let html = "";
        for (; i < lines.length && (/^\s*([-*+]|\d+\.)\s/.test(lines[i]) || (/^\s{2,}\S/.test(lines[i]) && stack.length)); i++) {
          const li = lines[i];
          const m = li.match(/^(\s*)([-*+]|\d+\.)\s+(.*)$/);
          if (!m) { html += " " + inline(li.trim(), dir); continue; }
          const ind = m[1].length, tag = /\d/.test(m[2]) ? "ol" : "ul";
          while (stack.length && ind < stack[stack.length - 1].ind) html += `</li></${stack.pop().tag}>`;
          if (!stack.length || ind > stack[stack.length - 1].ind) { html += `<${tag}>`; stack.push({ ind, tag }); }
          else html += "</li>";
          html += `<li>${inline(m[3], dir)}`;
        }
        while (stack.length) html += `</li></${stack.pop().tag}>`;
        out.push(html);
      } else if (/^>\s?/.test(l)) {
        const buf = [];
        for (; i < lines.length && /^>\s?/.test(lines[i]); i++) buf.push(lines[i].replace(/^>\s?/, ""));
        out.push(`<blockquote>${inline(buf.join(" "), dir)}</blockquote>`);
      } else if (/^\s*(-{3,}|\*{3,})\s*$/.test(l)) {
        out.push("<hr>");
        i++;
      } else if (!l.trim()) {
        i++;
      } else {
        const buf = [];
        for (; i < lines.length && lines[i].trim() && !/^(#{1,6}\s|```|>|\s*([-*+]|\d+\.)\s)/.test(lines[i]) && !isTable(i); i++) buf.push(lines[i]);
        out.push(`<p>${buf.map((x) => inline(x, dir)).join("<br>")}</p>`);
      }
    }
    return out.join("\n");
  }

  window.addEventListener("hashchange", route);
  // 自動交易紀錄每 5 分鐘重讀一次（網站本身每 15 分鐘更新）
  setInterval(async () => {
    try { trader = await getJson("data/trader.json"); if (page === "perf" && location.hash !== "#perf/bt") renderPerf(); } catch { /* 下次再試 */ }
  }, 300000);
  route();
})();
