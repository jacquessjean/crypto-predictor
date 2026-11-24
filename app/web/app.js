"use strict";

/* Crypto Predictor web client.
 * Talks only to the application API (same origin, /api/...). No build step, no dependencies.
 * Everything coming from the server is inserted with textContent, never innerHTML. */

const API = "/api";
const FEE_PCT = 0.2;                 // round-trip spot taker fee, in percent
const BAR_MS = 15 * 60 * 1000;       // 15-minute bars
const CHART_LIMIT = 96;              // 24 h of 15-minute candles
const WEAK_SIGNAL_NOTE =
  "The model’s predictive power is weak, so most of the time the honest answer is Wait. " +
  "This is not financial advice.";

const state = {
  pair: null,
  busy: false,
  last: null,        // most recent prediction shown on the Predict tab
  candles: [],
  watch: null,       // position watch (re-check after a Buy)
  tab: "predict",
  chartToken: 0,
};

/* ---------- small helpers ---------- */

const $ = (id) => document.getElementById(id);
const SVG_NS = "http://www.w3.org/2000/svg";

function el(tag, props = {}, ...kids) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(props)) {
    if (value === false || value == null) continue;
    if (key === "class") node.className = value;
    else if (key === "text") node.textContent = value;
    else if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
    else node.setAttribute(key, value === true ? "" : value);
  }
  for (const kid of kids.flat()) if (kid != null) node.append(kid);
  return node;
}

function svg(tag, attrs = {}, ...kids) {
  const node = document.createElementNS(SVG_NS, tag);
  for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
  for (const kid of kids.flat()) if (kid != null) node.append(kid);
  return node;
}

const fmtTime = (t) => new Date(t).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
const fmtDateTime = (t) =>
  new Date(t).toLocaleString([], { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
const fmtPct = (p) => `${(p * 100).toFixed(1)}%`;
const fmtRet = (r) => (r == null ? "—" : `${r >= 0 ? "+" : ""}${(r * 100).toFixed(2)}%`);
const retClass = (r) => (r == null || r === 0 ? "" : r > 0 ? "pos" : "neg");

function fmtPrice(v) {
  if (v == null) return "—";
  const digits = v >= 1000 ? 2 : v >= 1 ? 4 : 6;
  return v.toLocaleString(undefined, { minimumFractionDigits: digits, maximumFractionDigits: digits });
}

function minutesLeft(ms) {
  const mins = Math.round(ms / 60000);
  return mins >= 60 ? `${Math.floor(mins / 60)} h ${mins % 60} min` : `${mins} min`;
}

/* ---------- API ---------- */

class ApiError extends Error {
  constructor(message, status) {
    super(message);
    this.status = status;
  }
}

async function api(path, options = {}, timeoutMs = 60000) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  let response;
  try {
    response = await fetch(API + path, {
      ...options,
      signal: controller.signal,
      headers: { "Content-Type": "application/json" },
    });
  } catch (err) {
    throw new ApiError(
      err.name === "AbortError" ? "The server took too long to answer." : "Can’t reach the application server.",
      0
    );
  } finally {
    clearTimeout(timer);
  }
  if (!response.ok) {
    let detail = "";
    try {
      const body = await response.json();
      detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch (_) { /* body was not JSON */ }
    throw new ApiError(detail || response.statusText, response.status);
  }
  return response.json();
}

const postPredict = (pair) => api("/predict", { method: "POST", body: JSON.stringify({ pair }) });

function friendlyError(err) {
  if (err.status === 400) return { title: "That pair isn’t supported.", detail: err.message };
  if (err.status === 503)
    return {
      title: "No prediction right now.",
      detail: `${err.message} Market data or the model service may be temporarily unavailable. Please try again in a moment.`,
    };
  if (err.status === 0) return { title: err.message, detail: "Check that the server is running, then try again." };
  return { title: "Something went wrong.", detail: err.message };
}

/* ---------- chart ---------- */

function drawChart(candles, marker) {
  const host = $("chart");
  host.replaceChildren();
  if (!candles.length) return;

  const W = Math.max(320, Math.min(900, host.clientWidth || 800));   // 1 SVG unit = 1 CSS pixel
  const H = W < 520 ? 260 : 320;
  const padL = 8, padR = 70, padT = 10, padB = 24;
  const lo = Math.min(...candles.map((c) => c.low));
  const hi = Math.max(...candles.map((c) => c.high));
  const span = hi - lo || hi * 0.001 || 1;
  const min = lo - span * 0.06, max = hi + span * 0.06;
  const step = (W - padL - padR) / candles.length;
  const x = (i) => padL + (i + 0.5) * step;
  const y = (v) => padT + ((max - v) / (max - min)) * (H - padT - padB);

  const root = svg("svg", { viewBox: `0 0 ${W} ${H}`, preserveAspectRatio: "xMidYMid meet" });

  const lastClose = candles[candles.length - 1].close;
  for (let k = 0; k <= 4; k++) {                       // horizontal grid + price labels
    const value = min + ((max - min) * k) / 4;
    root.append(svg("line", { class: "grid-line", x1: padL, x2: W - padR, y1: y(value), y2: y(value) }));
    if (Math.abs(y(value) - y(lastClose)) > 14)        // don't collide with the last-price tag
      root.append(Object.assign(svg("text", { x: W - padR + 6, y: y(value) + 4 }), { textContent: fmtPrice(value) }));
  }

  const maxLabels = Math.max(2, Math.floor((W - padL - padR) / 80));   // time labels, ~80 px apart
  const labelEvery = Math.ceil(candles.length / maxLabels);
  candles.forEach((c, i) => {
    if (i % labelEvery === 0 && x(i) > padL + 24)      // skip a label that would be clipped at the left edge
      root.append(Object.assign(svg("text", { x: x(i), y: H - 6, "text-anchor": "middle" }), { textContent: fmtTime(c.open_time) }));
  });

  if (marker) {                                        // scored bar + entry price
    const idx = candles.findIndex((c) => c.open_time === marker.barOpen);
    if (idx >= 0) {
      root.append(svg("rect", { class: "marker-band", x: x(idx) - step / 2, y: padT, width: step, height: H - padT - padB }));
      root.append(
        svg("line", { class: "entry-line", x1: padL, x2: W - padR, y1: y(marker.price), y2: y(marker.price) }),
        Object.assign(svg("text", { class: "entry-tag", x: padL + 4, y: y(marker.price) - 4 }), { textContent: "entry" })
      );
    }
  }

  const bodyW = Math.max(2, step * 0.66);
  candles.forEach((c, i) => {
    const dir = c.close >= c.open ? "up" : "down";
    const title = svg("title");
    title.textContent =
      `${fmtDateTime(c.open_time)}\nO ${fmtPrice(c.open)}  H ${fmtPrice(c.high)}\nL ${fmtPrice(c.low)}  C ${fmtPrice(c.close)}`;
    root.append(
      svg("g", { class: dir },
        svg("line", { class: "wick", x1: x(i), x2: x(i), y1: y(c.high), y2: y(c.low) }),
        svg("rect", {
          x: x(i) - bodyW / 2, y: y(Math.max(c.open, c.close)),
          width: bodyW, height: Math.max(1, Math.abs(y(c.open) - y(c.close))),
        }),
        title
      )
    );
  });

  root.append(                                          // last price line
    svg("line", { class: "last-line", x1: padL, x2: W - padR, y1: y(lastClose), y2: y(lastClose) }),
    Object.assign(svg("text", { class: "last-tag", x: W - padR + 6, y: y(lastClose) - 5 }), { textContent: fmtPrice(lastClose) })
  );
  host.append(root);
}

function redrawChart() {
  const scored = state.last && state.last.pair === state.pair ? state.last : null;
  drawChart(state.candles, scored && { barOpen: Date.parse(scored.bar_time), price: scored.entry_price });
}

async function refreshChart() {
  const pair = state.pair;
  const token = ++state.chartToken;                     // ignore out-of-order responses
  $("chart-title").textContent = `${pair} · 15-minute candles, last 24 h`;
  try {
    const candles = await api(`/candles?pair=${encodeURIComponent(pair)}&interval=15m&limit=${CHART_LIMIT}`);
    if (token !== state.chartToken) return;
    state.candles = candles;
    $("chart-msg").hidden = true;
    const last = candles[candles.length - 1];
    $("chart-last").textContent = last ? `last ${fmtPrice(last.close)}` : "";
    redrawChart();
  } catch (err) {
    if (token !== state.chartToken) return;
    $("chart").replaceChildren();
    $("chart-last").textContent = "";
    $("chart-msg").hidden = false;
    $("chart-msg").textContent = `Couldn’t load the chart: ${friendlyError(err).detail}`;
  }
}

/* ---------- result card (idle → loading → result / error) ---------- */

const EXPLAIN = {
  BUY: `The price is expected to rise by more than the ${FEE_PCT}% round-trip fee over the next hour.`,
  SELL: `The price is expected to fall by more than the ${FEE_PCT}% fee over the next hour. If you hold this coin, consider selling; if you don’t, don’t buy.`,
  WAIT: `No clear edge after the ${FEE_PCT}% fee. Waiting is the honest answer here.`,
};

const noteNode = () => el("p", { class: "note", text: WEAK_SIGNAL_NOTE });

function setResult(...nodes) {
  clearInterval(setResult.ticker);
  $("result").replaceChildren(...nodes.filter(Boolean), noteNode());   // replaceChildren(null) would print "null"
}

function renderIdle() {
  setResult(el("h2", { text: "Prediction" }), el("p", { class: "muted", text: "Pick a pair and press Predict to get advice for the next hour." }));
}

function renderLoading(pair) {
  setResult(
    el("h2", { text: "Prediction" }),
    el("div", { class: "loading" },
      el("div", { class: "spinner", "aria-hidden": "true" }),
      el("div", {}, el("strong", { text: `Analysing ${pair}…` }),
        el("div", { class: "muted small", text: "Fetching the latest market data and asking the model. This can take a few seconds." })))
  );
}

function renderError(err, retry) {
  const { title, detail } = friendlyError(err);
  setResult(
    el("h2", { text: "Prediction" }),
    el("div", { class: "error-box", role: "alert" }, el("strong", { text: title }), el("span", { text: detail })),
    el("button", { class: "primary", onclick: retry, text: "Try again" })
  );
}

function probRow(label, cls, value) {
  return el("div", { class: "prob" },
    el("span", { text: label }),
    el("div", { class: "track" }, el("div", { class: `fill ${cls}`, style: `width:${(value * 100).toFixed(1)}%` })),
    el("span", { class: "pct", text: fmtPct(value) })
  );
}

function renderPrediction(pred) {
  const expiresAt = Date.parse(pred.expires_at);
  const barClose = Date.parse(pred.bar_time) + BAR_MS;
  const expiryNode = el("dd", {});
  const updateExpiry = () => {
    const left = expiresAt - Date.now();
    expiryNode.textContent = left > 0 ? `${fmtTime(expiresAt)} (in ${minutesLeft(left)})` : `${fmtTime(expiresAt)} (expired)`;
  };
  updateExpiry();

  const watchable = pred.action === "BUY" && expiresAt > Date.now();
  const alreadyWatching = state.watch && state.watch.status === "active" && state.watch.pair === pred.pair;

  setResult(
    el("div", { class: "head-line" }, el("span", { text: pred.pair }), el("span", { text: `data up to ${fmtTime(barClose)}` })),
    el("div", { class: `action ${pred.action}`, text: pred.action }),
    el("p", { class: "explain", text: EXPLAIN[pred.action] }),
    el("div", { class: "probs" },
      probRow(`Up (> +${FEE_PCT}%)`, "up", pred.p_up),
      probRow(`Flat (±${FEE_PCT}%)`, "flat", pred.p_flat),
      probRow(`Down (< −${FEE_PCT}%)`, "down", pred.p_down)),
    el("dl", { class: "meta" },
      el("dt", { text: "Price at signal" }), el("dd", { text: fmtPrice(pred.entry_price) }),
      el("dt", { text: "Advice expires" }), expiryNode,
      el("dt", { text: "Model" }), el("dd", { text: pred.model_version })),
    watchable && !alreadyWatching
      ? el("button", { onclick: () => startWatch(pred), text: "Re-check this position every 15 minutes" })
      : null
  );
  setResult.ticker = setInterval(updateExpiry, 15000);
}

async function onPredict() {
  if (state.busy) return;
  const pair = $("pair").value;
  state.busy = true;
  $("predict-btn").disabled = true;
  renderLoading(pair);
  try {
    const pred = await postPredict(pair);
    state.last = pred;
    renderPrediction(pred);
    refreshChart();
  } catch (err) {
    renderError(err, onPredict);
  } finally {
    state.busy = false;
    $("predict-btn").disabled = false;
  }
}

/* ---------- position watch: after a Buy, re-run the model every 15 minutes ---------- */
/* Runs in this page only: closing the tab stops it. Each check is a normal /api/predict
 * request made shortly after a bar closes, so it is also stored in the history. */

function startWatch(pred) {
  stopWatch();
  state.watch = {
    pair: pred.pair, endsAt: Date.parse(pred.expires_at), status: "active",
    timer: null, nextAt: null, log: [{ at: Date.now(), text: `Signal: ${pred.action}` }], message: "",
  };
  scheduleCheck();
  renderWatch();
  renderPrediction(pred);                 // hides the "re-check" button
}

function scheduleCheck() {
  const w = state.watch;
  if (!w) return;
  const nextClose = (Math.floor(Date.now() / BAR_MS) + 1) * BAR_MS;
  if (nextClose >= w.endsAt) {            // no bar closes before the hour is up: just wait for the end
    w.final = true;
    w.nextAt = w.endsAt;
    w.timer = setTimeout(
      () => finishWatch("expired", "One hour has passed since the signal. Time to sell."),
      Math.max(0, w.endsAt - Date.now())
    );
  } else {
    w.nextAt = nextClose + 15000;         // give Binance a few seconds to publish the closed bar
    w.timer = setTimeout(runCheck, w.nextAt - Date.now());
  }
  renderWatch();
}

async function runCheck() {
  const w = state.watch;
  if (!w || w.status !== "active") return;
  try {
    const pred = await postPredict(w.pair);
    w.log.unshift({ at: Date.now(), text: `Check: ${pred.action}` });
    if (state.last && state.last.pair === w.pair) {
      state.last = pred;
      renderPrediction(pred);
      refreshChart();
    }
    if (pred.action === "SELL") {
      finishWatch("sell", "The signal turned to Sell. Consider selling now.");
      return;
    }
  } catch (err) {
    w.log.unshift({ at: Date.now(), text: `Check failed: ${friendlyError(err).title}` });
  }
  scheduleCheck();
}

function stopWatch() {
  if (state.watch) clearTimeout(state.watch.timer);
  state.watch = null;
  renderWatch();
}

function finishWatch(status, message) {
  const w = state.watch;
  if (!w) return;
  clearTimeout(w.timer);
  w.status = status;
  w.message = message;
  renderWatch();
}

function renderWatch() {
  const host = $("watch");
  const w = state.watch;
  host.hidden = !w;
  host.classList.toggle("alert", !!w && w.status !== "active");
  if (!w) return host.replaceChildren();

  const active = w.status === "active";
  host.replaceChildren(
    el("h2", { text: active ? `Watching ${w.pair}` : `${w.pair}: watch finished` }),
    el("p", { class: active ? "muted small" : "", text: active
      ? (w.final
          ? `No more checks. The hour ends at ${fmtTime(w.endsAt)}, and then it is time to sell. Keep this tab open.`
          : `Next check at ${fmtTime(w.nextAt)}. The watch ends at ${fmtTime(w.endsAt)} (one hour after the signal). Keep this tab open.`)
      : w.message }),
    el("ul", {}, w.log.slice(0, 5).map((e) => el("li", { text: `${fmtTime(e.at)}: ${e.text}` }))),
    el("div", { class: "row-actions" },
      el("button", { onclick: stopWatch, text: active ? "Stop watching" : "Dismiss" }))
  );
}

/* ---------- history tab ---------- */

const STATUS_LABEL = { pending: "pending", correct: "correct", incorrect: "incorrect", flat: "within fee", neutral: "no trade" };

function statCard(label, value, sub, cls = "") {
  return el("div", { class: "stat" },
    el("div", { class: "label", text: label }),
    el("div", { class: `value ${cls}`, text: value }),
    el("div", { class: "sub", text: sub }));
}

function renderStats(s) {
  const n = s.resolved_signals;
  const host = $("stats");
  host.replaceChildren(
    statCard("Resolved Buy/Sell signals", String(n), `n = ${n}; ${s.pending_predictions} still pending`),
    statCard("Signals that made money after fees",
      s.signal_win_rate == null ? "—" : `${(s.signal_win_rate * 100).toFixed(0)}%`, `n = ${n}`),
    statCard("Average net return per signal",
      s.signal_avg_net_return == null ? "—" : fmtRet(s.signal_avg_net_return), `n = ${n}, after the ${FEE_PCT}% fee`,
      retClass(s.signal_avg_net_return)),
    statCard("All predictions", String(s.total_predictions), `${s.resolved_predictions} resolved (n = ${s.resolved_predictions})`)
  );
  if (n < 30 && n > 0)
    host.append(el("p", { class: "muted small", text: `Only ${n} resolved signals so far, too few to draw conclusions.` }));
}

function renderHistory(rows) {
  const body = $("history-table").querySelector("tbody");
  body.replaceChildren(...rows.map((p) =>
    el("tr", {},
      el("td", { text: fmtDateTime(Date.parse(p.bar_time) + BAR_MS) }),
      el("td", { text: p.pair }),
      el("td", { class: `act-${p.action}`, text: p.action }),
      el("td", { class: "num", text: `${(p.p_up * 100).toFixed(0)} / ${(p.p_flat * 100).toFixed(0)} / ${(p.p_down * 100).toFixed(0)} %` }),
      el("td", { class: "num", text: fmtPrice(p.entry_price) }),
      el("td", { class: "num", text: fmtPrice(p.exit_price) }),
      el("td", { class: `num ${p.action === "WAIT" ? "" : retClass(p.net_return)}`, text: p.action === "WAIT" ? "—" : fmtRet(p.net_return) }),
      el("td", {}, el("span", { class: `badge ${p.status}`, text: STATUS_LABEL[p.status] || p.status })))
  ));
  const msg = $("history-msg");
  msg.hidden = rows.length > 0;
  msg.textContent = "No predictions yet. Make one on the Predict tab.";
}

async function loadHistory() {
  const pair = $("history-pair").value;
  try {
    const [stats, rows] = await Promise.all([
      api("/stats"),
      api(`/predictions?limit=100${pair ? `&pair=${encodeURIComponent(pair)}` : ""}`),
    ]);
    renderStats(stats);
    renderHistory(rows);
  } catch (err) {
    $("history-msg").hidden = false;
    $("history-msg").textContent = `Couldn’t load the history: ${friendlyError(err).detail}`;
  }
}

/* ---------- tabs and start-up ---------- */

function showTab(name) {
  state.tab = name;
  for (const btn of document.querySelectorAll("[role=tab]"))
    btn.setAttribute("aria-selected", String(btn.dataset.tab === name));
  $("view-predict").hidden = name !== "predict";
  $("view-history").hidden = name !== "history";
  if (name === "history") loadHistory();
  else refreshChart();
}

async function init() {
  for (const btn of document.querySelectorAll("[role=tab]")) btn.addEventListener("click", () => showTab(btn.dataset.tab));
  $("predict-btn").addEventListener("click", onPredict);
  $("history-refresh").addEventListener("click", loadHistory);
  $("history-pair").addEventListener("change", loadHistory);
  renderIdle();

  let pairs;
  try {
    pairs = await api("/pairs");
  } catch (err) {
    $("pair").replaceChildren(el("option", { text: "Unavailable" }));
    renderError(err, () => location.reload());
    return;
  }
  let saved = null;
  try { saved = localStorage.getItem("pair"); } catch (_) { /* storage unavailable */ }
  $("pair").replaceChildren(...pairs.map((p) => el("option", { value: p, text: p })));
  $("history-pair").append(...pairs.map((p) => el("option", { value: p, text: p })));
  state.pair = pairs.includes(saved) ? saved : pairs[0];
  $("pair").value = state.pair;
  $("pair").disabled = false;
  $("predict-btn").disabled = false;

  $("pair").addEventListener("change", () => {
    state.pair = $("pair").value;
    try { localStorage.setItem("pair", state.pair); } catch (_) { /* ignore */ }
    if (state.last && state.last.pair !== state.pair) renderIdle();
    else if (state.last) renderPrediction(state.last);
    refreshChart();
  });

  refreshChart();
  let resizeTimer = null;
  window.addEventListener("resize", () => {            // keep the chart sharp when the window changes size
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => { if (state.candles.length && state.tab === "predict") redrawChart(); }, 150);
  });
  setInterval(() => { if (state.tab === "predict" && !document.hidden) refreshChart(); }, 60000);
  setInterval(() => { if (state.tab === "history" && !document.hidden) loadHistory(); }, 30000);
}

init();