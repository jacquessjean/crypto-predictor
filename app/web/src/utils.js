export const API = "/api";
export const FEE_PCT = 0.2;               // round-trip spot taker fee, in percent
export const BAR_MS = 15 * 60 * 1000;     // 15-minute bars
export const CHART_LIMIT = 96;            // 24 h of 15-minute candles
export const WEAK_SIGNAL_NOTE =
  "The model’s predictive power is weak, so most of the time the honest answer is Wait. " +
  "This is not financial advice.";

export const fmtTime = (t) => new Date(t).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
export const fmtDateTime = (t) =>
  new Date(t).toLocaleString([], { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
export const cap = (s) => s.charAt(0) + s.slice(1).toLowerCase();
export const fmtPct = (p) => `${(p * 100).toFixed(1)}%`;
export const fmtRet = (r) => (r == null ? "—" : `${r >= 0 ? "+" : ""}${(r * 100).toFixed(2)}%`);
export const retClass = (r) => (r == null || r === 0 ? "" : r > 0 ? "pos" : "neg");

export function fmtPrice(v) {
  if (v == null) return "—";
  const digits = v >= 1000 ? 2 : v >= 1 ? 4 : 6;
  return v.toLocaleString(undefined, { minimumFractionDigits: digits, maximumFractionDigits: digits });
}

export function minutesLeft(ms) {
  const mins = Math.round(ms / 60000);
  return mins >= 60 ? `${Math.floor(mins / 60)} h ${mins % 60} min` : `${mins} min`;
}

export const EXPLAIN = {
  BUY: `The price is expected to rise by more than the ${FEE_PCT}% round-trip fee over the next hour.`,
  SELL: `The price is expected to fall by more than the ${FEE_PCT}% fee over the next hour. If you hold this coin, consider selling; if you don’t, don’t buy.`,
  WAIT: `No clear edge after the ${FEE_PCT}% fee. Waiting is the honest answer here.`,
};

export const STATUS_LABEL = {
  pending: "pending", correct: "correct", incorrect: "incorrect", flat: "within fee", neutral: "no trade",
};
