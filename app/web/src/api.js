import { API } from "./utils.js";

export class ApiError extends Error {
  constructor(message, status) {
    super(message);
    this.status = status;
  }
}

export async function api(path, options = {}, timeoutMs = 60000) {
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

export const getPairs = () => api("/pairs");
export const getCandles = (pair, limit) =>
  api(`/candles?pair=${encodeURIComponent(pair)}&interval=15m&limit=${limit}`);
export const postPredict = (pair) => api("/predict", { method: "POST", body: JSON.stringify({ pair }) });
export const getStats = () => api("/stats");
export const getModel = () => api("/model");
export const postRetrain = () => api("/model/retrain", { method: "POST" });
export const getBacktest = () => api("/backtest");
export const getPredictions = (pair) =>
  api(`/predictions?limit=100${pair ? `&pair=${encodeURIComponent(pair)}` : ""}`);

export function friendlyError(err) {
  if (err.status === 400) return { title: "That pair isn’t supported.", detail: err.message };
  if (err.status === 503)
    return {
      title: "No prediction right now.",
      detail: `${err.message} Market data or the model service may be temporarily unavailable. Please try again in a moment.`,
    };
  if (err.status === 0) return { title: err.message, detail: "Check that the server is running, then try again." };
  return { title: "Something went wrong.", detail: err.message };
}
