import { useEffect, useState } from "react";
import { friendlyError } from "./api.js";
import { EXPLAIN, FEE_PCT, BAR_MS, WEAK_SIGNAL_NOTE, fmtPct, fmtPrice, fmtTime, minutesLeft } from "./utils.js";

function ProbRow({ label, cls, value }) {
  return (
    <div className="prob">
      <span>{label}</span>
      <div className="track"><div className={`fill ${cls}`} style={{ width: `${(value * 100).toFixed(1)}%` }} /></div>
      <span className="pct">{fmtPct(value)}</span>
    </div>
  );
}

function Expiry({ expiresAt }) {
  const [, tick] = useState(0);
  useEffect(() => {
    const id = setInterval(() => tick((n) => n + 1), 15000);   // refresh the countdown
    return () => clearInterval(id);
  }, []);
  const left = expiresAt - Date.now();
  return <dd>{left > 0 ? `${fmtTime(expiresAt)} (in ${minutesLeft(left)})` : `${fmtTime(expiresAt)} (expired)`}</dd>;
}

function Prediction({ pred, watching, onWatch }) {
  const expiresAt = Date.parse(pred.expires_at);
  const barClose = Date.parse(pred.bar_time) + BAR_MS;
  const watchable = pred.action === "BUY" && expiresAt > Date.now();
  return (
    <>
      <div className="head-line"><span>{pred.pair}</span><span>{`data up to ${fmtTime(barClose)}`}</span></div>
      <div className={`action ${pred.action}`}>{pred.action}</div>
      <p className="explain">{EXPLAIN[pred.action]}</p>
      <div className="probs">
        <ProbRow label={`Up (> +${FEE_PCT}%)`} cls="up" value={pred.p_up} />
        <ProbRow label={`Flat (±${FEE_PCT}%)`} cls="flat" value={pred.p_flat} />
        <ProbRow label={`Down (< −${FEE_PCT}%)`} cls="down" value={pred.p_down} />
      </div>
      <dl className="meta">
        <dt>Price at signal</dt><dd>{fmtPrice(pred.entry_price)}</dd>
        <dt>Advice expires</dt><Expiry expiresAt={expiresAt} />
        <dt>Model</dt><dd>{pred.model_version}</dd>
      </dl>
      {watchable && !watching && (
        <button onClick={() => onWatch(pred)}>Re-check this position every 15 minutes</button>
      )}
    </>
  );
}

/* phase: "idle" | "loading" | "error"; with phase "idle" a stored prediction is shown if there is one */
export default function ResultCard({ phase, pair, pred, error, onRetry, watching, onWatch }) {
  let content;
  if (phase === "loading") {
    content = (
      <>
        <h2>Prediction</h2>
        <div className="loading">
          <div className="spinner" aria-hidden="true" />
          <div>
            <strong>{`Analysing ${pair}…`}</strong>
            <div className="muted small">Fetching the latest market data and asking the model. This can take a few seconds.</div>
          </div>
        </div>
      </>
    );
  } else if (phase === "error") {
    const { title, detail } = friendlyError(error);
    content = (
      <>
        <h2>Prediction</h2>
        <div className="error-box" role="alert"><strong>{title}</strong><span>{detail}</span></div>
        <button className="primary" onClick={onRetry}>Try again</button>
      </>
    );
  } else if (pred) {
    content = <Prediction pred={pred} watching={watching} onWatch={onWatch} />;
  } else {
    content = (
      <>
        <h2>Prediction</h2>
        <p className="muted">Pick a pair and press Predict to get advice for the next hour.</p>
      </>
    );
  }
  return (
    <div className="card result-card" aria-live="polite">
      {content}
      <p className="note">{WEAK_SIGNAL_NOTE}</p>
    </div>
  );
}
