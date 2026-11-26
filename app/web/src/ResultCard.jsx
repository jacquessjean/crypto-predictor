import { useEffect, useState } from "react";
import { friendlyError } from "./api.js";
import { EXPLAIN, FEE_PCT, BAR_MS, WEAK_SIGNAL_NOTE, fmtPct, fmtPrice, fmtRet, fmtTime, minutesLeft, retClass, cap } from "./utils.js";

function Probs({ pred }) {
  const rows = [
    ["up", `Up (> +${FEE_PCT}%)`, pred.p_up],
    ["flat", `Flat (±${FEE_PCT}%)`, pred.p_flat],
    ["down", `Down (< −${FEE_PCT}%)`, pred.p_down],
  ];
  return (
    <div className="probs">
      <div className="stack" role="img" aria-label={rows.map(([, l, v]) => `${l} ${fmtPct(v)}`).join(", ")}>
        {rows.map(([cls, , v]) => <span key={cls} className={cls} style={{ flexGrow: v }} />)}
      </div>
      <ul className="legend">
        {rows.map(([cls, l, v]) => (
          <li key={cls}><i className={cls} /><span>{l}</span><b>{fmtPct(v)}</b></li>
        ))}
      </ul>
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
      <div className={`action ${pred.action}`}>{cap(pred.action)}</div>
      <p className="explain">{EXPLAIN[pred.action]}</p>
      <Probs pred={pred} />
      <dl className="meta">
        {pred.expected_net != null && pred.buy_margin != null && (
          <>
            <dt title={`The three probabilities weighted by how far prices typically move, minus the ${FEE_PCT}% fee`}>Expected return (after fee)</dt>
            <dd className={retClass(pred.expected_net)}>{fmtRet(pred.expected_net)}</dd>
            <dt title="A safety margin learned from past data: the model buys only above it">Needed for a Buy</dt>
            <dd>{`above ${fmtRet(pred.buy_margin)}`}</dd>
          </>
        )}
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
