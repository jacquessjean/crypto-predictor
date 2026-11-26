import { useCallback, useEffect, useState } from "react";
import { getModel, getPredictions, getStats, postRetrain, friendlyError } from "./api.js";
import {
  BAR_MS, FEE_PCT, STATUS_LABEL, fmtDateTime, fmtPrice, fmtRet, retClass, cap,
} from "./utils.js";

function Stat({ label, value, sub, cls = "" }) {
  return (
    <div className="stat">
      <div className="label">{label}</div>
      <div className={`value ${cls}`}>{value}</div>
      <div className="sub">{sub}</div>
    </div>
  );
}

function Stats({ s }) {
  const n = s.resolved_signals;
  return (
    <div className="stats">
      <Stat label="Resolved Buy/Sell signals" value={String(n)} sub={`n = ${n}; ${s.pending_predictions} still pending`} />
      <Stat
        label="Signals that made money after fees"
        value={s.signal_win_rate == null ? "—" : `${(s.signal_win_rate * 100).toFixed(0)}%`}
        sub={`n = ${n}`}
      />
      <Stat
        label="Average net return per signal"
        value={s.signal_avg_net_return == null ? "—" : fmtRet(s.signal_avg_net_return)}
        sub={`n = ${n}, after the ${FEE_PCT}% fee`}
        cls={retClass(s.signal_avg_net_return)}
      />
      <Stat label="All predictions" value={String(s.total_predictions)} sub={`${s.resolved_predictions} resolved (n = ${s.resolved_predictions})`} />
      {n < 30 && n > 0 && <p className="muted small">{`Only ${n} resolved signals so far, too few to draw conclusions.`}</p>}
    </div>
  );
}

const RETRAIN_LABEL = { promoted: "Promoted", rejected: "Rejected", skipped: "Skipped", error: "Failed" };
const RETRAIN_NOTE = { promoted: "now live", rejected: "kept the live model", skipped: "kept the live model", error: "kept the live model" };

/* Online learning: every resolved prediction is sent back to the model service as a new training sample;
 * it retrains once a day and switches to the new version only if it passes the safety gate. */
function ModelStatus({ m, error, onRetrain, retrainMsg }) {
  if (error) return <p className="muted small">{`Model status unavailable: ${error}`}</p>;
  if (!m) return null;
  const r = m.last_retrain;
  return (
    <div className="stats model-status" aria-label="Model and online learning">
      <Stat label="Live model" value={m.model_version} sub={`${m.horizon_minutes}-minute horizon`} />
      <Stat label="Sent back for learning" value={String(m.feedback_samples)}
            sub={`${Math.min(m.new_feedback, m.min_new_feedback)} of ${m.min_new_feedback} new needed for the next retrain`} />
      <Stat
        label={m.retraining ? "Retrain" : "Last retrain"}
        value={m.retraining ? "Running…" : r ? (RETRAIN_LABEL[r.status] || r.status) : "None yet"}
        sub={m.retraining ? `about a minute; ${m.model_version} keeps answering meanwhile` : r ? [r.status === "promoted" && r.version ? `${r.version} ${RETRAIN_NOTE.promoted}` : RETRAIN_NOTE[r.status],
                  r.at && fmtDateTime(Date.parse(r.at)), r.reason].filter(Boolean).join(", ")
               : `starts by itself after ${m.min_new_feedback} new outcomes`}
      />
      <div className="model-actions">
        <p className="muted small">The model retrains by itself once enough new outcomes arrive. A new version goes live only if it is not worse than the current model on held-out data.</p>
        <button onClick={onRetrain} disabled={m.retraining || m.new_feedback === 0}
                title={m.new_feedback === 0 ? "No new outcomes since the live model was trained" : "Retrain now on the new outcomes; the safety gate still decides"}>
          Retrain now
        </button>
        {retrainMsg && <p className="muted small">{retrainMsg}</p>}
      </div>
    </div>
  );
}

export default function HistoryTab({ pairs }) {
  const [pair, setPair] = useState("");
  const [stats, setStats] = useState(null);
  const [rows, setRows] = useState([]);
  const [error, setError] = useState("");
  const [model, setModel] = useState(null);
  const [modelError, setModelError] = useState("");
  const [retrainMsg, setRetrainMsg] = useState("");

  async function onRetrain() {
    setRetrainMsg("");
    try {
      await postRetrain();
      setModel(await getModel());               // shows Running… and starts the 5-second checks
    } catch (err) {
      setRetrainMsg(err.status === 409 ? "A retrain is already running." : friendlyError(err).detail);
    }
  }

  const load = useCallback(async () => {
    getModel()                                    // separate, so a model-service hiccup never hides the history
      .then((m) => { setModel(m); setModelError(""); })
      .catch((err) => setModelError(friendlyError(err).detail));
    try {
      const [s, r] = await Promise.all([getStats(), getPredictions(pair)]);
      setStats(s);
      setRows(r);
      setError("");
    } catch (err) {
      setError(`Couldn’t load the history: ${friendlyError(err).detail}`);
    }
  }, [pair]);

  // while a retrain runs, check the model every 5 s so the card switches to the new version by itself
  const retraining = model?.retraining;
  useEffect(() => {
    if (!retraining) return;
    const id = setInterval(() => { getModel().then(setModel).catch(() => {}); }, 5000);
    return () => clearInterval(id);
  }, [retraining]);

  useEffect(() => {
    load();
    const id = setInterval(() => { if (!document.hidden) load(); }, 30000);
    return () => clearInterval(id);
  }, [load]);

  return (
    <section role="tabpanel" aria-labelledby="tab-history">
      <ModelStatus m={model} error={modelError} onRetrain={onRetrain} retrainMsg={retrainMsg} />
      {stats && <Stats s={stats} />}
      <div className="controls">
        <label htmlFor="history-pair">Show</label>
        <select id="history-pair" value={pair} onChange={(e) => setPair(e.target.value)}>
          <option value="">All pairs</option>
          {pairs.map((p) => <option key={p} value={p}>{p}</option>)}
        </select>
        <button onClick={load}>Refresh</button>
      </div>
      {error && <p className="muted">{error}</p>}
      {!error && rows.length === 0 && <p className="muted">No predictions yet. Make one on the Predict tab.</p>}
      <div className="table-wrap">
        <table id="history-table">
          <thead>
            <tr>
              <th>Bar closed</th><th>Pair</th><th>Action</th>
              <th title="Probability of up / flat / down">Up / Flat / Down</th>
              <th>Entry</th><th>Exit</th><th title="After the 0.2% round-trip fee">Net return</th><th>Status</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((p) => (
              <tr key={p.id}>
                <td>{fmtDateTime(Date.parse(p.bar_time) + BAR_MS)}</td>
                <td>{p.pair}</td>
                <td className={`act-${p.action}`}>{cap(p.action)}</td>
                <td className="num">{`${(p.p_up * 100).toFixed(0)} / ${(p.p_flat * 100).toFixed(0)} / ${(p.p_down * 100).toFixed(0)} %`}</td>
                <td className="num">{fmtPrice(p.entry_price)}</td>
                <td className="num">{fmtPrice(p.exit_price)}</td>
                <td className={`num ${p.action === "WAIT" ? "" : retClass(p.net_return)}`}>
                  {p.action === "WAIT" ? "—" : fmtRet(p.net_return)}
                </td>
                <td>
                  <span className={`badge ${p.status}`}>{STATUS_LABEL[p.status] || p.status}</span>
                  {p.source === "replay" && <span className="badge replay" title="Made afterwards for this past bar, from the data available when it closed">replay</span>}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}
