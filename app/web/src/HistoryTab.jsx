import { useCallback, useEffect, useState } from "react";
import { getPredictions, getStats, friendlyError } from "./api.js";
import {
  BAR_MS, FEE_PCT, STATUS_LABEL, fmtDateTime, fmtPrice, fmtRet, retClass,
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

export default function HistoryTab({ pairs }) {
  const [pair, setPair] = useState("");
  const [stats, setStats] = useState(null);
  const [rows, setRows] = useState([]);
  const [error, setError] = useState("");

  const load = useCallback(async () => {
    try {
      const [s, r] = await Promise.all([getStats(), getPredictions(pair)]);
      setStats(s);
      setRows(r);
      setError("");
    } catch (err) {
      setError(`Couldn’t load the history: ${friendlyError(err).detail}`);
    }
  }, [pair]);

  useEffect(() => {
    load();
    const id = setInterval(() => { if (!document.hidden) load(); }, 30000);
    return () => clearInterval(id);
  }, [load]);

  return (
    <section role="tabpanel" aria-labelledby="tab-history">
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
                <td className={`act-${p.action}`}>{p.action}</td>
                <td className="num">{`${(p.p_up * 100).toFixed(0)} / ${(p.p_flat * 100).toFixed(0)} / ${(p.p_down * 100).toFixed(0)} %`}</td>
                <td className="num">{fmtPrice(p.entry_price)}</td>
                <td className="num">{fmtPrice(p.exit_price)}</td>
                <td className={`num ${p.action === "WAIT" ? "" : retClass(p.net_return)}`}>
                  {p.action === "WAIT" ? "—" : fmtRet(p.net_return)}
                </td>
                <td><span className={`badge ${p.status}`}>{STATUS_LABEL[p.status] || p.status}</span></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}
