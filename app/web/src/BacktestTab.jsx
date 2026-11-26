import { useEffect, useState } from "react";
import { getBacktest, friendlyError } from "./api.js";
import { FEE_PCT, fmtDateTime, fmtPrice, fmtRet, retClass } from "./utils.js";

/* Model v2's Buy signals replayed on November 2026, data no choice had seen during training or tuning.
 * Live, Buy is rare (most answers are Wait); this shows what following it would have done. */
function Tile({ label, value, sub, cls = "" }) {
  return (
    <div className="stat">
      <div className="label">{label}</div>
      <div className={`value ${cls}`}>{value}</div>
      <div className="sub">{sub}</div>
    </div>
  );
}

export default function BacktestTab() {
  const [data, setData] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    getBacktest().then(setData).catch((err) => setError(friendlyError(err).detail));
  }, []);

  if (error) return <p className="muted">{`Couldn’t load the past signals: ${error}`}</p>;
  if (!data) return <p className="muted">Loading…</p>;
  const s = data.summary;
  return (
    <section role="tabpanel" aria-labelledby="tab-signals">
      <p className="muted">
        {`Every Buy that model ${s.model_version} would have given in ${s.period}, data it never saw while it was built. `}
        {`Each trade buys when the 15-minute bar closes and sells one hour later, one position per coin, after the ${FEE_PCT}% fee.`}
      </p>
      <div className="stats">
        <Tile label="Buy trades" value={String(s.trades)} sub={`${s.first_bar} to ${s.last_bar}`} />
        <Tile label="Average net return per trade" value={fmtRet(s.mean_net)} cls={retClass(s.mean_net)}
              sub={`95% interval ${fmtRet(s.ci_low)} to ${fmtRet(s.ci_high)}`} />
        <Tile label="Trades that made money" value={`${(s.hit_rate * 100).toFixed(0)}%`} sub={`n = ${s.trades}`} />
        <Tile label="Buying at random instead" value={fmtRet(s.random_mean_net)} cls={retClass(s.random_mean_net)}
              sub={`per trade, n = ${s.random_trades}`} />
        <p className="muted small">{`${s.trades} trades are few: the interval includes zero, so this is weak evidence, not a promise.`}</p>
      </div>
      <div className="table-wrap">
        <table id="signals-table">
          <thead>
            <tr><th>Bought</th><th>Pair</th><th>Entry</th><th>Sold</th><th>Exit</th><th title={`After the ${FEE_PCT}% round-trip fee`}>Net return</th></tr>
          </thead>
          <tbody>
            {data.trades.map((t) => (
              <tr key={`${t.pair}-${t.entry_time}`}>
                <td>{fmtDateTime(Date.parse(t.entry_time))}</td>
                <td>{t.pair}</td>
                <td className="num">{fmtPrice(t.entry)}</td>
                <td>{fmtDateTime(Date.parse(t.exit_time))}</td>
                <td className="num">{fmtPrice(t.exit)}</td>
                <td className={`num ${retClass(t.net)}`}>{fmtRet(t.net)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}
