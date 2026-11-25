import { fmtTime } from "./utils.js";

export default function WatchCard({ watch, onStop }) {
  const active = watch.status === "active";
  return (
    <div className={`card watch-card${active ? "" : " alert"}`} aria-live="polite">
      <h2>{active ? `Watching ${watch.pair}` : `${watch.pair}: watch finished`}</h2>
      <p className={active ? "muted small" : ""}>
        {active
          ? watch.final
            ? `No more checks. The hour ends at ${fmtTime(watch.endsAt)}, and then it is time to sell. Keep this tab open.`
            : `Next check at ${watch.nextAt ? fmtTime(watch.nextAt) : "…"}. The watch ends at ${fmtTime(watch.endsAt)} (one hour after the signal). Keep this tab open.`
          : watch.message}
      </p>
      <ul>
        {watch.log.slice(0, 5).map((e, i) => <li key={i}>{`${fmtTime(e.at)}: ${e.text}`}</li>)}
      </ul>
      <div className="row-actions">
        <button onClick={onStop}>{active ? "Stop watching" : "Dismiss"}</button>
      </div>
    </div>
  );
}
