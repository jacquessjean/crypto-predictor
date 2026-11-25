import { useEffect, useRef, useState } from "react";
import { getPairs, postPredict, friendlyError } from "./api.js";
import { BAR_MS } from "./utils.js";
import Chart from "./Chart.jsx";
import ResultCard from "./ResultCard.jsx";
import WatchCard from "./WatchCard.jsx";
import HistoryTab from "./HistoryTab.jsx";

function loadSavedPair() {
  try { return localStorage.getItem("pair"); } catch (_) { return null; }   // storage may be unavailable
}

export default function App() {
  const [tab, setTab] = useState("predict");
  const [pairs, setPairs] = useState([]);
  const [pair, setPair] = useState(null);
  const [initError, setInitError] = useState(null);

  const [phase, setPhase] = useState("idle");       // idle | loading | error
  const [error, setError] = useState(null);
  const [last, setLast] = useState(null);           // most recent prediction shown on the Predict tab
  const [chartKey, setChartKey] = useState(0);      // bumped to reload the chart after a prediction
  const [busy, setBusy] = useState(false);

  const [watch, setWatch] = useState(null);         // position watch (re-check after a Buy)
  const watchRef = useRef(null);
  watchRef.current = watch;

  /* ---------- start-up ---------- */
  useEffect(() => {
    getPairs()
      .then((list) => {
        setPairs(list);
        const saved = loadSavedPair();
        setPair(list.includes(saved) ? saved : list[0]);
      })
      .catch((err) => setInitError(err));
  }, []);

  function onPairChange(value) {
    setPair(value);
    try { localStorage.setItem("pair", value); } catch (_) { /* ignore */ }
    if (!busy) { setPhase("idle"); setError(null); }
  }

  /* ---------- predict ---------- */
  async function onPredict() {
    if (busy || !pair) return;
    setBusy(true);
    setPhase("loading");
    try {
      const pred = await postPredict(pair);
      setLast(pred);
      setPhase("idle");
      setChartKey((k) => k + 1);
    } catch (err) {
      setError(err);
      setPhase("error");
    } finally {
      setBusy(false);
    }
  }

  /* ---------- position watch: after a Buy, re-run the model every 15 minutes ---------- */
  /* Runs in this page only: closing the tab stops it. Each check is a normal /api/predict
   * request made shortly after a bar closes, so it is also stored in the history. */
  function startWatch(pred) {
    setWatch({
      pair: pred.pair, endsAt: Date.parse(pred.expires_at), status: "active", tick: 0,
      nextAt: null, final: false, message: "",
      log: [{ at: Date.now(), text: `Signal: ${pred.action}` }],
    });
  }

  function finishWatch(status, message) {
    setWatch((w) => (w ? { ...w, status, message } : w));
  }

  async function runCheck() {
    const w = watchRef.current;
    if (!w || w.status !== "active") return;
    try {
      const pred = await postPredict(w.pair);
      setWatch((cur) => cur && { ...cur, log: [{ at: Date.now(), text: `Check: ${pred.action}` }, ...cur.log] });
      setLast((l) => (l && l.pair === w.pair ? pred : l));
      setChartKey((k) => k + 1);
      if (pred.action === "SELL") {
        finishWatch("sell", "The signal turned to Sell. Consider selling now.");
        return;
      }
    } catch (err) {
      setWatch((cur) => cur && { ...cur, log: [{ at: Date.now(), text: `Check failed: ${friendlyError(err).title}` }, ...cur.log] });
    }
    setWatch((cur) => cur && { ...cur, tick: cur.tick + 1 });      // schedule the next check
  }

  const watchActive = watch?.status === "active";
  const watchPair = watch?.pair;
  const watchTick = watch?.tick;
  const watchEndsAt = watch?.endsAt;
  useEffect(() => {
    if (!watchActive) return;
    const nextClose = (Math.floor(Date.now() / BAR_MS) + 1) * BAR_MS;
    let timer;
    if (nextClose >= watchEndsAt) {                // no bar closes before the hour is up: just wait for the end
      setWatch((w) => w && { ...w, final: true, nextAt: w.endsAt });
      timer = setTimeout(
        () => finishWatch("expired", "One hour has passed since the signal. Time to sell."),
        Math.max(0, watchEndsAt - Date.now())
      );
    } else {
      const nextAt = nextClose + 15000;            // give Binance a few seconds to publish the closed bar
      setWatch((w) => w && { ...w, final: false, nextAt });
      timer = setTimeout(() => runCheck(), nextAt - Date.now());
    }
    return () => clearTimeout(timer);
  }, [watchActive, watchPair, watchTick, watchEndsAt]);

  /* ---------- render ---------- */
  const shown = last && last.pair === pair ? last : null;
  const marker = shown ? { barOpen: Date.parse(shown.bar_time), price: shown.entry_price } : null;
  const watching = watchActive && watchPair === last?.pair;

  return (
    <>
      <header className="topbar">
        <h1>Crypto Predictor</h1>
        <nav className="tabs" role="tablist" aria-label="Sections">
          {["predict", "history"].map((name) => (
            <button
              key={name}
              role="tab"
              id={`tab-${name}`}
              aria-selected={tab === name}
              onClick={() => setTab(name)}
            >
              {name === "predict" ? "Predict" : "History"}
            </button>
          ))}
        </nav>
      </header>

      <main>
        {tab === "predict" && (
          <section role="tabpanel" aria-labelledby="tab-predict">
            <div className="controls">
              <label htmlFor="pair">Pair</label>
              <select id="pair" value={pair ?? ""} disabled={!pair} onChange={(e) => onPairChange(e.target.value)}>
                {pair ? pairs.map((p) => <option key={p} value={p}>{p}</option>)
                      : <option>{initError ? "Unavailable" : "Loading…"}</option>}
              </select>
              <button id="predict-btn" className="primary" disabled={!pair || busy} onClick={onPredict}>Predict</button>
            </div>

            <div className="grid">
              <Chart pair={pair} marker={marker} refreshKey={chartKey} />
              <div className="side">
                <ResultCard
                  phase={initError ? "error" : phase}
                  pair={pair}
                  pred={shown}
                  error={initError || error}
                  onRetry={initError ? () => window.location.reload() : onPredict}
                  watching={watching}
                  onWatch={startWatch}
                />
                {watch && <WatchCard watch={watch} onStop={() => setWatch(null)} />}
              </div>
            </div>
          </section>
        )}

        {tab === "history" && <HistoryTab pairs={pairs} />}
      </main>

      <footer>Educational project · not financial advice</footer>
    </>
  );
}
