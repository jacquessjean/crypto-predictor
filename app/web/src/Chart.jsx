import { useEffect, useRef, useState } from "react";
import { getCandles, friendlyError } from "./api.js";
import { CHART_LIMIT, fmtPrice, fmtRet, fmtTime, retClass } from "./utils.js";

/* Candlestick chart drawn as SVG. `marker` = { barOpen, price } marks the scored bar and entry price. */
function Candles({ candles, marker, width, hover, setHover }) {
  const W = Math.max(320, Math.min(900, width || 800));   // 1 SVG unit = 1 CSS pixel
  const H = W < 520 ? 260 : 320;
  const padL = 8, padR = 70, padT = 10, padB = 24;
  const lo = Math.min(...candles.map((c) => c.low));
  const hi = Math.max(...candles.map((c) => c.high));
  const span = hi - lo || hi * 0.001 || 1;
  const min = lo - span * 0.06, max = hi + span * 0.06;
  const step = (W - padL - padR) / candles.length;
  const x = (i) => padL + (i + 0.5) * step;
  const y = (v) => padT + ((max - v) / (max - min)) * (H - padT - padB);
  const lastClose = candles[candles.length - 1].close;
  const bodyW = Math.max(2, step * 0.66);

  const maxLabels = Math.max(2, Math.floor((W - padL - padR) / 80));   // time labels, ~80 px apart
  const labelEvery = Math.ceil(candles.length / maxLabels);
  const markerIdx = marker ? candles.findIndex((c) => c.open_time === marker.barOpen) : -1;

  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      preserveAspectRatio="xMidYMid meet"
      onPointerMove={(e) => {
        const r = e.currentTarget.getBoundingClientRect();
        const i = Math.floor(((e.clientX - r.left) * (W / r.width) - padL) / step);
        setHover(i >= 0 && i < candles.length ? i : null);
      }}
      onPointerLeave={() => setHover(null)}
    >
      {[0, 1, 2, 3, 4].map((k) => {
        const value = min + ((max - min) * k) / 4;
        return (
          <g key={k}>
            <line className="grid-line" x1={padL} x2={W - padR} y1={y(value)} y2={y(value)} />
            {Math.abs(y(value) - y(lastClose)) > 14 && (
              <text x={W - padR + 6} y={y(value) + 4}>{fmtPrice(value)}</text>
            )}
          </g>
        );
      })}

      {candles.map((c, i) =>
        i % labelEvery === 0 && x(i) > padL + 24 ? (
          <text key={`t${c.open_time}`} x={x(i)} y={H - 6} textAnchor="middle">{fmtTime(c.open_time)}</text>
        ) : null
      )}

      {markerIdx >= 0 && (
        <>
          <rect className="marker-band" x={x(markerIdx) - step / 2} y={padT} width={step} height={H - padT - padB} />
          <line className="entry-line" x1={padL} x2={W - padR} y1={y(marker.price)} y2={y(marker.price)} />
          <text className="entry-tag" x={padL + 4} y={y(marker.price) - 4}>entry</text>
        </>
      )}

      {candles.map((c, i) => (
        <g key={c.open_time} className={c.close >= c.open ? "up" : "down"}>
          <line className="wick" x1={x(i)} x2={x(i)} y1={y(c.high)} y2={y(c.low)} />
          <rect
            x={x(i) - bodyW / 2}
            y={y(Math.max(c.open, c.close))}
            width={bodyW}
            height={Math.max(1, Math.abs(y(c.open) - y(c.close)))}
          />
        </g>
      ))}

      <line className="last-line" x1={padL} x2={W - padR} y1={y(lastClose)} y2={y(lastClose)} />
      {hover != null && <line className="crosshair" x1={x(hover)} x2={x(hover)} y1={padT} y2={H - padB} />}
      <rect className="last-chip" x={W - padR + 2} y={y(lastClose) - 9} width={padR - 4} height={18} />
      <text className="last-tag" x={W - padR + 6} y={y(lastClose) + 4}>{fmtPrice(lastClose)}</text>
    </svg>
  );
}

export default function Chart({ pair, marker, refreshKey }) {
  const [candles, setCandles] = useState([]);
  const [message, setMessage] = useState("");
  const [width, setWidth] = useState(0);
  const [hover, setHover] = useState(null);
  const hostRef = useRef(null);

  // keep the chart sharp when the container changes size
  useEffect(() => {
    const el = hostRef.current;
    if (!el) return;
    const ro = new ResizeObserver(() => setWidth(el.clientWidth));
    ro.observe(el);
    setWidth(el.clientWidth);
    return () => ro.disconnect();
  }, []);

  // load candles when the pair changes, after each prediction, and every minute
  useEffect(() => {
    if (!pair) return;
    let cancelled = false;                              // ignore out-of-order responses
    const load = async () => {
      try {
        const data = await getCandles(pair, CHART_LIMIT);
        if (cancelled) return;
        setCandles(data);
        setMessage("");
      } catch (err) {
        if (cancelled) return;
        setCandles([]);
        setMessage(`Couldn’t load the chart: ${friendlyError(err).detail}`);
      }
    };
    load();
    const timer = setInterval(() => { if (!document.hidden) load(); }, 60000);
    return () => { cancelled = true; clearInterval(timer); };
  }, [pair, refreshKey]);

  const last = candles[candles.length - 1];
  const c = candles[hover ?? candles.length - 1];
  return (
    <div className="card chart-card">
      <div className="card-head">
        <h2>{pair ? `${pair}, 15-minute candles, last 24 h` : "Chart"}</h2>
        {c && (
          <span className="ohlc" aria-live="off">
            <span>O {fmtPrice(c.open)}</span><span>H {fmtPrice(c.high)}</span>
            <span>L {fmtPrice(c.low)}</span><span>C {fmtPrice(c.close)}</span>
            <b className={retClass(c.close - c.open)}>{fmtRet((c.close - c.open) / c.open)}</b>
            <time>{fmtTime(c.open_time)}</time>
          </span>
        )}
      </div>
      <div id="chart" role="img" aria-label="Candlestick chart" ref={hostRef}>
        {candles.length > 0 && <Candles candles={candles} marker={marker} width={width} hover={hover} setHover={setHover} />}
      </div>
      {message && <p className="muted">{message}</p>}
    </div>
  );
}
