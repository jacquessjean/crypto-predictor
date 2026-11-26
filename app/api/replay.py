"""Replay: make the predictions the app would have made for every 15-minute bar of the past hours, and resolve them.

    docker compose exec api python -m app.api.replay --hours 24

Each past bar is scored exactly as a live request one second after it closed: the same live feature code, given only
the 1-minute klines that existed then, and the same model service. The hour after it is already over, so the outcome
is read from the real prices, with the live resolver's rule, and every outcome goes back to the model service as
feedback. Rows are stored with source = 'replay' and never replace a live prediction for the same bar.
"""
import argparse
import json
import logging
from datetime import datetime, timedelta, timezone

from .binance import get_latest_closed_15m_bar
from .config import HORIZON_SECONDS, SUPPORTED_PAIRS
from .db import get_prediction_by_bar, init_db, insert_prediction, update_prediction_outcome
from .features_live import LiveFeatureError, build_live_feature_row, fetch_pair_klines
from .model_client import predict_from_features
from .resolver import evaluate_outcome, send_pending_feedback

logger = logging.getLogger(__name__)
MIN_MS, BAR_MS, HORIZON_MS = 60_000, 900_000, HORIZON_SECONDS * 1000


def _close_before(klines: list[dict], t_ms: int) -> float | None:
    """The close at time t: the close of the last 1-minute bar that ended by t (within 5 minutes, as the resolver)."""
    inside = [k for k in klines if t_ms - 5 * MIN_MS <= int(k["open_time"]) < t_ms]
    return float(max(inside, key=lambda k: int(k["open_time"]))["close"]) if inside else None


def replay_pair(pair: str, hours: float, now_ms: int) -> dict:
    """Replays one pair's bars that closed in the last `hours` and whose hour is over: closes in (now - hours, now - 1 h]."""
    klines = fetch_pair_klines(pair, margin=timedelta(hours=6 + hours), end_time=now_ms // BAR_MS * BAR_MS)
    last_close = (now_ms - HORIZON_MS) // BAR_MS * BAR_MS
    closes = range(last_close, now_ms - int(hours * 3_600_000), -BAR_MS)
    counts = {"replayed": 0, "skipped": 0, "unusable": 0}
    for bar_close in sorted(closes):
        asked_at = bar_close + 1000  # one second after the bar closed, like a live request
        bar_time, bar_close_ms, entry = get_latest_closed_15m_bar(klines[pair], pair, now_ms=asked_at)
        if get_prediction_by_bar(pair, bar_time) is not None:
            counts["skipped"] += 1  # a live (or earlier replayed) prediction already exists: keep it
            continue
        try:
            features = build_live_feature_row(pair, klines[pair], klines.get("BTCUSDT"), now=asked_at)
        except LiveFeatureError as exc:
            logger.warning("%s %s not replayed: %s", pair, bar_time, exc)
            counts["unusable"] += 1
            continue
        exit_price = _close_before(klines[pair], bar_close_ms + HORIZON_MS)
        if exit_price is None:
            counts["unusable"] += 1
            continue
        out = predict_from_features(pair=pair, bar_time=bar_time, features=features)
        row = insert_prediction({
            "pair": pair, "request_time": datetime.now(timezone.utc).isoformat(), "bar_time": bar_time,
            "bar_close_ms": bar_close_ms, "expires_at_ms": bar_close_ms + HORIZON_MS,
            "expires_at": datetime.fromtimestamp((bar_close_ms + HORIZON_MS) / 1000, tz=timezone.utc).isoformat(),
            "p_up": out["p_up"], "p_flat": out["p_flat"], "p_down": out["p_down"], "action": out["action"],
            "entry_price": entry, "model_version": out["model_version"], "schema_version": out["schema_version"],
            "features_json": json.dumps(features), "source": "replay",
            "expected_net": out.get("expected_net"), "buy_margin": out.get("buy_margin")})
        realized, net, status = evaluate_outcome(out["action"], entry, exit_price)
        update_prediction_outcome(prediction_id=int(row["id"]), exit_price=exit_price, realized_return=realized,
                                  net_return=net, status=status)
        counts["replayed"] += 1
    return counts


def replay(hours: float = 24, pairs: list[str] | None = None, now_ms: int | None = None) -> dict:
    """Replays every pair, then sends all new outcomes to the model service. Returns the counts."""
    now_ms = now_ms or int(datetime.now(timezone.utc).timestamp() * 1000)
    total = {"replayed": 0, "skipped": 0, "unusable": 0}
    for pair in pairs or SUPPORTED_PAIRS:
        counts = replay_pair(pair, hours, now_ms)
        logger.info("%s: %s", pair, counts)
        total = {k: total[k] + counts[k] for k in total}
    sent = 0
    while batch := send_pending_feedback():  # up to 500 per call
        sent += batch
    # the app's own resolver (every 120 s) may send some of them first; /api/model shows the service's total
    return {**total, "feedback_sent_by_this_run": sent}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--hours", type=float, default=24, help="how far back to replay (default 24)")
    parser.add_argument("--pairs", nargs="*", help="pairs to replay (default: all 10)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)  # one line per pair, not per request
    init_db()
    print(replay(args.hours, [p.upper() for p in args.pairs] if args.pairs else None))
