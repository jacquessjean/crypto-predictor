import json
import logging
from datetime import datetime, timezone

from .binance import BinanceRestError, fetch_close_at
from .config import ROUND_TRIP_FEE
from .db import get_due_pending_predictions, get_unsent_feedback, mark_feedback_sent, update_prediction_outcome
from .model_client import FeedbackRejected, ModelServiceError, send_feedback

logger = logging.getLogger(__name__)


def evaluate_outcome(
    action: str, entry_price: float, exit_price: float
) -> tuple[float, float, str]:
    """Compute (realized_return, net_return, status) using the 0.2% fee hurdle."""
    realized_return = (exit_price - entry_price) / entry_price
    action = action.upper()

    if action == "BUY":
        net_return = realized_return - ROUND_TRIP_FEE
    elif action == "SELL":
        net_return = -realized_return - ROUND_TRIP_FEE
    else:
        net_return = 0.0

    if action not in ("BUY", "SELL"):
        status = "neutral"  # WAIT is not a trade: nothing to be right or wrong about
    elif abs(realized_return) <= ROUND_TRIP_FEE:
        status = "flat"
    elif action == "BUY":
        status = "correct" if realized_return > ROUND_TRIP_FEE else "incorrect"
    else:  # SELL
        status = "correct" if realized_return < -ROUND_TRIP_FEE else "incorrect"

    return realized_return, net_return, status


def resolve_pending_predictions() -> int:
    """Find all pending predictions past their 1-hour horizon and record outcomes."""
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    due = get_due_pending_predictions(now_ms)
    resolved_count = 0

    for row in due:
        try:
            exit_price = fetch_close_at(row["pair"], int(row["expires_at_ms"]))
            if exit_price is None:
                logger.warning("No exit price yet for prediction %s (%s)", row["id"], row["pair"])
                continue

            realized_ret, net_ret, status = evaluate_outcome(
                row["action"], float(row["entry_price"]), exit_price
            )
            update_prediction_outcome(
                prediction_id=int(row["id"]),
                exit_price=exit_price,
                realized_return=realized_ret,
                net_return=net_ret,
                status=status,
            )
            resolved_count += 1
        except BinanceRestError as exc:
            logger.warning("Failed to resolve prediction %s: %s", row["id"], exc)
        except Exception:
            logger.exception("Unexpected error resolving prediction %s", row["id"])

    send_pending_feedback()
    return resolved_count


def send_pending_feedback() -> int:
    """Online learning: send every resolved prediction (its features and realized return) back to the model service
    as a labeled sample. Rows are marked only once the service has them, so an outage is retried on the next run."""
    rows = get_unsent_feedback()
    if not rows:
        return 0
    samples = [{"pair": r["pair"], "bar_time": r["bar_time"], "features": json.loads(r["features_json"]),
                "realized_return": float(r["realized_return"])} for r in rows]
    try:
        send_feedback(samples)
    except FeedbackRejected:
        return _send_one_by_one(rows, samples)  # one bad sample must not block the rest forever
    except ModelServiceError as exc:
        logger.warning("Feedback not sent (will retry): %s", exc)
        return 0
    mark_feedback_sent([int(r["id"]) for r in rows])
    return len(rows)


def _send_one_by_one(rows: list[dict], samples: list[dict]) -> int:
    """After the service refused a batch: send each sample alone, set aside the ones it refuses."""
    sent = 0
    for row, sample in zip(rows, samples):
        try:
            send_feedback([sample])
        except FeedbackRejected as exc:
            logger.warning("Feedback for prediction %s refused, not sent again: %s", row["id"], exc)
            mark_feedback_sent([int(row["id"])], rejected=True)
            continue
        except ModelServiceError as exc:
            logger.warning("Feedback not sent (will retry): %s", exc)
            break
        mark_feedback_sent([int(row["id"])])
        sent += 1
    return sent