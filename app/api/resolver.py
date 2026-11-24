import logging
from datetime import datetime, timezone

from .binance import BinanceRestError, fetch_close_at
from .config import ROUND_TRIP_FEE
from .db import get_due_pending_predictions, update_prediction_outcome

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

    return resolved_count