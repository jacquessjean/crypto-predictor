import httpx

from .config import settings


class ModelServiceError(RuntimeError):
    """Raised when the model inference service is unreachable or returns an error."""


class FeedbackRejected(ModelServiceError):
    """The model service refused a feedback batch as malformed (HTTP 422): resending it unchanged cannot succeed."""


def predict_from_features(
    pair: str,
    bar_time: str,
    features: dict[str, float | int],
) -> dict:
    """
    POST the 36 engineered features to the model service (`POST /v1/predict`).

    Expected model service response:
    {
        "p_up": 0.38,
        "p_flat": 0.30,
        "p_down": 0.32,
        "action": "WAIT",
        "model_version": "v1",
        "schema_version": "v1"
    }
    """
    url = f"{settings.model_url}/v1/predict"
    payload = {
        "pair": pair,
        "bar_time": bar_time,
        "features": features,
    }

    try:
        with httpx.Client(timeout=settings.request_timeout_seconds) as client:
            response = client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
    except httpx.TimeoutException as exc:
        raise ModelServiceError("Model service timed out") from exc
    except httpx.HTTPStatusError as exc:
        raise ModelServiceError(
            f"Model service returned HTTP {exc.response.status_code}"
        ) from exc
    except (httpx.RequestError, ValueError) as exc:
        raise ModelServiceError("Model service is unavailable") from exc

    required_keys = {"p_up", "p_flat", "p_down", "action"}
    missing = required_keys - set(data.keys())
    if missing:
        raise ModelServiceError(f"Malformed model response; missing: {', '.join(sorted(missing))}")

    return {
        "p_up": float(data["p_up"]),
        "p_flat": float(data["p_flat"]),
        "p_down": float(data["p_down"]),
        "action": str(data["action"]).upper(),
        "model_version": str(data.get("model_version", "unknown")),
        "schema_version": str(data.get("schema_version", "unknown")),
        # how close to a Buy: expected return after the fee, and the learned margin it must exceed (None if not sent)
        "expected_net": None if data.get("expected_net") is None else float(data["expected_net"]),
        "buy_margin": None if data.get("buy_margin") is None else float(data["buy_margin"]),
    }


def send_feedback(samples: list[dict], transport: httpx.BaseTransport | None = None) -> int:
    """POST resolved predictions to the model service (`POST /v1/feedback`) as labeled samples for retraining.

    Each sample: {"pair", "bar_time", "features": {36 names}, "realized_return"}. Returns how many the service took
    (new plus already known); raises FeedbackRejected when it refuses the batch as malformed, ModelServiceError when
    it cannot be reached. `transport` is for tests.
    """
    try:
        with httpx.Client(timeout=settings.request_timeout_seconds, transport=transport) as client:
            response = client.post(f"{settings.model_url}/v1/feedback", json={"samples": samples})
            if response.status_code == 422:
                raise FeedbackRejected(f"Model service refused the feedback: {response.text[:200]}")
            response.raise_for_status()
            data = response.json()
    except httpx.HTTPStatusError as exc:
        raise ModelServiceError(f"Model service returned HTTP {exc.response.status_code}") from exc
    except (httpx.RequestError, ValueError) as exc:
        raise ModelServiceError("Model service is unavailable") from exc
    return int(data.get("accepted", 0)) + int(data.get("duplicates", 0))


def get_model_status(transport: httpx.BaseTransport | None = None) -> dict:
    """The live model and its online learning, from the model service's `GET /health`: version, horizon, number of
    feedback samples received, and the last retrain (status, reason, version, time) or None before the first one.
    `transport` is for tests. Raises ModelServiceError when the service cannot be reached or answers badly."""
    try:
        with httpx.Client(timeout=settings.request_timeout_seconds, transport=transport) as client:
            response = client.get(f"{settings.model_url}/health")
            response.raise_for_status()
            data = response.json()
        last = data.get("last_retrain")
        return {
            "model_version": str(data["model_version"]),
            "horizon_minutes": int(data.get("horizon_minutes", 60)),
            "feedback_samples": int(data.get("feedback_samples", 0)),
            "new_feedback": int(data.get("new_feedback", 0)),
            "min_new_feedback": int(data.get("min_new_feedback", 50)),
            "retraining": bool(data.get("retraining", False)),
            # an error's reason is the raw exception (paths, internals): it stays in the model service's log
            "last_retrain": None if not last else {**{k: last.get(k) for k in ("status", "reason", "version", "at")},
                                                   **({"reason": "see the model service log"} if last.get("status") == "error" else {})},
        }
    except httpx.HTTPStatusError as exc:
        raise ModelServiceError(f"Model service returned HTTP {exc.response.status_code}") from exc
    except (httpx.RequestError, ValueError, KeyError, TypeError) as exc:
        raise ModelServiceError("Model service is unavailable") from exc


class RetrainRunning(ModelServiceError):
    """A retrain is already running on the model service (HTTP 409)."""


def start_retrain(transport: httpx.BaseTransport | None = None) -> bool:
    """Asks the model service to start a retrain in the background now (`POST /v1/retrain/start`).
    Raises RetrainRunning if one is running, ModelServiceError if the service cannot be reached."""
    try:
        with httpx.Client(timeout=settings.request_timeout_seconds, transport=transport) as client:
            response = client.post(f"{settings.model_url}/v1/retrain/start")
            if response.status_code == 409:
                raise RetrainRunning("a retrain is already running")
            response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise ModelServiceError(f"Model service returned HTTP {exc.response.status_code}") from exc
    except httpx.RequestError as exc:
        raise ModelServiceError("Model service is unavailable") from exc
    return True
