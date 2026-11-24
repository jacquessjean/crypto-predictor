import httpx

from .config import settings


class ModelServiceError(RuntimeError):
    """Raised when the model inference service is unreachable or returns an error."""


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
    }