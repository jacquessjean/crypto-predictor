import json
from datetime import datetime, timezone
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .features_live import (
    BINANCE_KLINES_URL,
    BinanceKlineClient,
    BinanceRestError,
    LiveFeatureError,
    _default_client,
    fetch_klines,
    fetch_pair_klines,
)

BAR_MS = 15 * 60 * 1000  # 900,000 ms


def get_latest_closed_15m_bar(
    pair_klines: list[dict[str, object]],
    pair: str,
    now_ms: int | None = None,
) -> tuple[str, int, float]:
    """
    Identify the most recent closed 15-minute bar from 1-minute klines.

    Returns:
        (bar_time_iso, bar_close_ms, entry_price)
    """
    if now_ms is None:
        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

    # Latest 15m bucket boundary at or before now_ms is the currently OPEN bar;
    # the bar immediately preceding it is the latest CLOSED 15m bar.
    current_15m_open_ms = (now_ms // BAR_MS) * BAR_MS
    target_open_ms = current_15m_open_ms - BAR_MS
    target_close_ms = current_15m_open_ms

    in_bar = [
        k for k in pair_klines
        if target_open_ms <= int(k["open_time"]) < target_close_ms
    ]
    if not in_bar:
        raise LiveFeatureError(f"no closed 1-minute bars in latest 15m window for {pair}")

    in_bar.sort(key=lambda k: int(k["open_time"]))
    entry_price = float(in_bar[-1]["close"])
    bar_time_iso = datetime.fromtimestamp(target_open_ms / 1000, tz=timezone.utc).isoformat()
    return bar_time_iso, target_close_ms, entry_price


def fetch_close_at(pair: str, target_close_ms: int) -> float | None:
    """
    Fetch the closing price at `target_close_ms` (e.g. 1 hour after prediction bar close).
    Looks at a 5-minute window ending at `target_close_ms` to tolerate quiet 1m intervals.
    """
    window_start = target_close_ms - (5 * 60 * 1000)
    klines = fetch_klines(pair.upper(), window_start, target_close_ms)
    if not klines:
        return None
    klines.sort(key=lambda k: int(k["open_time"]))
    return float(klines[-1]["close"])


def fetch_chart_candles(
    pair: str, interval: str = "15m", limit: int = 96
) -> list[dict[str, object]]:
    """Fetch aggregated chart candles directly from Binance for GET /api/candles."""
    query = urlencode(
        {
            "symbol": pair.upper(),
            "interval": interval,
            "limit": min(max(limit, 1), 500),
        }
    )
    url = f"{BINANCE_KLINES_URL}?{query}"
    req = Request(url, headers={"Accept": "application/json", "User-Agent": "crypto-predictor"})
    try:
        with urlopen(req, timeout=10.0) as resp:
            raw = json.loads(resp.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise BinanceRestError(f"Failed to fetch chart candles for {pair}") from exc

    if not isinstance(raw, list):
        raise BinanceRestError("Unexpected Binance response format for chart candles")

    return [
        {
            "open_time": int(row[0]),
            "open": float(row[1]),
            "high": float(row[2]),
            "low": float(row[3]),
            "close": float(row[4]),
            "volume": float(row[5]),
        }
        for row in raw
    ]


__all__ = [
    "BinanceKlineClient",
    "BinanceRestError",
    "_default_client",
    "fetch_chart_candles",
    "fetch_close_at",
    "fetch_klines",
    "fetch_pair_klines",
    "get_latest_closed_15m_bar",
]