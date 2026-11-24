import json
import math
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

try:
    from src.cleanup import FEATURES as MODEL_FEATURES
    from src.cleanup import add_features, add_market_features, clean_symbol
except ImportError:
    MODEL_FEATURES = []
    add_features = add_market_features = clean_symbol = None


BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"
KLINE_COLUMNS = (
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_asset_volume",
    "number_of_trades",
    "taker_buy_base_asset_volume",
    "taker_buy_quote_asset_volume",
    "ignore",
)

_OPEN_BAR_MARGIN_MS = 60_000
_STALE_AFTER_MS = 30 * 60_000
_MAX_15M_GAP_MS = 15 * 60_000

FEATURES = tuple(MODEL_FEATURES) if MODEL_FEATURES else tuple(
    [f"symbol={symbol}" for symbol in (
        "BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT",
        "DOGEUSDT", "SOLUSDT", "LTCUSDT", "LINKUSDT", "AVAXUSDT",
    )] + [
        "returns", "volatility", "upper_wick", "lower_wick", "body_ratio",
        "taker_buy_ratio", "taker_buy_ratio_16", "relative_volume",
        "normalized_volatility", "rel_trade_size", "ret_4", "ret_16",
        "ret_96", "ret_672", "vol_96", "rv_15", "rv_60", "rv_ratio",
        "idle_minutes", "hour_sin", "hour_cos", "dow_sin", "dow_cos",
        "btc_ret_4", "btc_ret_16", "rel_ret_4",
    ]
)


class BinanceRestError(RuntimeError):
    pass


class LiveFeatureError(ValueError):
    """Live feature pipeline could not produce a safe feature row."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class BinanceKlineClient:
    """Fetch raw 1-minute Binance klines with retries and a short in-memory cache."""

    def __init__(
        self,
        base_url: str = BINANCE_KLINES_URL,
        cache_ttl_seconds: float = 60.0,
        timeout_seconds: float = 10.0,
        retries: int = 3,
        opener: Callable[..., object] | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.base_url = base_url
        self.cache_ttl_seconds = cache_ttl_seconds
        self.timeout_seconds = timeout_seconds
        self.retries = retries
        self.opener = opener or urlopen
        self.sleep = sleep
        self.clock = clock
        self._cache: dict[tuple[str, int, int], tuple[float, list[dict[str, object]]]] = {}
        self._cache_lock = threading.Lock()

    def fetch_klines(
        self,
        symbol: str,
        start_time: datetime | int,
        end_time: datetime | int,
    ) -> list[dict[str, object]]:
        symbol = symbol.upper()
        start_ms = self._to_ms(start_time)
        end_ms = self._to_ms(end_time)
        if start_ms >= end_ms:
            raise ValueError("start_time must be before end_time")

        cache_key = (symbol, start_ms, end_ms)
        now = self.clock()
        with self._cache_lock:
            cached = self._cache.get(cache_key)
            if cached is not None and now - cached[0] < self.cache_ttl_seconds:
                return [row.copy() for row in cached[1]]

        klines: list[dict[str, object]] = []
        seen_open_times: set[int] = set()
        page_start = start_ms

        while page_start < end_ms:
            query = urlencode({
                "symbol": symbol,
                "interval": "1m",
                "startTime": page_start,
                "endTime": end_ms,
                "limit": 1000,
            })
            raw = self._request(query)
            if not raw:
                break

            for row in raw:
                if not isinstance(row, list) or len(row) < 12:
                    raise BinanceRestError("Binance returned a malformed kline")
                open_time = int(row[0])
                if open_time < start_ms or open_time >= end_ms or open_time in seen_open_times:
                    continue
                seen_open_times.add(open_time)
                klines.append({
                    "open_time": open_time,
                    "open": float(row[1]),
                    "high": float(row[2]),
                    "low": float(row[3]),
                    "close": float(row[4]),
                    "volume": float(row[5]),
                    "close_time": int(row[6]),
                    "quote_asset_volume": float(row[7]),
                    "number_of_trades": int(row[8]),
                    "taker_buy_base_asset_volume": float(row[9]),
                    "taker_buy_quote_asset_volume": float(row[10]),
                    "ignore": str(row[11]),
                })

            last_open_time = int(raw[-1][0])
            if len(raw) < 1000 or last_open_time >= end_ms - 60_000:
                break
            next_start = last_open_time + 60_000
            if next_start <= page_start:
                raise BinanceRestError("Binance pagination did not advance")
            page_start = next_start

        klines.sort(key=lambda row: int(row["open_time"]))
        with self._cache_lock:
            self._cache[cache_key] = (now, klines)
        return [row.copy() for row in klines]

    def fetch_last_7_days(
        self,
        symbol: str,
        margin: timedelta = timedelta(hours=6),
        end_time: datetime | int | None = None,
    ) -> list[dict[str, object]]:
        end = datetime.now(timezone.utc) if end_time is None else end_time
        end_ms = self._to_ms(end)
        start_ms = end_ms - (timedelta(days=7) + margin).total_seconds() * 1000
        return self.fetch_klines(symbol, int(start_ms), end_ms)

    def fetch_pair(
        self,
        pair: str,
        margin: timedelta = timedelta(hours=6),
        end_time: datetime | int | None = None,
    ) -> dict[str, list[dict[str, object]]]:
        pair = pair.upper()
        end_ms = self._to_ms(datetime.now(timezone.utc) if end_time is None else end_time)
        if pair == "BTCUSDT":
            return {"BTCUSDT": self.fetch_last_7_days(pair, margin, end_ms)}
        return {
            pair: self.fetch_last_7_days(pair, margin, end_ms),
            "BTCUSDT": self.fetch_last_7_days("BTCUSDT", margin, end_ms),
        }

    def _request(self, query: str) -> list[list[object]]:
        url = f"{self.base_url}?{query}"
        for attempt in range(self.retries):
            request = Request(url, headers={"Accept": "application/json", "User-Agent": "crypto-predictor"})
            try:
                with self.opener(request, timeout=self.timeout_seconds) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                if not isinstance(payload, list):
                    raise BinanceRestError("Binance returned a non-array klines response")
                return payload
            except HTTPError as error:
                retryable = error.code == 429 or error.code == 418 or 500 <= error.code < 600
                if not retryable or attempt == self.retries - 1:
                    raise BinanceRestError(f"Binance HTTP {error.code}: {error.reason}") from error
            except (URLError, TimeoutError, json.JSONDecodeError) as error:
                if attempt == self.retries - 1:
                    raise BinanceRestError("Binance REST request failed") from error
            self.sleep(0.25 * (2 ** attempt))
        raise BinanceRestError("Binance REST request failed")

    @staticmethod
    def _to_ms(value: datetime | int) -> int:
        if isinstance(value, datetime):
            if value.tzinfo is None:
                value = value.replace(tzinfo=timezone.utc)
            return int(value.timestamp() * 1000)
        if isinstance(value, (int, float)) and math.isfinite(value):
            return int(value)
        raise ValueError("time value must be a timezone-aware datetime or millisecond integer")


_default_client = BinanceKlineClient()


def fetch_klines(
    symbol: str,
    start_time: datetime | int,
    end_time: datetime | int,
) -> list[dict[str, object]]:
    return _default_client.fetch_klines(symbol, start_time, end_time)


def fetch_last_7d_klines(
    symbol: str,
    margin: timedelta = timedelta(hours=6),
    end_time: datetime | int | None = None,
) -> list[dict[str, object]]:
    return _default_client.fetch_last_7_days(symbol, margin, end_time)


def fetch_pair_klines(
    pair: str,
    margin: timedelta = timedelta(hours=6),
    end_time: datetime | int | None = None,
) -> dict[str, list[dict[str, object]]]:
    return _default_client.fetch_pair(pair, margin, end_time)


def _klines_frame(
    klines: list[dict[str, object]],
    symbol: str,
    now_ms: int,
) -> pd.DataFrame:
    if not klines:
        raise LiveFeatureError(f"no 1-minute klines for {symbol}")

    frame = pd.DataFrame(klines).copy()
    required = set(("open_time", "open", "high", "low", "close", "volume",
                    "quote_asset_volume", "number_of_trades",
                    "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume"))
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise LiveFeatureError(f"malformed {symbol} klines: missing {', '.join(missing)}")

    numeric = sorted(required)
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame[numeric].isna().any().any():
        raise LiveFeatureError(f"{symbol} klines contain missing numeric values")

    frame["open_time"] = frame["open_time"].astype("int64")
    if "close_time" in frame:
        frame["close_time"] = pd.to_numeric(frame["close_time"], errors="coerce")
        frame = frame[
            ((frame["close_time"].notna()) & (frame["close_time"] < now_ms))
            | ((frame["close_time"].isna()) & (frame["open_time"] + 59_999 < now_ms))
        ]
    else:
        current_minute = now_ms // _OPEN_BAR_MARGIN_MS * _OPEN_BAR_MARGIN_MS
        frame = frame[frame["open_time"] < current_minute]

    if frame.empty:
        raise LiveFeatureError(f"{symbol} has no closed 1-minute bars")

    frame = frame.drop_duplicates("open_time", keep="last")
    frame = frame.sort_values("open_time").set_index(
        pd.to_datetime(frame["open_time"], unit="ms", utc=True)
    )
    frame.index.name = "open_time"
    frame = frame.drop(columns=["open_time"])
    frame = frame.drop(columns=["close_time"], errors="ignore")
    return frame


def _check_live_gaps(
    frame: pd.DataFrame,
    symbol: str,
    now_ms: int,
) -> None:
    times = sorted(int(value.timestamp() * 1000) for value in frame.index)
    if not times:
        raise LiveFeatureError(f"no closed 1-minute bars for {symbol}")

    latest = times[-1]
    if now_ms - latest > _STALE_AFTER_MS:
        minutes = (now_ms - latest) / 60_000
        raise LiveFeatureError(f"stale {symbol} data: last closed bar is {minutes:.1f} minutes old")

    bar_grid = set(
        timestamp // 900_000 * 900_000
        for timestamp in times
    )
    for previous, current in zip(times, times[1:]):
        gap = current - previous
        if gap > _MAX_15M_GAP_MS:
            raise LiveFeatureError(f"gap in {symbol} data: {gap / 60_000:.1f} minutes")

    if len(bar_grid) < 2:
        raise LiveFeatureError(f"insufficient closed bars for {symbol}")


def build_live_feature_row(
    pair: str,
    pair_klines: list[dict[str, object]],
    btc_klines: list[dict[str, object]] | None = None,
    now: datetime | int | None = None,
) -> dict[str, float | int]:
    """Drop open bars, validate freshness/gaps, and return the last safe 36-column row."""
    pair = pair.upper()
    now_ms = _default_client._to_ms(datetime.now(timezone.utc) if now is None else now)

    pair_1m = _klines_frame(pair_klines, pair, now_ms)
    _check_live_gaps(pair_1m, pair, now_ms)

    if pair == "BTCUSDT":
        btc_1m = pair_1m
    else:
        if btc_klines is None:
            raise LiveFeatureError("BTCUSDT klines are required for non-BTC pairs")
        btc_1m = _klines_frame(btc_klines, "BTCUSDT", now_ms)
        _check_live_gaps(btc_1m, "BTCUSDT", now_ms)

    if clean_symbol is None:
        raise LiveFeatureError("feature pipeline is unavailable")

    pair_bars, _ = clean_symbol(pair_1m.reset_index())
    pair_bars = pair_bars.assign(symbol=pair)
    pair_bars = pair_bars.reset_index().rename(columns={"index": "open_time"})

    btc_bars, _ = clean_symbol(btc_1m.reset_index())
    btc_bars = btc_bars.reset_index().rename(columns={"index": "open_time"})
    btc_bars["symbol"] = "BTCUSDT"

    pair_features = add_features(pair_bars.set_index("open_time"))
    btc_features = add_features(btc_bars.set_index("open_time"))

    pair_features = pair_features.reset_index().assign(symbol=pair)
    btc_features = btc_features.reset_index().assign(symbol="BTCUSDT")
    market = add_market_features(pd.concat(
        [pair_features, btc_features],
        ignore_index=True,
    ))
    pair_market = market.loc[market["symbol"] == pair].copy()
    if pair_market.empty:
        raise LiveFeatureError(f"no completed 15-minute bar for {pair}")

    last = pair_market.iloc[-1]
    row: dict[str, float | int] = {}
    missing = []
    for name in FEATURES:
        if name not in pair_market.columns:
            missing.append(name)
            continue
        value = last[name]
        if pd.isna(value) or not np.isfinite(float(value)):
            missing.append(name)
            continue
        row[name] = float(value) if name not in [f"symbol={symbol}" for symbol in (
            "BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT",
            "DOGEUSDT", "SOLUSDT", "LTCUSDT", "LINKUSDT", "AVAXUSDT",
        )] else int(value)

    if missing:
        raise LiveFeatureError(
            f"last closed bar has missing or non-finite features: {', '.join(missing)}"
        )
    return row


def get_live_feature_row(
    pair: str,
    pair_klines: list[dict[str, object]],
    btc_klines: list[dict[str, object]] | None = None,
    now: datetime | int | None = None,
) -> dict[str, float | int]:
    return build_live_feature_row(pair, pair_klines, btc_klines, now)


def fetch_live_feature_row(
    pair: str,
    margin: timedelta = timedelta(hours=6),
    end_time: datetime | int | None = None,
) -> dict[str, float | int]:
    klines = fetch_pair_klines(pair, margin, end_time)
    btc = klines.get("BTCUSDT")
    return build_live_feature_row(
        pair,
        klines[pair.upper()],
        btc,
        end_time,
    )
