"""Parity check between the training feature build and the live REST feature path."""
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

from cleanup import (
    COLUMNS,
    FEATURES,
    add_features,
    add_market_features,
    clean_symbol,
)
from app.api.features_live import build_live_feature_row


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = PROJECT_ROOT / "out" / "raw"


def _load_dump(path: Path) -> pd.DataFrame:
    with ZipFile(path) as archive:
        csv_name = next(name for name in archive.namelist() if name.endswith(".csv"))
        with archive.open(csv_name) as csv_file:
            frame = pd.read_csv(csv_file, header=None, names=COLUMNS)
    times = frame["open_time"].to_numpy()
    frame["open_time"] = pd.to_datetime(
        np.where(times > 1e14, times // 1000, times),
        unit="ms",
        utc=True,
    )
    return frame


def _training_row(pair: pd.DataFrame, btc: pd.DataFrame) -> pd.Series:
    pair_bars, _ = clean_symbol(pair.reset_index() if isinstance(pair.index, pd.DatetimeIndex) else pair)
    btc_bars, _ = clean_symbol(btc.reset_index() if isinstance(btc.index, pd.DatetimeIndex) else btc)
    pair_features = add_features(pair_bars).assign(symbol="ETHUSDT")
    btc_features = add_features(btc_bars).assign(symbol="BTCUSDT")
    market = add_market_features(
        pd.concat([pair_features, btc_features])
        .rename_axis("open_time")
        .reset_index()
    )
    return market.loc[market["symbol"] == "ETHUSDT"].iloc[-1]


def _real_raw_pair() -> tuple[Path, Path] | None:
    pair_root = RAW_ROOT / "ETHUSDT"
    btc_root = RAW_ROOT / "BTCUSDT"
    if not pair_root.is_dir() or not btc_root.is_dir():
        return None

    month = lambda path: "-".join(path.stem.split("-")[-2:])  # "ETHUSDT-1m-2026-10" -> "2026-10" (year and month)
    pair_dumps = {month(path): path for path in pair_root.glob("*.zip")}
    btc_dumps = {month(path): path for path in btc_root.glob("*.zip")}
    months = sorted(set(pair_dumps).intersection(btc_dumps), reverse=True)
    return (pair_dumps[months[0]], btc_dumps[months[0]]) if months else None


def _make_synthetic_raw_pair(tmp_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    start = pd.Timestamp("2025-07-01", tz="UTC")
    minutes = pd.date_range(start, periods=31 * 24 * 60 + 1, freq="1min")
    rng = np.random.default_rng(7)

    def frame(symbol: str) -> pd.DataFrame:
        returns = rng.normal(0.00001, 0.0004, len(minutes))
        close = 100.0 * np.exp(np.cumsum(returns))
        high = close * (1.0 + rng.uniform(0.0001, 0.001, len(minutes)))
        low = close * (1.0 - rng.uniform(0.0001, 0.001, len(minutes)))
        open_ = np.r_[close[0], close[:-1]]
        volume = rng.uniform(10.0, 100.0, len(minutes))
        return pd.DataFrame({
            "open_time": (minutes.asi8 // 1_000),
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "close_time": (minutes.asi8 // 1_000) + 59_999,
            "quote_asset_volume": volume * close,
            "number_of_trades": rng.integers(1, 100, len(minutes)),
            "taker_buy_base_asset_volume": volume * 0.6,
            "taker_buy_quote_asset_volume": volume * close * 0.6,
            "ignore": "0",
        }).assign(symbol=symbol)

    pair = frame("ETHUSDT")
    btc = frame("BTCUSDT")
    pair.to_csv(tmp_path / "ETHUSDT.csv", index=False)
    btc.to_csv(tmp_path / "BTCUSDT.csv", index=False)
    with ZipFile(tmp_path / "ETHUSDT-1m-2025-07.zip", "w") as archive:
        archive.write(tmp_path / "ETHUSDT.csv", "ETHUSDT-1m-2025-07.csv")
    with ZipFile(tmp_path / "BTCUSDT-1m-2025-07.zip", "w") as archive:
        archive.write(tmp_path / "BTCUSDT.csv", "BTCUSDT-1m-2025-07.csv")
    return pair, btc


def test_live_training_feature_parity(tmp_path):
    raw_pair = _real_raw_pair()
    if raw_pair is not None:
        pair_path, btc_path = raw_pair
        pair = _load_dump(pair_path)
        btc = _load_dump(btc_path)
        pair["symbol"] = "ETHUSDT"
    else:
        pair, btc = _make_synthetic_raw_pair(tmp_path)
        pair["open_time"] = pd.to_datetime(pair["open_time"], unit="ms", utc=True)
        btc["open_time"] = pd.to_datetime(btc["open_time"], unit="ms", utc=True)

    pair = pair.drop(columns="close_time", errors="ignore")
    btc = btc.drop(columns="close_time", errors="ignore")
    cutoff = pd.Timestamp(
        int(pair.iloc[-1]["open_time"].timestamp() * 1000),
        unit="ms",
        tz="UTC",
    ).floor("15min")  # the live path keeps only closed 15-minute bars, so compare against the same closed bar
    pair = pair[pair["open_time"] < cutoff]
    btc = btc[btc["open_time"] < cutoff]
    pair.attrs["symbol"] = "ETHUSDT"

    pair_training = pair.set_index("open_time") if "open_time" in pair else pair
    btc_training = btc.set_index("open_time") if "open_time" in btc else btc
    expected = _training_row(pair_training, btc_training)
    now_ms = int(cutoff.timestamp() * 1000)

    pair_klines = []
    for row in pair.drop(columns="symbol", errors="ignore").to_dict("records"):
        open_ms = int(pd.Timestamp(row["open_time"]).timestamp() * 1000)
        pair_klines.append({
            **row,
            "open_time": open_ms,
            "close_time": open_ms + 59_999,
        })
    btc_klines = []
    for row in btc.drop(columns="symbol", errors="ignore").to_dict("records"):
        open_ms = int(pd.Timestamp(row["open_time"]).timestamp() * 1000)
        btc_klines.append({
            **row,
            "open_time": open_ms,
            "close_time": open_ms + 59_999,
        })

    live = build_live_feature_row("ETHUSDT", pair_klines, btc_klines, now_ms)
    for name in FEATURES:
        assert np.isclose(live[name], expected[name], rtol=1e-10, atol=1e-12), (
            f"feature mismatch at {name}: live={live[name]}, training={expected[name]}"
        )
