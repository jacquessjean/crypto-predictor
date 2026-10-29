import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import urlretrieve

import numpy as np
import pandas as pd

URL = "https://data.binance.vision/data/spot/monthly/klines/{s}/1m/{s}-1m-{m}.zip"
RAW = Path("out/raw")
OUT = Path("out/processed")
COLUMNS = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_asset_volume",
           "number_of_trades", "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"]

# Large-cap pairs that trade through the whole period (MATIC was renamed to POL in 2024, so it is left out).
SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT", "SOLUSDT", "LTCUSDT", "LINKUSDT", "AVAXUSDT"]
FIRST_MONTH = "2020-01"
STEP = "15min" # one sample every 15 minutes: 1m rows would share 59 of their 60 future minutes
HORIZON = 4 # predict 4 steps (1 hour) ahead
DELTA = 0.002 # |move| <= 0.2% (Binance round-trip taker fee) counts as flat and is dropped
VAL_START = pd.Timestamp("2024-10-01") # one cutoff for every pair: no cross-asset leakage
TEST_START = pd.Timestamp("2025-07-01")

FEATURES = [
    "symbol=BTCUSDT", "symbol=ETHUSDT", "symbol=BNBUSDT", "symbol=XRPUSDT", "symbol=ADAUSDT",
    "symbol=DOGEUSDT", "symbol=SOLUSDT", "symbol=LTCUSDT", "symbol=LINKUSDT", "symbol=AVAXUSDT",
    "returns", "volatility", "upper_wick", "lower_wick", "body_ratio",
    "taker_buy_ratio", "taker_buy_ratio_16", "relative_volume", "normalized_volatility",
    "rel_trade_size", "ret_4", "ret_16", "ret_96", "ret_672", "vol_96",
    "rv_15", "rv_60", "rv_ratio", "idle_minutes",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "btc_ret_4", "btc_ret_16", "rel_ret_4",
]
SYMBOL_FEATURES = [f"symbol={sym}" for sym in SYMBOLS]
NUMERIC_FEATURES = [f for f in FEATURES if f not in SYMBOL_FEATURES]


def _month_zip(sym: str, month: str) -> Path | None:
    path = RAW / sym / f"{sym}-1m-{month}.zip"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            urlretrieve(URL.format(s=sym, m=month), path)
        except HTTPError:  # month before listing, or not published yet
            return None
    return path


def load_symbol(sym: str) -> pd.DataFrame:
    months = pd.period_range(FIRST_MONTH, pd.Timestamp.now() - pd.offsets.MonthBegin(1), freq="M").astype(str)
    with ThreadPoolExecutor(8) as pool:
        paths = [p for p in pool.map(lambda m: _month_zip(sym, m), months) if p]
    d = pd.concat([pd.read_csv(p, header=None, names=COLUMNS) for p in paths], ignore_index=True)
    t = d["open_time"]
    d["open_time"] = pd.to_datetime(np.where(t > 1e14, t // 1000, t), unit="ms")  # files switched to microseconds in 2025
    return d.drop(columns=["ignore", "close_time"])


def clean_symbol(d: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """1m klines -> 15m bars plus intra-bar features that only 1m data can give."""
    d = d.drop_duplicates("open_time").set_index("open_time").sort_index()
    idle = d["volume"] == 0
    log = {"rows_1m": len(d), "zero_volume_1m": int(idle.sum()), "start": d.index[0], "end": d.index[-1]}
    # A minute with no trades has no real price; treat it as missing rather than a flat candle.
    d.loc[idle, d.columns] = np.nan
    d = d.reindex(pd.date_range(d.index[0], d.index[-1], freq="1min"))
    log["missing_minutes"] = int(d["close"].isna().sum())

    r1 = np.log(d["close"]).diff()
    bars = d.resample(STEP).agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum",
                                 "quote_asset_volume": "sum", "number_of_trades": "sum",
                                 "taker_buy_base_asset_volume": "sum", "taker_buy_quote_asset_volume": "sum"})
    minutes = d["close"].resample(STEP).count()
    bars.loc[minutes == 0] = np.nan  # no trade at all in the 15 minutes: missing bar, not a zero-volume one
    bars["rv_15"] = r1.resample(STEP).std()                                     # realized vol inside the bar
    bars["rv_60"] = r1.rolling(60, min_periods=30).std().resample(STEP).last()  # over the last hour
    bars["idle_minutes"] = 1 - minutes / 15
    log["bars"] = int(bars["close"].notna().sum())
    return bars, log


def roll(x: pd.Series, w: int):
    # Bars are occasionally missing; tolerate up to half the window missing instead of losing w rows per hole.
    return x.rolling(w, min_periods=w // 2)


def add_features(d: pd.DataFrame) -> pd.DataFrame:
    o, h, l, c, v = d["open"], d["high"], d["low"], d["close"], d["volume"]
    f = pd.DataFrame(index=d.index)
    f["returns"] = c / o - 1
    f["volatility"] = (h - l) / c
    f["upper_wick"] = (h - np.maximum(o, c)) / c
    f["lower_wick"] = (np.minimum(o, c) - l) / c
    f["body_ratio"] = ((c - o) / (h - l).replace(0, np.nan)).fillna(0)  # zero-range candle has no body
    f["taker_buy_ratio"] = d["taker_buy_base_asset_volume"] / v
    f["taker_buy_ratio_16"] = roll(d["taker_buy_base_asset_volume"], 16).sum() / roll(v, 16).sum()
    f["relative_volume"] = v / roll(v, 20).mean()
    f["normalized_volatility"] = (f["volatility"] - roll(f["volatility"], 20).mean()) / roll(f["volatility"], 20).std()
    trade_size = d["quote_asset_volume"] / d["number_of_trades"]
    f["rel_trade_size"] = trade_size / roll(trade_size, 96).mean()
    for k in (4, 16, 96, 672):  # 1h, 4h, 1d, 7d momentum
        f[f"ret_{k}"] = c / c.ffill(limit=96).shift(k) - 1  # last real price at or before t-k (past only)
    f["vol_96"] = roll(f["returns"], 96).std()
    f["rv_15"], f["rv_60"], f["idle_minutes"] = d["rv_15"], d["rv_60"], d["idle_minutes"]
    f["rv_ratio"] = d["rv_60"] / roll(d["rv_60"], 96).mean()  # is this hour unusually volatile for the day?
    hour = d.index.hour + d.index.minute / 60
    f["hour_sin"], f["hour_cos"] = np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24)
    f["dow_sin"], f["dow_cos"] = np.sin(2 * np.pi * d.index.dayofweek / 7), np.cos(2 * np.pi * d.index.dayofweek / 7)
    f["fwd_ret"] = c.shift(-HORIZON) / c - 1  # NaN when the future bar is missing, so no fake labels across gaps
    return f


def add_market_features(df: pd.DataFrame) -> pd.DataFrame:
    # Crypto moves with BTC: give every pair the market's recent move and its own move relative to it.
    btc = df.loc[df["symbol"] == "BTCUSDT", ["open_time", "ret_4", "ret_16"]]
    df = df.merge(btc.rename(columns={"ret_4": "btc_ret_4", "ret_16": "btc_ret_16"}), on="open_time", how="left")
    df["rel_ret_4"] = df["ret_4"] - df["btc_ret_4"]
    for sym in SYMBOLS:
        df[f"symbol={sym}"] = (df["symbol"] == sym).astype("int8")
    return df


def build() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    parts, logs = [], []
    for sym in SYMBOLS:
        bars, log = clean_symbol(load_symbol(sym))  # one pair at a time: all 1m data at once would not fit in memory
        logs.append({"symbol": sym, **log})
        parts.append(add_features(bars).assign(symbol=sym))
        print(sym, log)
    pd.DataFrame(logs).to_csv(OUT / "cleaning_log.csv", index=False)

    df = add_market_features(pd.concat(parts).rename_axis("open_time").reset_index())
    rows_before = len(df)
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURES + ["fwd_ret"])
    print(f"dropped {rows_before - len(df)} rows with missing features/target, {len(df)} left")
    df["label"] = np.select([df["fwd_ret"] > DELTA, df["fwd_ret"] < -DELTA], [1, 0], -1)
    df["split"] = np.select([df["open_time"] >= TEST_START, df["open_time"] >= VAL_START], ["test", "val"], "train")
    df.to_parquet(OUT / "features.parquet", index=False)  # unscaled, includes flat rows: used by analysis.py

    df = df[df["label"] != -1]
    train = df[df["split"] == "train"]
    lo, hi = train[NUMERIC_FEATURES].quantile(0.001), train[NUMERIC_FEATURES].quantile(0.999)  # fitted on train only
    clipped = train[NUMERIC_FEATURES].clip(lo, hi, axis=1)
    mean, std = clipped.mean(), clipped.std()
    json.dump({"clip_low": lo.to_dict(), "clip_high": hi.to_dict(), "mean": mean.to_dict(), "std": std.to_dict(),
               "features": FEATURES, "numeric_features": NUMERIC_FEATURES,
               "horizon": HORIZON, "delta": DELTA}, open(OUT / "scaler.json", "w"), indent=1)
    df[NUMERIC_FEATURES] = (df[NUMERIC_FEATURES].clip(lo, hi, axis=1) - mean) / std
    df = df[["open_time"] + FEATURES + ["label", "split"]]
    for name, part in df.groupby("split"):
        part.drop(columns="split").to_parquet(OUT / f"{name}.parquet", index=False)
        print(name, len(part), f"up share {part['label'].mean():.3f}")
    df.to_csv(OUT / "cleaned_dataset.csv", index=False, float_format="%.6g")  # single-file dump for submission


if __name__ == "__main__":
    build()
