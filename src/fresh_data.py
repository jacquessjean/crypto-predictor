"""Features for data after the training dump (Binance daily files), built with the unchanged cleanup.py functions."""
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import urlretrieve
from zipfile import ZipFile

import numpy as np
import pandas as pd

from cleanup import COLUMNS, SYMBOLS, add_features, add_market_features, clean_symbol

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DAILY_URL = "https://data.binance.vision/data/spot/daily/klines/{s}/1m/{s}-1m-{d}.zip"


def _read(path: Path) -> pd.DataFrame:
    with ZipFile(path) as z:
        d = pd.read_csv(z.open(z.namelist()[0]), header=None, names=COLUMNS)
    t = d["open_time"]
    d["open_time"] = pd.to_datetime(np.where(t > 1e14, t // 1000, t), unit="ms")  # microseconds since 2025
    return d.drop(columns=["ignore", "close_time"])


def load_daily(sym: str, start: str, end: str) -> pd.DataFrame:
    """1-minute klines for start..end (inclusive), downloaded once into out/raw/<SYM>/daily/."""
    folder = PROJECT_ROOT / "out" / "raw" / sym / "daily"
    folder.mkdir(parents=True, exist_ok=True)
    frames = []
    for day in pd.date_range(start, end, freq="D").strftime("%Y-%m-%d"):
        path = folder / f"{sym}-1m-{day}.zip"
        if not path.exists():
            try:
                urlretrieve(DAILY_URL.format(s=sym, d=day), path)
            except HTTPError:
                continue  # not published (yet): a gap the checks downstream will see
        frames.append(_read(path))
    return pd.concat(frames, ignore_index=True)


def build_features(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Same steps and columns as cleanup.build() (open_time, symbol, close, features), for any 1-minute frames."""
    parts = [add_features(clean_symbol(d)[0]).assign(symbol=sym) for sym, d in frames.items()]
    return add_market_features(pd.concat(parts).rename_axis("open_time").reset_index())


def fresh_features(start: str = "2026-10-25", end: str = "2026-11-24") -> pd.DataFrame:
    return build_features({sym: load_daily(sym, start, end) for sym in SYMBOLS})


def cached_fresh(path, build=fresh_features) -> pd.DataFrame:
    """The fresh features at path, built (download included) and saved only the first time."""
    path = Path(path)
    if not path.exists():
        build().to_parquet(path, index=False)
    return pd.read_parquet(path)
