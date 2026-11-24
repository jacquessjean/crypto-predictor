import json
from pathlib import Path

import numpy as np
import pandas as pd

from cleanup import FEATURES, HORIZON, NUMERIC_FEATURES


DELTA = 0.002
PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURES_PATH = PROJECT_ROOT / "out" / "processed" / "features.parquet"
SCALER_PATH = PROJECT_ROOT / "out" / "processed" / "scaler.json"
SCHEMA_PATH = PROJECT_ROOT / "out" / "processed" / "schema.json"


def build_three_class_label(df: pd.DataFrame) -> pd.DataFrame:
    """Add the three-class target: down (0), flat (-1), or up (1)."""
    if "fwd_ret" not in df.columns:
        raise ValueError("features.parquet must contain a 'fwd_ret' column")

    df = df.copy()
    df["label"] = np.select(
        [df["fwd_ret"] > DELTA, df["fwd_ret"] < -DELTA],
        [1, 0],
        default=-1,
    )
    return df


def fit_scaler(df: pd.DataFrame) -> dict:
    """Fit clipping bounds and standardization statistics on every training row."""
    train = df.loc[df["split"] == "train", NUMERIC_FEATURES]
    if train.empty:
        raise ValueError("No rows have split='train'")

    clip_low = train.quantile(0.001)
    clip_high = train.quantile(0.999)
    clipped = train.clip(clip_low, clip_high, axis=1)
    mean = clipped.mean()
    std = clipped.std()

    if (std == 0).any():
        zero_variance = std.index[std == 0].tolist()
        raise ValueError(f"Zero-variance numeric features: {zero_variance}")

    return {
        "clip_low": clip_low.to_dict(),
        "clip_high": clip_high.to_dict(),
        "mean": mean.to_dict(),
        "std": std.to_dict(),
        "features": FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "horizon": HORIZON,
        "delta": DELTA,
    }


def main() -> None:
    df = pd.read_parquet(FEATURES_PATH)
    df = build_three_class_label(df)

    scaler = fit_scaler(df)
    SCALER_PATH.parent.mkdir(parents=True, exist_ok=True)
    with SCALER_PATH.open("w") as file:
        json.dump(scaler, file, indent=1)
    with SCHEMA_PATH.open("w") as file:
        json.dump({"features": FEATURES}, file, indent=2)

    print(df["label"].value_counts().sort_index().rename(index={-1: "flat", 0: "down", 1: "up"}).to_string())


if __name__ == "__main__":
    main()