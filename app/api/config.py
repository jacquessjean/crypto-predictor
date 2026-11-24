import os
from dataclasses import dataclass

SUPPORTED_PAIRS: tuple[str, ...] = (
    "BTCUSDT",
    "ETHUSDT",
    "BNBUSDT",
    "XRPUSDT",
    "ADAUSDT",
    "DOGEUSDT",
    "SOLUSDT",
    "LTCUSDT",
    "LINKUSDT",
    "AVAXUSDT",
)

ROUND_TRIP_FEE: float = 0.002
BAR_SECONDS: int = 15 * 60          # 15-minute bars (900s)
HORIZON_SECONDS: int = 60 * 60      # 1-hour prediction horizon (3600s)


@dataclass(frozen=True)
class Settings:
    model_url: str = os.getenv("MODEL_URL", "http://localhost:9000").rstrip("/")
    db_path: str = os.getenv("DB_PATH", "appdata/predictions.db")
    request_timeout_seconds: float = float(os.getenv("MODEL_TIMEOUT_SECONDS", "10.0"))
    resolver_interval_seconds: int = int(os.getenv("RESOLVER_INTERVAL_SECONDS", "120"))


settings = Settings()