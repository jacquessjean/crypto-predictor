from typing import Literal
from pydantic import BaseModel, Field, field_validator


ActionType = Literal["BUY", "SELL", "WAIT"]
OutcomeStatus = Literal["pending", "correct", "incorrect", "flat", "neutral"]


class PredictRequest(BaseModel):
    pair: str = Field(..., examples=["ETHUSDT"])

    @field_validator("pair")
    @classmethod
    def normalize_pair(cls, value: str) -> str:
        return value.strip().upper()


class PredictionResponse(BaseModel):
    id: int
    pair: str
    request_time: str
    bar_time: str
    expires_at: str
    p_up: float
    p_flat: float
    p_down: float
    action: ActionType
    entry_price: float
    model_version: str
    schema_version: str
    status: OutcomeStatus = "pending"
    exit_price: float | None = None
    realized_return: float | None = None
    net_return: float | None = None
    resolved_at: str | None = None


class CandleItem(BaseModel):
    open_time: int
    open: float
    high: float
    low: float
    close: float
    volume: float


class ActionStats(BaseModel):
    action: ActionType
    total: int
    resolved: int
    profitable_count: int
    win_rate: float | None
    avg_net_return: float | None
    avg_realized_return: float | None


class StatsResponse(BaseModel):
    total_predictions: int
    resolved_predictions: int
    pending_predictions: int
    resolved_signals: int
    signal_win_rate: float | None
    signal_avg_net_return: float | None
    by_action: list[ActionStats]