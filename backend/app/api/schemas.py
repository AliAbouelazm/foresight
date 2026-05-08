from typing import Optional
from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    input_sequence: list[list[float]] = Field(
        ...,
        description="50 timesteps of [x, y, dx, dy] features, shape (50, 4)",
        min_length=50,
        max_length=50,
    )
    model_name: str = Field(default="transformer", pattern="^(cnn|transformer)$")
    mc_passes: int = Field(default=10, ge=1, le=50)


class PredictResponse(BaseModel):
    mean: list[list[float]]
    std: list[list[float]]
    samples: list[list[list[float]]]
    model_name: str


class TrajectoryPoint(BaseModel):
    x: float
    y: float


class SampleTrajectory(BaseModel):
    id: int
    input: list[TrajectoryPoint]
    target: list[TrajectoryPoint]


class TrainingHistoryResponse(BaseModel):
    model: str
    epochs: list[int]
    train_loss: list[float]
    val_loss: list[float]
    best_val_loss: float


class MetricsResponse(BaseModel):
    model: str
    n_test: int
    ade: float
    fde: float
    best_epoch: Optional[int] = None
    best_val_loss: Optional[float] = None


class HealthResponse(BaseModel):
    status: str
    models_trained: dict[str, bool]
