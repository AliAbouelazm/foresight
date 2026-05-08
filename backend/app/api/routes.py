"""
API routes for trajectory prediction, training history, metrics, and sample data.
"""
import random
from pathlib import Path

import numpy as np
import torch
from fastapi import APIRouter, HTTPException

from app.api.schemas import (
    HealthResponse,
    MetricsResponse,
    PredictRequest,
    PredictResponse,
    SampleTrajectory,
    TrajectoryPoint,
    TrainingHistoryResponse,
)
from app.model.predict import get_training_history, model_is_trained, predict

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        models_trained={
            "cnn": model_is_trained("cnn"),
            "transformer": model_is_trained("transformer"),
        },
    )


@router.post("/predict", response_model=PredictResponse)
def predict_trajectory(req: PredictRequest):
    if not model_is_trained(req.model_name):
        raise HTTPException(
            status_code=503,
            detail=f"Model '{req.model_name}' has not been trained yet.",
        )

    if len(req.input_sequence) != 50 or any(len(row) != 4 for row in req.input_sequence):
        raise HTTPException(
            status_code=422,
            detail="input_sequence must be shape (50, 4).",
        )

    input_arr = np.array(req.input_sequence, dtype=np.float32)
    result = predict(input_arr, model_name=req.model_name, mc_passes=req.mc_passes)

    return PredictResponse(
        mean=result["mean"],
        std=result["std"],
        samples=result["samples"],
        model_name=req.model_name,
    )


@router.get("/model/training-history", response_model=TrainingHistoryResponse)
def training_history(model_name: str = "cnn"):
    history = get_training_history(model_name)
    if history is None:
        raise HTTPException(status_code=404, detail=f"No training history found for '{model_name}'.")
    return TrainingHistoryResponse(**history)


@router.get("/model/metrics", response_model=MetricsResponse)
def model_metrics(model_name: str = "transformer"):
    if model_name not in ("cnn", "transformer"):
        raise HTTPException(status_code=422, detail="model_name must be 'cnn' or 'transformer'.")

    if not model_is_trained(model_name):
        raise HTTPException(
            status_code=503,
            detail=f"Model '{model_name}' has not been trained yet.",
        )

    from app.model.evaluate import evaluate
    metrics = evaluate(model_name)
    return MetricsResponse(**metrics)


@router.get("/trajectories/samples/{sample_id}", response_model=SampleTrajectory)
def get_sample_trajectory(sample_id: int):
    test_path = PROCESSED_DIR / "test.pt"
    if not test_path.exists():
        raise HTTPException(status_code=503, detail="Processed data not available.")

    data = torch.load(test_path, weights_only=True)
    n = len(data["X"])

    if sample_id < 0 or sample_id >= n:
        raise HTTPException(
            status_code=404,
            detail=f"Sample ID {sample_id} out of range [0, {n - 1}].",
        )

    x = data["X"][sample_id].numpy()   # (20, 4)
    y = data["y"][sample_id].numpy()   # (20, 2)

    return SampleTrajectory(
        id=sample_id,
        input=[TrajectoryPoint(x=float(row[0]), y=float(row[1])) for row in x],
        target=[TrajectoryPoint(x=float(row[0]), y=float(row[1])) for row in y],
    )


@router.get("/trajectories/random", response_model=SampleTrajectory)
def get_random_trajectory():
    test_path = PROCESSED_DIR / "test.pt"
    if not test_path.exists():
        raise HTTPException(status_code=503, detail="Processed data not available.")

    data = torch.load(test_path, weights_only=True)
    n = len(data["X"])
    return get_sample_trajectory(random.randint(0, n - 1))
