"""
Inference utilities: single prediction with Monte Carlo dropout uncertainty.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from app.model.architecture import build_model

MODEL_DIR = Path(__file__).parent / "checkpoints"
def _history_path(model_name: str) -> Path:
    return Path(__file__).parent / f"{model_name}_training_history.json"
MC_PASSES = 10

_cache: dict[str, torch.nn.Module] = {}


def _load(model_name: str) -> torch.nn.Module:
    if model_name in _cache:
        return _cache[model_name]

    ckpt_path = MODEL_DIR / f"{model_name}_best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No checkpoint found at {ckpt_path}")

    model = build_model(model_name)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model_state"])
    _cache[model_name] = model
    return model


def predict(
    input_seq: np.ndarray,
    model_name: str = "transformer",
    mc_passes: int = MC_PASSES,
) -> dict:
    """
    Run Monte Carlo dropout prediction on a single (20, 4) input sequence.

    Returns dict with:
      - mean: (20, 2) mean predicted trajectory
      - std: (20, 2) per-timestep std across MC passes (epistemic uncertainty)
      - samples: (mc_passes, 20, 2) all raw MC samples
    """
    model = _load(model_name)
    x = torch.tensor(input_seq, dtype=torch.float32).unsqueeze(0)   # (1, 20, 4)

    # MC dropout: keep dropout active during inference
    model.train()
    with torch.no_grad():
        samples = torch.cat([model(x) for _ in range(mc_passes)], dim=0)   # (MC, 20, 2)
    model.eval()

    mean = samples.mean(dim=0).numpy()     # (20, 2)
    std = samples.std(dim=0).numpy()       # (20, 2)

    return {
        "mean": mean.tolist(),
        "std": std.tolist(),
        "samples": samples.numpy().tolist(),
    }


def get_training_history(model_name: str = "transformer") -> Optional[dict]:
    p = _history_path(model_name)
    if p.exists():
        return json.loads(p.read_text())
    return None


def model_is_trained(model_name: str = "transformer") -> bool:
    return (MODEL_DIR / f"{model_name}_best.pt").exists()
