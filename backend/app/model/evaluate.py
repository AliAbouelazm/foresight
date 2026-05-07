"""
Evaluate a trained model on the test split.
Computes ADE (Average Displacement Error) and FDE (Final Displacement Error).
"""
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from app.model.architecture import build_model

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
MODEL_DIR = Path(__file__).parent / "checkpoints"


def _ade(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Mean L2 distance across all timesteps and samples."""
    dist = torch.norm(pred - target, dim=2)   # (B, T)
    return dist.mean().item()


def _fde(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Mean L2 distance at the final timestep."""
    dist = torch.norm(pred[:, -1, :] - target[:, -1, :], dim=1)   # (B,)
    return dist.mean().item()


def evaluate(model_name: str = "transformer") -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = torch.load(PROCESSED_DIR / "test.pt", weights_only=True)
    test_ds = TensorDataset(data["X"], data["y"])
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False)

    model = build_model(model_name).to(device)
    ckpt = torch.load(MODEL_DIR / f"{model_name}_best.pt", map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    all_preds, all_targets = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            all_preds.append(model(X_batch).cpu())
            all_targets.append(y_batch)

    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets)

    metrics = {
        "model": model_name,
        "n_test": len(test_ds),
        "ade": round(_ade(preds, targets), 6),
        "fde": round(_fde(preds, targets), 6),
        "best_epoch": ckpt.get("epoch"),
        "best_val_loss": round(float(ckpt.get("val_loss", 0)), 6),
    }

    print(f"\nTest metrics for {model_name}:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    return metrics


if __name__ == "__main__":
    import sys

    model_name = sys.argv[1] if len(sys.argv) > 1 else "transformer"
    evaluate(model_name)
