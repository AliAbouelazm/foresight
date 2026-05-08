"""
Train TemporalCNN or TransformerPredictor on preprocessed trajectory windows.
Saves best checkpoint and training history JSON.
"""
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from app.model.architecture import build_model

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
MODEL_DIR = Path(__file__).parent / "checkpoints"
def _history_path(model_name: str) -> Path:
    return Path(__file__).parent / f"{model_name}_training_history.json"

EPOCHS = 100
BATCH_SIZE = 128
LR = 5e-4
PATIENCE = 12


def _load_split(name: str) -> TensorDataset:
    data = torch.load(PROCESSED_DIR / f"{name}.pt", weights_only=True)
    return TensorDataset(data["X"], data["y"])


def train(model_name: str = "transformer") -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training {model_name} on {device}")

    train_ds = _load_split("train")
    val_ds = _load_split("val")
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

    model = build_model(model_name).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = nn.HuberLoss()

    best_val_loss = float("inf")
    patience_counter = 0
    history = {"model": model_name, "train_loss": [], "val_loss": [], "epochs": []}

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()

        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            pred = model(X_batch, target=y_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * len(X_batch)
        train_loss /= len(train_ds)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                pred = model(X_batch)   # autoregressive, no teacher forcing
                val_loss += criterion(pred, y_batch).item() * len(X_batch)
        val_loss /= len(val_ds)

        scheduler.step(val_loss)
        elapsed = time.time() - t0

        history["train_loss"].append(round(train_loss, 6))
        history["val_loss"].append(round(val_loss, 6))
        history["epochs"].append(epoch)

        print(
            f"Epoch {epoch:3d}/{EPOCHS}  "
            f"train={train_loss:.6f}  val={val_loss:.6f}  "
            f"lr={optimizer.param_groups[0]['lr']:.2e}  {elapsed:.1f}s"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(
                {"epoch": epoch, "model_state": model.state_dict(), "val_loss": val_loss},
                MODEL_DIR / f"{model_name}_best.pt",
            )
            print(f"  Saved best checkpoint (val={val_loss:.6f})")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break

    history["best_val_loss"] = best_val_loss
    hp = _history_path(model_name)
    hp.write_text(json.dumps(history, indent=2))
    print(f"\nTraining complete. Best val loss: {best_val_loss:.6f}")
    print(f"History saved to {hp}")


if __name__ == "__main__":
    import sys

    model_name = sys.argv[1] if len(sys.argv) > 1 else "transformer"
    train(model_name)
