"""
Convert raw trajectories into sliding window training samples and save as .pt files.
"""
from pathlib import Path

import numpy as np
import torch

PROCESSED_DIR = Path(__file__).parent / "processed"

INPUT_LEN = 50
TARGET_LEN = 50
SEQ_LEN = INPUT_LEN + TARGET_LEN  # 100 frames total
STRIDE = 25  # 75% overlap for more windows


def _compute_features(coords: np.ndarray) -> np.ndarray:
    """
    Convert (T, 2) coordinate array to (T, 4) feature array.
    Features per timestep: x, y, dx, dy. Velocity is zero-padded at t=0.
    """
    T = len(coords)
    features = np.zeros((T, 4), dtype=np.float32)
    features[:, :2] = coords
    features[1:, 2] = coords[1:, 0] - coords[:-1, 0]
    features[1:, 3] = coords[1:, 1] - coords[:-1, 1]
    return features


def build_windows(trajectories: list) -> tuple[np.ndarray, np.ndarray]:
    """
    Slide a 100-frame window across each trajectory with STRIDE overlap.
    Input window: first 50 frames as (50, 4) features.
    Target window: next 50 frames as (50, 2) coordinates.
    """
    inputs_list = []
    targets_list = []

    for traj in trajectories:
        if len(traj) < SEQ_LEN:
            continue

        features = _compute_features(traj)

        for start in range(0, len(features) - SEQ_LEN + 1, STRIDE):
            inp = features[start : start + INPUT_LEN]
            tgt = traj[start + INPUT_LEN : start + SEQ_LEN]
            inputs_list.append(inp)
            targets_list.append(tgt)

    if not inputs_list:
        raise RuntimeError("No training windows could be built. Trajectories may be too short.")

    inputs = np.array(inputs_list, dtype=np.float32)
    targets = np.array(targets_list, dtype=np.float32)
    return inputs, targets


def split_and_save(trajectories: list) -> None:
    """
    Build windows from all trajectories, shuffle, split 70/15/15, and save .pt files.
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    inputs, targets = build_windows(trajectories)
    N = len(inputs)

    rng = np.random.default_rng(42)
    idx = rng.permutation(N)
    inputs = inputs[idx]
    targets = targets[idx]

    n_train = int(N * 0.70)
    n_val = int(N * 0.15)

    splits = {
        "train": (inputs[:n_train], targets[:n_train]),
        "val": (inputs[n_train : n_train + n_val], targets[n_train : n_train + n_val]),
        "test": (inputs[n_train + n_val :], targets[n_train + n_val :]),
    }

    print(f"\nTotal windows: {N}")
    for name, (X, y) in splits.items():
        torch.save({"X": torch.tensor(X), "y": torch.tensor(y)}, PROCESSED_DIR / f"{name}.pt")
        print(f"  {name}: {len(X)} samples")

    x_all = inputs
    print(
        f"\nCoordinate range:"
        f"  x=[{x_all[:,:,0].min():.4f}, {x_all[:,:,0].max():.4f}]"
        f"  y=[{x_all[:,:,1].min():.4f}, {x_all[:,:,1].max():.4f}]"
    )


if __name__ == "__main__":
    from app.data.loader import load_trajectories

    trajs = load_trajectories()
    split_and_save(trajs)
