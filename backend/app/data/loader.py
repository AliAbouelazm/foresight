"""
Load multi-object tracking annotations from SportsMOT or DanceTrack.
Falls back through multiple sources until one succeeds.
"""
import configparser
import os
from pathlib import Path

import numpy as np

CACHE_DIR = Path(__file__).parent / "raw_cache"
MIN_TRACK_LEN = 100
MAX_FRAME_GAP = 5


def _parse_mot_file(gt_path: str, im_width: int, im_height: int) -> list:
    """
    Parse a MOT-format annotation file.
    Format: frame_id, track_id, left, top, width, height, ...
    Returns list of (T, 2) arrays of normalized (cx, cy) per track.
    """
    tracks: dict[int, dict[int, tuple]] = {}

    with open(gt_path, "r") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 6:
                continue
            try:
                frame_id = int(parts[0])
                track_id = int(parts[1])
                left = float(parts[2])
                top = float(parts[3])
                w = float(parts[4])
                h = float(parts[5])
            except ValueError:
                continue

            cx = (left + w / 2.0) / im_width
            cy = (top + h / 2.0) / im_height
            cx = float(np.clip(cx, 0.0, 1.0))
            cy = float(np.clip(cy, 0.0, 1.0))

            if track_id not in tracks:
                tracks[track_id] = {}
            tracks[track_id][frame_id] = (cx, cy)

    trajectories = []
    for track_id, frames in tracks.items():
        if len(frames) < MIN_TRACK_LEN:
            continue

        sorted_frames = sorted(frames.keys())
        gaps = [sorted_frames[i + 1] - sorted_frames[i] for i in range(len(sorted_frames) - 1)]
        if gaps and max(gaps) > MAX_FRAME_GAP:
            continue

        coords = np.array([frames[f] for f in sorted_frames], dtype=np.float32)
        trajectories.append(coords)

    return trajectories


def _read_seqinfo(seqinfo_path: Path) -> tuple[int, int]:
    """Read imWidth and imHeight from seqinfo.ini."""
    config = configparser.ConfigParser()
    config.read(str(seqinfo_path))
    try:
        w = int(config["Sequence"]["imWidth"])
        h = int(config["Sequence"]["imHeight"])
        return w, h
    except (KeyError, ValueError):
        return 1920, 1080


def _parse_local_dir(local_dir: Path) -> list:
    """Walk a local directory tree and parse all gt.txt annotation files."""
    all_trajectories = []

    for gt_path in sorted(local_dir.rglob("gt.txt")):
        seqinfo_path = gt_path.parent.parent / "seqinfo.ini"
        if seqinfo_path.exists():
            im_w, im_h = _read_seqinfo(seqinfo_path)
        else:
            im_w, im_h = 1920, 1080

        trajs = _parse_mot_file(str(gt_path), im_w, im_h)
        all_trajectories.extend(trajs)

    return all_trajectories


def _try_hf_dataset(repo_id: str) -> list | None:
    """
    Download annotation files only from a HuggingFace dataset repo.
    Skips images so only text files are fetched.
    """
    try:
        from huggingface_hub import snapshot_download, list_repo_files

        print(f"Trying HuggingFace dataset: {repo_id}")
        all_files = list(list_repo_files(repo_id, repo_type="dataset"))
        gt_files = [f for f in all_files if f.endswith("gt.txt")]

        if not gt_files:
            print(f"  No gt.txt files found in {repo_id}")
            return None

        print(f"  Found {len(gt_files)} annotation sequences")

        local_dir = CACHE_DIR / repo_id.replace("/", "_")
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=str(local_dir),
            ignore_patterns=["*.jpg", "*.jpeg", "*.png", "*.mp4", "*.avi", "*.mov"],
        )

        trajs = _parse_local_dir(local_dir)
        print(f"  Extracted {len(trajs)} valid trajectories")
        return trajs if trajs else None

    except Exception as e:
        print(f"  Failed ({repo_id}): {e}")
        return None


def _try_github_sportsmot() -> list | None:
    """
    Attempt to download SportsMOT annotation archives from the official GitHub release.
    These are annotation-only zips, no video files.
    """
    import io
    import zipfile

    import requests

    candidates = [
        "https://github.com/MCG-NJU/SportsMOT/releases/download/v1.0/sportsmot_publish.zip",
        "https://github.com/MCG-NJU/SportsMOT/archive/refs/heads/main.zip",
    ]

    for url in candidates:
        print(f"Trying GitHub fallback: {url}")
        try:
            resp = requests.get(url, timeout=120, stream=True)
            if resp.status_code != 200:
                continue

            local_dir = CACHE_DIR / "github_sportsmot"
            local_dir.mkdir(parents=True, exist_ok=True)

            with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                for name in zf.namelist():
                    if name.endswith("gt.txt") or name.endswith("seqinfo.ini"):
                        zf.extract(name, str(local_dir))

            trajs = _parse_local_dir(local_dir)
            if trajs:
                print(f"  Extracted {len(trajs)} valid trajectories")
                return trajs
        except Exception as e:
            print(f"  Failed: {e}")
            continue

    return None


def load_trajectories() -> list:
    """
    Load player tracking trajectories, trying multiple sources in priority order:
    1. Local cache
    2. MCG-NJU/SportsMOT on HuggingFace
    3. noahcao/dancetrack on HuggingFace
    4. SportsMOT GitHub release

    Returns list of numpy arrays, each shape (T, 2), normalized coordinates.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    cache_file = CACHE_DIR / "trajectories.npy"
    if cache_file.exists():
        print("Loading cached trajectories...")
        data = np.load(str(cache_file), allow_pickle=True)
        trajs = list(data)
        print(f"Loaded {len(trajs)} cached trajectories")
        return trajs

    trajectories = _try_hf_dataset("MCG-NJU/SportsMOT")

    if not trajectories:
        trajectories = _try_hf_dataset("noahcao/dancetrack")

    if not trajectories:
        trajectories = _try_github_sportsmot()

    if not trajectories:
        raise RuntimeError(
            "Could not load trajectory data from any source. "
            "Check your internet connection or set HF_TOKEN for gated datasets."
        )

    np.save(str(cache_file), np.array(trajectories, dtype=object))
    print(f"\nTotal valid trajectories: {len(trajectories)}")
    return trajectories
