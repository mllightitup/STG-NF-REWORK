"""Data loading, preprocessing, and caching for ShanghaiTech pose data.

Deviations from original:
  - **Caching**: Preprocessed segments saved as memory-mapped .npy files.
    Original parses all JSONs on every run.
  - **Memory-mapped arrays**: Dataset uses np.load(mmap_mode='r') so
    multiprocessing workers share OS page cache without pickling data.
  - **Normalization per-sample**: Done identically to original (per-segment
    center + scale by std_y), but implemented more concisely.
  - **No separate gen_dataset / gen_clip_seg_data_np hierarchy**: Flattened
    into a single `_load_and_segment()` function for clarity.
  - **Test metadata stored alongside segments**: Each test segment carries
    (clip_id, person_id, frame_start) so scoring can assign to frames.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

if TYPE_CHECKING:
    from .config import Config

# ── Constants ────────────────────────────────────────────────────────────────

# AlphaPose 17kp → COCO18 reorder (after appending neck as kp[17])
_COCO18_ORDER: list[int] = [0, 17, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3]


# ── Keypoint helpers ─────────────────────────────────────────────────────────


def _keypoints17_to_coco18(kp: np.ndarray) -> np.ndarray:
    """Convert AlphaPose 17-joint format to COCO18 with synthetic neck.

    Args:
        kp: Array of shape (..., 17, D) where D is typically 3 (x, y, conf).

    Returns:
        Array of shape (..., 18, D) in COCO18 joint order.
    """
    neck = 0.5 * (kp[..., 5, :] + kp[..., 6, :])  # avg shoulders
    kp_with_neck = np.concatenate([kp, neck[..., np.newaxis, :]], axis=-2)
    return kp_with_neck[..., _COCO18_ORDER, :]


# ── JSON parsing & segmentation ─────────────────────────────────────────────


def _parse_single_json(path: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Parse one AlphaPose tracked-person JSON into per-person pose arrays.

    Args:
        path: Path to a JSON file with structure {person_id: {frame_id: {keypoints, scores}}}.

    Returns:
        Dict mapping person_id → (poses (T, 17, 3), frame_ids (T,) int64).
        Frame IDs are actual video frame indices from the JSON.
    """
    with open(path, "r") as f:
        data: dict[str, dict[str, dict]] = json.load(f)

    persons: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for pid, frames in data.items():
        sorted_fids = sorted(frames.keys(), key=int)
        poses = []
        for fid in sorted_fids:
            kp_flat = frames[fid]["keypoints"]
            pose = np.array(kp_flat, dtype=np.float64).reshape(17, 3)
            poses.append(pose)
        frame_ids = np.array([int(f) for f in sorted_fids], dtype=np.int64)
        persons[pid] = (np.stack(poses, axis=0), frame_ids)  # (T, 17, 3), (T,)
    return persons


def _segment_track(
    track: np.ndarray,
    frame_ids: np.ndarray,
    seg_len: int,
    stride: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Split a single-person track into fixed-length overlapping segments.

    Args:
        track: Shape (T, 17, 3).
        frame_ids: Shape (T,) — actual video frame indices.
        seg_len: Segment length (24).
        stride: Sliding window stride.

    Returns:
        (segments (N_segs, seg_len, 17, 3), frame_starts (N_segs,) int64)
        where frame_starts are the actual video frame IDs at the start of each segment.
        Empty arrays if track too short.
    """
    t_total = track.shape[0]
    if t_total < seg_len:
        return (
            np.empty((0, seg_len, 17, 3), dtype=track.dtype),
            np.empty(0, dtype=np.int64),
        )

    valid_starts = []
    missing_th = 2
    for s in range(0, t_total - seg_len + 1, stride):
        start_frame = int(frame_ids[s])
        act_frames = set(int(f) for f in frame_ids[s : s + seg_len])
        expected_frames = set(range(start_frame, start_frame + seg_len))
        if len(act_frames.intersection(expected_frames)) >= (seg_len - missing_th):
            valid_starts.append(s)

    if not valid_starts:
        return (
            np.empty((0, seg_len, 17, 3), dtype=track.dtype),
            np.empty(0, dtype=np.int64),
        )

    starts = np.array(valid_starts, dtype=np.int64)
    segments = np.stack([track[s : s + seg_len] for s in starts], axis=0)
    # Use actual frame IDs from the JSON, not relative indices
    frame_starts = frame_ids[starts]
    return segments, frame_starts


# ── Full dataset preparation ────────────────────────────────────────────────


def _load_and_segment(
    data_dir: Path,
    split: str,
    seg_len: int,
    stride: int,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Load all JSONs for a split, segment, convert 17→18, transpose.

    Args:
        data_dir: Root data directory (e.g. data/ShanghaiTech).
        split: "train" or "test".
        seg_len: Temporal window length.
        stride: Sliding window stride.

    Returns:
        For train: (segments (N, 3, 24, 18) float32, None)
        For test:  (segments (N, 3, 24, 18) float32,
                    metadata (N, 3) int64 — [clip_idx, person_idx_within_clip, frame_start])
    """
    pose_dir = data_dir / "pose" / split
    json_files = sorted(pose_dir.glob("*_alphapose_tracked_person.json"))

    all_segments: list[np.ndarray] = []
    all_meta: list[np.ndarray] = []  # only for test

    for clip_idx, jf in enumerate(json_files):
        persons = _parse_single_json(jf)
        for pid_idx, (pid, (track, frame_ids)) in enumerate(sorted(persons.items())):
            segs, frame_starts = _segment_track(track, frame_ids, seg_len, stride)
            if segs.shape[0] == 0:
                continue
            segs_18 = _keypoints17_to_coco18(segs)  # (N, 24, 18, 3)
            # Transpose to (N, 3, 24, 18) = (N, C, T, V)
            segs_transposed = segs_18.transpose(0, 3, 1, 2).astype(np.float32)
            all_segments.append(segs_transposed)

            if split == "test":
                meta = np.stack(
                    [
                        np.full(len(frame_starts), clip_idx, dtype=np.int64),
                        np.full(len(frame_starts), pid_idx, dtype=np.int64),
                        frame_starts,
                    ],
                    axis=1,
                )  # (N_segs, 3)
                all_meta.append(meta)

    segments = np.concatenate(all_segments, axis=0)  # (N_total, 3, 24, 18)
    metadata = np.concatenate(all_meta, axis=0) if all_meta else None
    return segments, metadata


# ── Caching (memory-mapped .npy files) ───────────────────────────────────────


def _cache_prefix(data_dir: Path, split: str, seg_len: int, stride: int) -> str:
    """Compute a deterministic cache filename prefix based on parameters."""
    raw = f"{data_dir.resolve()}|{split}|{seg_len}|{stride}"
    h = hashlib.sha256(raw.encode()).hexdigest()[:12]
    return f"pose_{split}_{h}"


def _load_cached_or_process(
    cfg: Config,
    split: str,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Load preprocessed data from cache (memory-mapped), or parse JSONs and cache.

    Uses .npy files instead of .npz to support memory-mapping.
    Memory-mapped arrays allow multiprocessing workers to share data via
    OS page cache without pickling the full array — fixes Windows spawn crashes
    and dramatically reduces RAM usage.

    Args:
        cfg: Configuration object.
        split: "train" or "test".

    Returns:
        (segments, metadata) — segments is memory-mapped, metadata is loaded.
    """
    stride = cfg.seg_stride_train if split == "train" else cfg.seg_stride_test
    prefix = _cache_prefix(cfg.data_dir, split, cfg.seg_len, stride)
    seg_path = cfg.cache_dir / f"{prefix}_segments.npy"
    meta_path = cfg.cache_dir / f"{prefix}_metadata.npy"

    if seg_path.exists():
        # Memory-mapped load — no RAM copy, shared across workers
        segments = np.load(seg_path, mmap_mode="r")
        metadata = np.load(meta_path, mmap_mode="r") if meta_path.exists() else None
        return segments, metadata

    segments, metadata = _load_and_segment(cfg.data_dir, split, cfg.seg_len, stride)

    np.save(seg_path, segments)
    if metadata is not None:
        np.save(meta_path, metadata)

    # Re-load as memory-mapped for consistency
    segments = np.load(seg_path, mmap_mode="r")
    metadata = np.load(meta_path, mmap_mode="r") if metadata is not None else None
    return segments, metadata


# ── Normalization ────────────────────────────────────────────────────────────


def normalize_batch(xy: torch.Tensor) -> torch.Tensor:
    """Batch-vectorized pose normalization in torch.

    Steps (matching original numpy per-sample logic exactly):
      1. x /= 856, y /= 480
      2. Center by subtracting mean over (T, V) per sample
      3. Scale by population std of y-coordinate over (T, V) per sample

    Uses correction=0 (population std) to match numpy's default std().

    Args:
        xy: (B, 2, T, V) float32 tensor — raw x, y coordinates.

    Returns:
        Normalized tensor, same shape.
    """
    out = xy.clone()
    out[:, 0] /= 856.0
    out[:, 1] /= 480.0

    mean_xy = out.mean(dim=(2, 3), keepdim=True)  # (B, 2, 1, 1)
    std_y = out[:, 1].std(dim=(1, 2), correction=0, keepdim=True)  # (B, 1, 1)
    std_y = std_y.unsqueeze(1).clamp(min=1e-8)  # (B, 1, 1, 1)

    return (out - mean_xy) / std_y


# ── Dataset ──────────────────────────────────────────────────────────────────


class PoseSegmentDataset(Dataset):
    """Dataset of pose segments for STG-NF training or evaluation.

    For training:
      - Effective length = num_augmentations × num_segments
      - Even indices = identity, odd indices = horizontal flip
      - Returns (pose_tensor, label) where label=1.0 (all training data is normal)

    For testing:
      - No augmentation
      - Returns (pose_tensor, clip_idx, person_idx, frame_start)

    Uses memory-mapped numpy arrays to avoid pickling large data for
    multiprocessing workers on Windows (spawn-based).
    """

    def __init__(self, cfg: Config, split: str) -> None:
        """Initialize dataset.

        Args:
            cfg: Configuration object.
            split: "train" or "test".
        """
        super().__init__()
        self.split = split
        self.num_augmentations = cfg.num_augmentations if split == "train" else 1

        segments, metadata = _load_cached_or_process(cfg, split)
        self._segments = segments       # (N, 3, 24, 18) float32, memory-mapped
        self._metadata = metadata       # (N, 3) int64 or None, memory-mapped
        self._n_real = segments.shape[0]

    def __len__(self) -> int:
        return self._n_real * self.num_augmentations

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, ...]:
        """Get a single sample (raw, unnormalized).

        Normalization is deferred to batch level via normalize_batch()
        for ~10× speedup over per-sample numpy normalization.

        Args:
            idx: Linear index (augmentation interleaved).

        Returns:
            For train: (pose_raw (2, 24, 18), label scalar 1.0)
            For test:  (pose_raw (2, 24, 18), clip_idx, person_idx, frame_start)
        """
        aug_id = idx % self.num_augmentations
        real_idx = idx // self.num_augmentations

        seg = self._segments[real_idx]  # (3, 24, 18) — may be mmap view
        xy = np.array(seg[:2], dtype=np.float32)  # (2, 24, 18) — copy from mmap

        # Horizontal flip augmentation
        if aug_id == 1:
            xy[0] = -xy[0]

        pose = torch.from_numpy(np.ascontiguousarray(xy))

        if self.split == "train":
            return pose, torch.tensor(1.0, dtype=torch.float32)
        else:
            m = self._metadata[real_idx]
            return (
                pose,
                torch.tensor(int(m[0]), dtype=torch.long),
                torch.tensor(int(m[1]), dtype=torch.long),
                torch.tensor(int(m[2]), dtype=torch.long),
            )


# ── DataLoader constructors ─────────────────────────────────────────────────


def get_train_loader(cfg: Config) -> DataLoader:
    """Build training DataLoader.

    Args:
        cfg: Configuration.

    Returns:
        DataLoader yielding (pose, label) batches.
    """
    use_cuda = torch.cuda.is_available() and cfg.device != "cpu"
    ds = PoseSegmentDataset(cfg, "train")
    return DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=use_cuda,
        drop_last=False,
        persistent_workers=cfg.num_workers > 0,
    )


def get_test_loader(cfg: Config) -> DataLoader:
    """Build test DataLoader.

    Args:
        cfg: Configuration.

    Returns:
        DataLoader yielding (pose, clip_idx, person_idx, frame_start) batches.
    """
    use_cuda = torch.cuda.is_available() and cfg.device != "cpu"
    ds = PoseSegmentDataset(cfg, "test")
    return DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=use_cuda,
        drop_last=False,
        persistent_workers=cfg.num_workers > 0,
    )


def get_test_clip_info(cfg: Config) -> list[Path]:
    """Return sorted list of test clip JSON paths (for GT mask lookup).

    Args:
        cfg: Configuration.

    Returns:
        List of JSON file paths in sorted order.
    """
    pose_dir = cfg.data_dir / "pose" / "test"
    return sorted(pose_dir.glob("*_alphapose_tracked_person.json"))
