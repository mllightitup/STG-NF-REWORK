"""Evaluation and AuC scoring for STG-NF.

Handles:
  - Running the model on the test set
  - Assigning per-segment scores to frames
  - Multi-person aggregation (min per frame)
  - Gaussian smoothing (6× sequential, σ=1..6)
  - ROC-AUC computation vs ground truth

Deviations from original:
  - **Vectorized frame scoring**: Instead of nested Python loops, uses numpy
    vectorized operations for assigning scores to frames.
  - **Accepts model directly**: Can evaluate without checkpoint reload —
    avoids re-creating test DataLoader each epoch.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

if TYPE_CHECKING:
    from .config import Config

from .dataset import get_test_clip_info, get_test_loader, normalize_batch
from .model import STG_NF


def evaluate(
    cfg: Config,
    *,
    model: STG_NF | None = None,
    checkpoint_path: Path | str | None = None,
    device: torch.device | str | None = None,
    test_loader: DataLoader | None = None,
    silent: bool = False,
) -> float:
    """Evaluate model on ShanghaiTech test set and return AuC.

    Either `model` or `checkpoint_path` must be provided.
    If `test_loader` is provided, it will be reused (saves RAM & time).

    Args:
        cfg: Configuration.
        model: Pre-loaded model (preferred for in-training eval).
        checkpoint_path: Path to .pt checkpoint (used if model is None).
        device: Device override (defaults to cfg.device).
        test_loader: Pre-built test DataLoader to reuse.
        silent: If True, suppress progress bar.

    Returns:
        AuC score as percentage (e.g. 85.9).
    """
    if device is None:
        device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    # ── Load or use provided model ───────────────────────────
    if model is None:
        assert checkpoint_path is not None, "Either model or checkpoint_path required"
        model = STG_NF(cfg).to(device)
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state = ckpt["model_state_dict"]
        # Strip _orig_mod. prefix added by torch.compile()
        state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
        model.load_state_dict(state)
        model.set_all_initialized()
        model.eval()
        model.fuse_bn_for_inference()

    model.eval()

    # ── Get or create test loader ────────────────────────────
    if test_loader is None:
        test_loader = get_test_loader(cfg)

    # ── Run inference ────────────────────────────────────────
    all_scores: list[float] = []
    all_clip_idx: list[int] = []
    all_person_idx: list[int] = []
    all_frame_start: list[int] = []

    iterator = tqdm(test_loader, desc="Evaluating", miniters=2.0) if not silent else test_loader

    with torch.no_grad():
        for batch in iterator:
            pose, clip_idx, person_idx, frame_start = batch
            pose = pose.to(device, non_blocking=True)
            pose = normalize_batch(pose)

            nll = model(pose)  # (B,)
            # Normality score = -nll (higher = more normal)
            scores = -nll.cpu().numpy()

            all_scores.extend(scores.tolist())
            all_clip_idx.extend(clip_idx.numpy().tolist())
            all_person_idx.extend(person_idx.numpy().tolist())
            all_frame_start.extend(frame_start.numpy().tolist())

    scores_arr = np.array(all_scores)
    clip_idx_arr = np.array(all_clip_idx)
    frame_start_arr = np.array(all_frame_start)

    # ── Compute AuC ──────────────────────────────────────────
    auc = score_dataset(
        cfg=cfg,
        scores=scores_arr,
        clip_indices=clip_idx_arr,
        frame_starts=frame_start_arr,
    )
    return auc


def score_dataset(
    cfg: Config,
    scores: np.ndarray,
    clip_indices: np.ndarray,
    frame_starts: np.ndarray,
) -> float:
    """Aggregate per-segment scores into per-frame AuC.

    Scoring logic (matching original):
      1. Each segment score is assigned to its midpoint frame (t + seg_len//2)
      2. For frames with multiple persons, take the minimum score (most anomalous)
      3. Frames without any score get +inf, replaced later with max score
      4. Sequential Gaussian smoothing 6 times (σ=1,2,3,4,5,6)
      5. ROC-AUC against ground truth masks

    Args:
        cfg: Configuration.
        scores: Per-segment normality scores (N_test,).
        clip_indices: Clip index per segment (N_test,).
        frame_starts: Start frame per segment (N_test,).

    Returns:
        AuC as percentage.
    """
    gt_dir = cfg.data_dir / "gt" / "test_frame_mask"
    clip_json_paths = get_test_clip_info(cfg)

    mid_offset = cfg.seg_len // 2  # = 12

    all_gt: list[np.ndarray] = []
    all_scores_frames: list[np.ndarray] = []

    for clip_idx, json_path in enumerate(clip_json_paths):
        # ── Load ground truth ────────────────────────────────
        clip_name = json_path.stem.replace("_alphapose_tracked_person", "")
        gt_path = gt_dir / f"{clip_name}.npy"
        if not gt_path.exists():
            continue

        gt_mask = np.load(gt_path)
        # Original convention: 1=abnormal, 0=normal → invert
        gt_mask = 1.0 - gt_mask  # now 1=normal, 0=abnormal
        n_frames = len(gt_mask)

        # ── Assign scores to frames ──────────────────────────
        frame_scores = np.full(n_frames, np.inf)

        mask = clip_indices == clip_idx
        if mask.any():
            seg_scores = scores[mask]
            seg_frames = frame_starts[mask]

            # Assign each segment score to its midpoint frame
            for i in range(len(seg_scores)):
                frame_idx = int(seg_frames[i]) + mid_offset
                if 0 <= frame_idx < n_frames:
                    frame_scores[frame_idx] = min(frame_scores[frame_idx], seg_scores[i])

        all_gt.append(gt_mask)
        all_scores_frames.append(frame_scores)

    if not all_scores_frames:
        return 0.0

    # ── Global Inf Replacement ───────────────────────────────
    # Replace np.inf with the max normality score across the ENTIRE dataset.
    scores_concat = np.concatenate(all_scores_frames)
    
    valid_max_mask = scores_concat != np.inf
    if valid_max_mask.any():
        global_max = scores_concat[valid_max_mask].max()
    else:
        global_max = 0.0
        
    valid_min_mask = scores_concat != -np.inf
    if valid_min_mask.any():
        global_min = scores_concat[valid_min_mask].min()
    else:
        global_min = 0.0

    for i in range(len(all_scores_frames)):
        frame_scores = all_scores_frames[i]
        frame_scores[frame_scores == np.inf] = global_max
        frame_scores[frame_scores == -np.inf] = global_min

        # ── Gaussian smoothing (6× sequential) ──────────────
        for sigma in range(1, 7):
            frame_scores = gaussian_filter1d(frame_scores, sigma=sigma)
            
        all_scores_frames[i] = frame_scores

    # ── Concatenate and compute ROC-AUC ──────────────────────
    gt_all = np.concatenate(all_gt)
    scores_all = np.concatenate(all_scores_frames)

    auc = roc_auc_score(gt_all, scores_all) * 100.0
    return auc
