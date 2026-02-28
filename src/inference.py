"""Real-time streaming inference for STG-NF.

Designed for live video feeds where pose detections arrive frame-by-frame.
Maintains a sliding window buffer and outputs anomaly scores once the
window is full.

Usage:
    python -m src.inference --checkpoint checkpoints/stg_nf_epoch8.pt
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    pass

from .config import Config
from .dataset import _COCO18_ORDER, normalize_batch
from .model import STG_NF


class PersonTracker:
    """Buffer for one tracked person's pose history.

    Maintains a sliding window of the last `seg_len` frames.
    Once full, provides preprocessed tensors for inference.
    """

    def __init__(self, seg_len: int) -> None:
        self.seg_len = seg_len
        self._buffer: list[np.ndarray] = []  # list of (18, 2) arrays

    def add_frame(self, keypoints_18_xy: np.ndarray) -> None:
        """Add a frame's keypoints.

        Args:
            keypoints_18_xy: Shape (18, 2) — COCO18-ordered x,y coordinates.
        """
        self._buffer.append(keypoints_18_xy)
        if len(self._buffer) > self.seg_len:
            self._buffer.pop(0)

    @property
    def ready(self) -> bool:
        """Whether the buffer has enough frames for inference."""
        return len(self._buffer) >= self.seg_len

    def get_segment(self) -> np.ndarray:
        """Extract raw (unnormalized) segment for inference.

        Returns:
            Array of shape (2, seg_len, 18) float32 — raw xy coordinates.
        """
        # Stack → (T=24, 18, 2)
        segment = np.stack(self._buffer[-self.seg_len :], axis=0)
        # → (2, 24, 18) — channels first, joints last
        xy = segment[:, :, :2].transpose(2, 0, 1).astype(np.float32)
        return xy


class RealtimeInference:
    """Real-time anomaly detection engine.

    Handles multiple tracked persons, maintains sliding windows,
    and runs STG-NF inference to produce per-frame anomaly scores.

    Args:
        cfg: Configuration.
        checkpoint_path: Path to trained model checkpoint.
        device: Torch device string.
    """

    def __init__(
        self,
        cfg: Config,
        checkpoint_path: str | Path,
        device: str = "cuda",
    ) -> None:
        self.cfg = cfg
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Load model
        self.model = STG_NF(cfg).to(self.device)
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        state = ckpt["model_state_dict"]
        state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
        self.model.load_state_dict(state)
        self.model.set_all_initialized()
        self.model.eval()
        self.model.fuse_bn_for_inference()

        # Person buffers
        self._trackers: dict[int | str, PersonTracker] = defaultdict(
            lambda: PersonTracker(cfg.seg_len)
        )

    def _convert_17_to_18(self, kp17: np.ndarray) -> np.ndarray:
        """Convert 17-joint AlphaPose output to 18-joint COCO18.

        Args:
            kp17: Shape (17, 2) — x, y for each joint.

        Returns:
            Shape (18, 2) in COCO18 order.
        """
        neck = 0.5 * (kp17[5] + kp17[6])
        kp_with_neck = np.concatenate([kp17, neck[np.newaxis]], axis=0)  # (18, 2)
        return kp_with_neck[_COCO18_ORDER]

    @torch.no_grad()
    def process_frame(
        self,
        detections: dict[int | str, np.ndarray],
    ) -> dict[int | str, float]:
        """Process one frame of pose detections.

        Args:
            detections: Dict mapping person_id → (17, 2) keypoint array (x, y).
                Use raw pixel coordinates — normalization is handled internally.

        Returns:
            Dict mapping person_id → anomaly_score (higher = more normal).
            Only persons with a full window are scored.
        """
        results: dict[int | str, float] = {}
        ready_ids: list[int | str] = []
        segments: list[np.ndarray] = []

        for pid, kp17 in detections.items():
            kp18 = self._convert_17_to_18(kp17)
            self._trackers[pid].add_frame(kp18)

            if self._trackers[pid].ready:
                ready_ids.append(pid)
                segments.append(self._trackers[pid].get_segment())

        if not segments:
            return results

        # Batch inference with normalization
        batch = torch.from_numpy(np.stack(segments, axis=0)).to(self.device)
        batch = normalize_batch(batch)
        nll = self.model(batch)
        scores = -nll.cpu().numpy()

        for pid, score in zip(ready_ids, scores):
            results[pid] = float(score)

        return results

    def remove_person(self, person_id: int | str) -> None:
        """Remove a tracked person (e.g. when they leave the scene).

        Args:
            person_id: ID of the person to remove.
        """
        self._trackers.pop(person_id, None)

    def reset(self) -> None:
        """Clear all person buffers."""
        self._trackers.clear()


# ── CLI ──────────────────────────────────────────────────────────────────────


def main() -> None:
    """Demonstration entry point for real-time inference."""
    parser = argparse.ArgumentParser(description="STG-NF Real-Time Inference")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint (.pt)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device (cuda or cpu)",
    )
    args = parser.parse_args()

    cfg = Config()
    engine = RealtimeInference(cfg, args.checkpoint, args.device)

    print(f"[inference] Loaded model from {args.checkpoint}")
    print(f"[inference] Device: {engine.device}")
    print(f"[inference] Ready for real-time pose input.")
    print(f"[inference] Feed detections via engine.process_frame({{pid: kp17_array}})")
    print()
    print("Example usage in Python:")
    print("  from src.inference import RealtimeInference")
    print("  from src.config import Config")
    print("  engine = RealtimeInference(Config(), 'checkpoint.pt')")
    print("  scores = engine.process_frame({0: np.random.randn(17, 2)})")


if __name__ == "__main__":
    main()
