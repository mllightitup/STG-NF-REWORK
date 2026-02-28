"""Configuration for STG-NF model — ShanghaiTech / uniform strategy."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Config:
    """All hyperparameters for STG-NF training & inference.

    Values are hardcoded for the ShanghaiTech dataset with
    uniform adjacency strategy and standard parameters matching
    the original paper.
    """

    # ── Data ──────────────────────────────────────────────────
    data_dir: Path = Path("data/ShanghaiTech")
    seg_len: int = 24              # temporal window τ
    seg_stride_train: int = 6      # sliding-window stride (train)
    seg_stride_test: int = 1       # sliding-window stride (test)
    video_res: tuple[int, int] = (856, 480)  # ShanghaiTech frame resolution

    # ── Skeleton ──────────────────────────────────────────────
    num_raw_joints: int = 17       # AlphaPose output joints
    num_joints: int = 18           # after adding synthetic neck
    in_channels: int = 2           # x, y (no confidence)

    # ── Model ─────────────────────────────────────────────────
    num_flow_steps: int = 8        # K flow steps
    temporal_kernel: int = 13      # seg_len // 2 + 1
    prior_mean: float = 3.0        # R — prior μ for normal class

    # ── Training ──────────────────────────────────────────────
    batch_size: int = 256
    epochs: int = 16
    warmup_epochs: int = 2         # linear warmup epochs
    lr: float = 5e-4               # Adamax learning rate
    lr_decay: float = 0.85         # multiplicative decay per epoch
    weight_decay: float = 5e-5     # Adamax weight decay (improves AuC)
    max_grad_norm: float = 100.0   # gradient clipping
    num_augmentations: int = 2     # identity + horizontal flip

    # ── Misc ──────────────────────────────────────────────────
    seed: int = 999
    device: str = "cuda"
    num_workers: int = 4
    profile_system: bool = False   # log CPU/RAM/VRAM usage to CSV
    checkpoint_dir: Path = Path("checkpoints")
    cache_dir: Path = Path("cache")

    def __post_init__(self) -> None:
        """Ensure output directories exist."""
        # frozen=True prevents assignment, but mkdir is fine
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
