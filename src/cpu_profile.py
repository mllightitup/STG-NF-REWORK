"""CPU performance profiler for STG-NF model.

Profiles both inference and training on CPU to find bottlenecks.
Outputs a breakdown of time spent per operation.

Usage: python -m src.cpu_profile
"""

from __future__ import annotations

import time

import numpy as np
import torch
import torch.profiler

from .config import Config
from .dataset import get_test_loader, get_train_loader
from .export_onnx import _load_model


def profile_inference(cfg: Config, ckpt_path) -> None:
    """Profile CPU inference with torch.profiler."""
    print("=" * 65)
    print("  CPU Inference Profile")
    print("=" * 65)

    model = _load_model(cfg, ckpt_path).cpu()
    model.eval()
    test_loader = get_test_loader(cfg)

    # Warmup
    batch = next(iter(test_loader))
    pose = batch[0]
    with torch.no_grad():
        for _ in range(5):
            model(pose)

    # Profile
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU],
        record_shapes=True,
        with_stack=True,
    ) as prof:
        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                if i >= 20:  # profile 20 batches
                    break
                pose = batch[0]
                model(pose)

    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=25))

    # Also profile individual components
    print("\n" + "=" * 65)
    print("  Component-level timing (20 batches)")
    print("=" * 65)
    dummy = torch.randn(256, 2, 24, 18)

    # Time encode vs nll separately
    n = 100
    with torch.no_grad():
        t0 = time.perf_counter()
        for _ in range(n):
            z, logdet = model.encode(dummy)
        encode_ms = (time.perf_counter() - t0) / n * 1000

        t0 = time.perf_counter()
        for _ in range(n):
            model._compute_nll(z, logdet)
        nll_ms = (time.perf_counter() - t0) / n * 1000

        t0 = time.perf_counter()
        for _ in range(n):
            model(dummy)
        total_ms = (time.perf_counter() - t0) / n * 1000

    print(f"  encode():       {encode_ms:.2f} ms")
    print(f"  _compute_nll(): {nll_ms:.2f} ms")
    print(f"  total forward:  {total_ms:.2f} ms")

    # Time each flow step
    with torch.no_grad():
        for k, step in enumerate(model.flow_steps):
            z_in = torch.randn(256, 2, 24, 18)
            ld_in = torch.zeros(256)

            t0 = time.perf_counter()
            for _ in range(n):
                step(z_in, ld_in)
            step_ms = (time.perf_counter() - t0) / n * 1000
            print(f"    FlowStep[{k}]: {step_ms:.2f} ms")


def profile_dataloader(cfg: Config) -> None:
    """Profile DataLoader throughput on CPU."""
    print("\n" + "=" * 65)
    print("  DataLoader Throughput")
    print("=" * 65)

    loader = get_train_loader(cfg)

    # Time just data loading (no model)
    t0 = time.perf_counter()
    for i, batch in enumerate(loader):
        if i >= 50:
            break
    data_ms = (time.perf_counter() - t0) / 50 * 1000
    print(f"  DataLoader: {data_ms:.2f} ms/batch (pure loading, no model)")

    # Compare numpy normalize_pose cost
    from .dataset import normalize_batch
    dummy_batch = torch.randn(256, 2, 24, 18)
    t0 = time.perf_counter()
    for _ in range(1000):
        normalize_batch(dummy_batch)
    norm_ms = (time.perf_counter() - t0) / 1000 * 1000
    print(f"  normalize_batch: {norm_ms:.2f} ms/batch (torch vectorized)")


def main() -> None:
    cfg = Config()
    ckpt_path = cfg.checkpoint_dir / "stg_nf_best.pt"

    profile_inference(cfg, ckpt_path)
    profile_dataloader(cfg)


if __name__ == "__main__":
    main()
