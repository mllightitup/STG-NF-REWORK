"""Realistic inference latency benchmark.

Tests:
  1. Latency vs batch_size (1, 2, 4, 8, 16, 32, 64, 128, 256)
  2. Simulated real-time: variable people per frame (1-10), dynamic batching
  3. Single-person-at-a-time vs batched inference

Covers: PyTorch GPU eager, torch.compile, ONNX CUDA, ONNX TensorRT.

Usage: python -m src.realtime_bench
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from .config import Config
from .dataset import normalize_batch
from .export_onnx import MODELS_DIR, _load_model


def _make_dummy(batch_size: int) -> torch.Tensor:
    """Create dummy normalized input."""
    raw = torch.randn(batch_size, 2, 24, 18)
    return normalize_batch(raw)


# ── 1. Latency vs Batch Size ────────────────────────────────────────────────


def bench_pytorch_batch_sizes(
    model: torch.nn.Module, device: torch.device, label: str,
    sizes: list[int] = [1, 2, 4, 8, 16, 32, 64, 128, 256],
    n_iter: int = 300, warmup: int = 30,
) -> dict[int, float]:
    """Benchmark latency for different batch sizes."""
    model.eval()
    results = {}

    for bs in sizes:
        dummy = _make_dummy(bs).to(device)
        with torch.no_grad():
            for _ in range(warmup):
                model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(n_iter):
                model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) / n_iter * 1000
        results[bs] = ms

    return results


def bench_ort_batch_sizes(
    session: ort.InferenceSession, label: str,
    sizes: list[int] = [1, 2, 4, 8, 16, 32, 64, 128, 256],
    n_iter: int = 300, warmup: int = 30,
) -> dict[int, float]:
    """Benchmark ORT latency for different batch sizes."""
    input_name = session.get_inputs()[0].name
    results = {}

    for bs in sizes:
        dummy = _make_dummy(bs).numpy()
        for _ in range(warmup):
            session.run(None, {input_name: dummy})

        t0 = time.perf_counter()
        for _ in range(n_iter):
            session.run(None, {input_name: dummy})
        ms = (time.perf_counter() - t0) / n_iter * 1000
        results[bs] = ms

    return results


# ── 2. Realistic Video Simulation ───────────────────────────────────────────


def simulate_video_stream(
    n_frames: int = 1000,
    max_people: int = 10,
    appear_prob: float = 0.15,
    disappear_prob: float = 0.10,
) -> list[int]:
    """Generate realistic per-frame person counts.

    People randomly appear/disappear, simulating a surveillance scene.
    Returns list of person counts per frame.
    """
    rng = np.random.default_rng(42)
    current_people = 3
    counts = []

    for _ in range(n_frames):
        # Random appear/disappear
        if current_people < max_people and rng.random() < appear_prob:
            current_people += 1
        if current_people > 1 and rng.random() < disappear_prob:
            current_people -= 1
        counts.append(current_people)

    return counts


def bench_realtime_pytorch(
    model: torch.nn.Module, device: torch.device,
    person_counts: list[int],
) -> tuple[float, float]:
    """Simulate real-time inference with variable batch sizes.

    Returns (total_ms, avg_ms_per_frame).
    """
    model.eval()

    # Warmup
    dummy = _make_dummy(10).to(device)
    with torch.no_grad():
        for _ in range(20):
            model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    with torch.no_grad():
        for n_people in person_counts:
            if n_people == 0:
                continue
            batch = _make_dummy(n_people).to(device)
            model(batch)
    if device.type == "cuda":
        torch.cuda.synchronize()
    total = (time.perf_counter() - t0) * 1000
    return total, total / len(person_counts)


def bench_realtime_ort(
    session: ort.InferenceSession,
    person_counts: list[int],
) -> tuple[float, float]:
    """Simulate real-time ORT inference."""
    input_name = session.get_inputs()[0].name

    dummy = _make_dummy(10).numpy()
    for _ in range(20):
        session.run(None, {input_name: dummy})

    t0 = time.perf_counter()
    for n_people in person_counts:
        if n_people == 0:
            continue
        batch = _make_dummy(n_people).numpy()
        session.run(None, {input_name: batch})
    total = (time.perf_counter() - t0) * 1000
    return total, total / len(person_counts)


# ── 3. Single vs Batched ────────────────────────────────────────────────────


def bench_single_vs_batch_pytorch(
    model: torch.nn.Module, device: torch.device, n_people: int = 5,
    n_iter: int = 300, warmup: int = 30,
) -> tuple[float, float]:
    """Compare single-person-at-a-time vs batched inference.

    Returns (single_ms, batched_ms) per frame.
    """
    model.eval()
    singles = [_make_dummy(1).to(device) for _ in range(n_people)]
    batch = _make_dummy(n_people).to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            model(batch)
    if device.type == "cuda":
        torch.cuda.synchronize()

    # Single person × N
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_iter):
            for s in singles:
                model(s)
    if device.type == "cuda":
        torch.cuda.synchronize()
    single_ms = (time.perf_counter() - t0) / n_iter * 1000

    # Batched N
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_iter):
            model(batch)
    if device.type == "cuda":
        torch.cuda.synchronize()
    batch_ms = (time.perf_counter() - t0) / n_iter * 1000

    return single_ms, batch_ms


def bench_single_vs_batch_ort(
    session: ort.InferenceSession, n_people: int = 5,
    n_iter: int = 300, warmup: int = 30,
) -> tuple[float, float]:
    """Compare single vs batched ORT inference."""
    input_name = session.get_inputs()[0].name
    singles = [_make_dummy(1).numpy() for _ in range(n_people)]
    batch = _make_dummy(n_people).numpy()

    for _ in range(warmup):
        session.run(None, {input_name: batch})

    t0 = time.perf_counter()
    for _ in range(n_iter):
        for s in singles:
            session.run(None, {input_name: s})
    single_ms = (time.perf_counter() - t0) / n_iter * 1000

    t0 = time.perf_counter()
    for _ in range(n_iter):
        session.run(None, {input_name: batch})
    batch_ms = (time.perf_counter() - t0) / n_iter * 1000

    return single_ms, batch_ms


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    cfg = Config()
    ckpt = cfg.checkpoint_dir / "stg_nf_best.pt"
    onnx_path = MODELS_DIR / "stg_nf.onnx"

    backends: dict[str, object] = {}

    # Prepare backends
    print("[setup] Loading models...")
    if torch.cuda.is_available():
        backends["PyTorch GPU"] = ("pytorch", _load_model(cfg, ckpt).cuda(), torch.device("cuda"))
        backends["torch.compile"] = ("pytorch", torch.compile(_load_model(cfg, ckpt).cuda()), torch.device("cuda"))

    if onnx_path.exists():
        if "CUDAExecutionProvider" in ort.get_available_providers():
            so = ort.SessionOptions()
            so.log_severity_level = 3
            backends["ONNX CUDA"] = ("ort", ort.InferenceSession(str(onnx_path), sess_options=so, providers=["CUDAExecutionProvider"]))
        if "TensorrtExecutionProvider" in ort.get_available_providers():
            so = ort.SessionOptions()
            so.log_severity_level = 3
            trt_cache = MODELS_DIR / "trt_cache"
            trt_cache.mkdir(exist_ok=True)
            backends["ONNX TensorRT"] = ("ort", ort.InferenceSession(
                str(onnx_path), sess_options=so,
                providers=[("TensorrtExecutionProvider", {
                    "trt_engine_cache_enable": True,
                    "trt_engine_cache_path": str(trt_cache),
                    "trt_fp16_enable": True,
                })]))

    # ═══════════════════════════════════════════════════════════
    #  Test 1: Latency vs Batch Size
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 75)
    print("  TEST 1: Latency vs Batch Size")
    print("=" * 75)

    sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    all_results: dict[str, dict[int, float]] = {}

    for name, backend in backends.items():
        if backend[0] == "pytorch":
            _, model, device = backend
            all_results[name] = bench_pytorch_batch_sizes(model, device, name, sizes)
        else:
            _, session = backend
            all_results[name] = bench_ort_batch_sizes(session, name, sizes)

    # Print table
    header = f"  {'Batch':>5}"
    for name in all_results:
        header += f"  {name:>14}"
    print(header)
    print(f"  {'-'*5}" + f"  {'-'*14}" * len(all_results))

    for bs in sizes:
        row = f"  {bs:>5}"
        for name in all_results:
            ms = all_results[name][bs]
            row += f"  {ms:>11.2f} ms"
        print(row)

    # Per-sample throughput
    print(f"\n  {'Batch':>5}", end="")
    for name in all_results:
        print(f"  {name:>14}", end="")
    print("  (µs/sample)")
    print(f"  {'-'*5}" + f"  {'-'*14}" * len(all_results))
    for bs in sizes:
        row = f"  {bs:>5}"
        for name in all_results:
            us = all_results[name][bs] / bs * 1000
            row += f"  {us:>11.1f} µs"
        print(row)

    # ═══════════════════════════════════════════════════════════
    #  Test 2: Realistic Video Stream
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 75)
    print("  TEST 2: Simulated Real-Time Video (1000 frames, 1-10 people)")
    print("=" * 75)

    counts = simulate_video_stream(n_frames=1000)
    mean_ppl = np.mean(counts)
    print(f"  Avg people/frame: {mean_ppl:.1f}, min: {min(counts)}, max: {max(counts)}")

    for name, backend in backends.items():
        if backend[0] == "pytorch":
            _, model, device = backend
            total, per_frame = bench_realtime_pytorch(model, device, counts)
        else:
            _, session = backend
            total, per_frame = bench_realtime_ort(session, counts)

        fps = 1000 / per_frame if per_frame > 0 else float("inf")
        print(f"  {name:<16}  {per_frame:>6.2f} ms/frame  →  {fps:>7.0f} FPS")

    # ═══════════════════════════════════════════════════════════
    #  Test 3: Single Person vs Batched (5 people)
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 75)
    print("  TEST 3: Single-Person × 5 vs Batched (batch=5)")
    print("=" * 75)

    for name, backend in backends.items():
        if backend[0] == "pytorch":
            _, model, device = backend
            single, batched = bench_single_vs_batch_pytorch(model, device, n_people=5)
        else:
            _, session = backend
            single, batched = bench_single_vs_batch_ort(session, n_people=5)

        ratio = single / batched if batched > 0 else float("inf")
        print(f"  {name:<16}  single: {single:>6.2f} ms  batch: {batched:>6.2f} ms  "
              f"→ batch {ratio:.1f}× faster")

    print("\n" + "=" * 75)


if __name__ == "__main__":
    main()
