"""ONNX export and comprehensive multi-provider benchmark.

Exports the best checkpoint to ONNX, then benchmarks:
  - PyTorch CPU (eager)
  - PyTorch GPU (eager)
  - PyTorch GPU (torch.compile)
  - ONNX CPU (CPUExecutionProvider)
  - ONNX CUDA (CUDAExecutionProvider)
  - ONNX TensorRT (TensorrtExecutionProvider)

For each backend: measures latency AND verifies AuC to ensure correctness.

Usage: python -m src.export_onnx
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from .config import Config
from .dataset import get_test_clip_info, get_test_loader, normalize_batch
from .evaluate import score_dataset
from .model import STG_NF


# ── Directory for exported models ────────────────────────────────────────────
MODELS_DIR = Path("models")


def _load_model(cfg: Config, checkpoint_path: Path, *, fuse_bn: bool = True) -> STG_NF:
    """Load STG_NF model from checkpoint, handling _orig_mod. prefix.

    Args:
        cfg: Configuration.
        checkpoint_path: Path to checkpoint.
        fuse_bn: If True, fold BatchNorm into Conv2d for inference speedup.
    """
    model = STG_NF(cfg)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = ckpt["model_state_dict"]
    state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
    model.load_state_dict(state)
    model.set_all_initialized()
    if fuse_bn:
        model.eval()
        model.fuse_bn_for_inference()
    return model


def export_onnx(cfg: Config, checkpoint_path: Path, output_path: Path) -> Path:
    """Export STG_NF to ONNX format.

    Args:
        cfg: Configuration.
        checkpoint_path: Path to .pt checkpoint.
        output_path: Where to save the .onnx file.

    Returns:
        Path to the exported ONNX file.
    """
    model = _load_model(cfg, checkpoint_path)
    model.eval()

    dummy = torch.randn(1, cfg.in_channels, cfg.seg_len, cfg.num_joints)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Define dynamic batch dimension for dynamo exporter
    batch_dim = torch.export.Dim("batch", min=1, max=1024)
    dynamic_shapes = {"x": {0: batch_dim}}

    torch.onnx.export(
        model,
        (dummy,),
        str(output_path),
        input_names=["pose"],
        output_names=["nll"],
        dynamic_shapes=dynamic_shapes,
        opset_version=18,
        do_constant_folding=True,
    )

    # Validate
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    print(f"[export] ONNX model saved: {output_path} "
          f"({output_path.stat().st_size / 1024:.1f} KB)")

    return output_path


# ── Inference runners ────────────────────────────────────────────────────────


def _run_pytorch_inference(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Run PyTorch inference, return (scores, clip_indices, frame_starts, latency_ms)."""
    model.eval()
    all_scores, all_clips, all_frames = [], [], []

    # Warmup
    with torch.no_grad():
        batch = next(iter(test_loader))
        pose = batch[0].to(device, non_blocking=True)
        pose = normalize_batch(pose)
        for _ in range(5):
            model(pose)
    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    with torch.no_grad():
        for batch in test_loader:
            pose, clip_idx, _, frame_start = batch
            pose = pose.to(device, non_blocking=True)
            pose = normalize_batch(pose)
            nll = model(pose)
            all_scores.extend((-nll).cpu().numpy().tolist())
            all_clips.extend(clip_idx.numpy().tolist())
            all_frames.extend(frame_start.numpy().tolist())

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    return (np.array(all_scores), np.array(all_clips),
            np.array(all_frames), elapsed * 1000)


def _run_ort_inference(
    session: ort.InferenceSession,
    test_loader: torch.utils.data.DataLoader,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Run ONNX Runtime inference, return (scores, clip_indices, frame_starts, latency_ms)."""
    input_name = session.get_inputs()[0].name
    all_scores, all_clips, all_frames = [], [], []

    # Warmup
    batch = next(iter(test_loader))
    dummy = normalize_batch(batch[0]).numpy()
    for _ in range(5):
        session.run(None, {input_name: dummy})

    t0 = time.perf_counter()
    for batch in test_loader:
        pose, clip_idx, _, frame_start = batch
        pose = normalize_batch(pose)
        nll = session.run(None, {input_name: pose.numpy()})[0]
        all_scores.extend((-nll).tolist())
        all_clips.extend(clip_idx.numpy().tolist())
        all_frames.extend(frame_start.numpy().tolist())

    elapsed = time.perf_counter() - t0

    return (np.array(all_scores), np.array(all_clips),
            np.array(all_frames), elapsed * 1000)


# ── Latency microbenchmark ───────────────────────────────────────────────────


def _microbench_pytorch(
    model: torch.nn.Module, device: torch.device,
    batch_size: int = 256, n_iter: int = 200, warmup: int = 20,
) -> float:
    """Return average ms per batch for PyTorch model."""
    dummy = torch.randn(batch_size, 2, 24, 18, device=device)
    model.eval()

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
    return (time.perf_counter() - t0) / n_iter * 1000


def _microbench_ort(
    session: ort.InferenceSession,
    batch_size: int = 256, n_iter: int = 200, warmup: int = 20,
) -> float:
    """Return average ms per batch for ORT session."""
    input_name = session.get_inputs()[0].name
    dummy = np.random.randn(batch_size, 2, 24, 18).astype(np.float32)

    for _ in range(warmup):
        session.run(None, {input_name: dummy})

    t0 = time.perf_counter()
    for _ in range(n_iter):
        session.run(None, {input_name: dummy})
    return (time.perf_counter() - t0) / n_iter * 1000


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    """Export to ONNX and run comprehensive benchmark across all providers."""
    cfg = Config()
    ckpt_path = cfg.checkpoint_dir / "stg_nf_best.pt"

    if not ckpt_path.exists():
        print(f"[error] Checkpoint not found: {ckpt_path}")
        return

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    onnx_path = MODELS_DIR / "stg_nf.onnx"

    # ── Export ────────────────────────────────────────────────
    print("=" * 65)
    print("  ONNX Export")
    print("=" * 65)
    export_onnx(cfg, ckpt_path, onnx_path)

    # ── Prepare test data ────────────────────────────────────
    print("\n[bench] Loading test data...")
    test_loader = get_test_loader(cfg)
    print(f"[bench] Test samples: {len(test_loader.dataset)}")

    results: list[dict] = []

    # ── 1. PyTorch CPU ────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  1. PyTorch CPU (eager)")
    print("=" * 65)
    cpu_model = _load_model(cfg, ckpt_path).cpu()
    scores, clips, frames, total_ms = _run_pytorch_inference(
        cpu_model, test_loader, torch.device("cpu"))
    auc = score_dataset(cfg, scores, clips, frames)
    latency = _microbench_pytorch(cpu_model, torch.device("cpu"))
    print(f"  AuC: {auc:.2f}%  |  Eval: {total_ms:.0f}ms  |  Latency: {latency:.2f} ms/batch")
    results.append({"name": "PyTorch CPU", "auc": auc, "eval_ms": total_ms, "latency_ms": latency})
    del cpu_model

    # ── 2. PyTorch GPU (eager) ────────────────────────────────
    if torch.cuda.is_available():
        print("\n" + "=" * 65)
        print("  2. PyTorch GPU (eager)")
        print("=" * 65)
        gpu_model = _load_model(cfg, ckpt_path).cuda()
        scores, clips, frames, total_ms = _run_pytorch_inference(
            gpu_model, test_loader, torch.device("cuda"))
        auc = score_dataset(cfg, scores, clips, frames)
        latency = _microbench_pytorch(gpu_model, torch.device("cuda"))
        print(f"  AuC: {auc:.2f}%  |  Eval: {total_ms:.0f}ms  |  Latency: {latency:.2f} ms/batch")
        results.append({"name": "PyTorch GPU", "auc": auc, "eval_ms": total_ms, "latency_ms": latency})
        del gpu_model
        torch.cuda.empty_cache()

    # ── 3. PyTorch GPU (torch.compile) ────────────────────────
    if torch.cuda.is_available():
        print("\n" + "=" * 65)
        print("  3. PyTorch GPU (torch.compile)")
        print("=" * 65)
        compiled_model = torch.compile(_load_model(cfg, ckpt_path).cuda())
        scores, clips, frames, total_ms = _run_pytorch_inference(
            compiled_model, test_loader, torch.device("cuda"))
        auc = score_dataset(cfg, scores, clips, frames)
        latency = _microbench_pytorch(compiled_model, torch.device("cuda"))
        print(f"  AuC: {auc:.2f}%  |  Eval: {total_ms:.0f}ms  |  Latency: {latency:.2f} ms/batch")
        results.append({"name": "PyTorch compile", "auc": auc, "eval_ms": total_ms, "latency_ms": latency})
        del compiled_model
        torch.cuda.empty_cache()

    # ── 4. ONNX CPU ──────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  4. ONNX CPU (CPUExecutionProvider)")
    print("=" * 65)
    sess_cpu = ort.InferenceSession(
        str(onnx_path), providers=["CPUExecutionProvider"])
    scores, clips, frames, total_ms = _run_ort_inference(sess_cpu, test_loader)
    auc = score_dataset(cfg, scores, clips, frames)
    latency = _microbench_ort(sess_cpu)
    print(f"  AuC: {auc:.2f}%  |  Eval: {total_ms:.0f}ms  |  Latency: {latency:.2f} ms/batch")
    results.append({"name": "ONNX CPU", "auc": auc, "eval_ms": total_ms, "latency_ms": latency})
    del sess_cpu

    # ── 5. ONNX CUDA ─────────────────────────────────────────
    if "CUDAExecutionProvider" in ort.get_available_providers():
        print("\n" + "=" * 65)
        print("  5. ONNX CUDA (CUDAExecutionProvider)")
        print("=" * 65)
        sess_opts = ort.SessionOptions()
        sess_opts.log_severity_level = 3  # suppress shape-op CPU assignment warnings
        sess_cuda = ort.InferenceSession(
            str(onnx_path), sess_options=sess_opts,
            providers=["CUDAExecutionProvider"])
        scores, clips, frames, total_ms = _run_ort_inference(sess_cuda, test_loader)
        auc = score_dataset(cfg, scores, clips, frames)
        latency = _microbench_ort(sess_cuda)
        print(f"  AuC: {auc:.2f}%  |  Eval: {total_ms:.0f}ms  |  Latency: {latency:.2f} ms/batch")
        results.append({"name": "ONNX CUDA", "auc": auc, "eval_ms": total_ms, "latency_ms": latency})
        del sess_cuda

    # ── 6. ONNX TensorRT ─────────────────────────────────────
    if "TensorrtExecutionProvider" in ort.get_available_providers():
        print("\n" + "=" * 65)
        print("  6. ONNX TensorRT (TensorrtExecutionProvider)")
        print("=" * 65)
        trt_cache = MODELS_DIR / "trt_cache"
        trt_cache.mkdir(exist_ok=True)
        try:
            sess_opts = ort.SessionOptions()
            sess_opts.log_severity_level = 3
            sess_trt = ort.InferenceSession(
                str(onnx_path),
                sess_options=sess_opts,
                providers=[
                    ("TensorrtExecutionProvider", {
                        "trt_engine_cache_enable": True,
                        "trt_engine_cache_path": str(trt_cache),
                        "trt_fp16_enable": True,
                    }),
                ],
            )
            scores, clips, frames, total_ms = _run_ort_inference(sess_trt, test_loader)
            auc = score_dataset(cfg, scores, clips, frames)
            latency = _microbench_ort(sess_trt)
            print(f"  AuC: {auc:.2f}%  |  Eval: {total_ms:.0f}ms  |  Latency: {latency:.2f} ms/batch")
            results.append({"name": "ONNX TensorRT", "auc": auc,
                          "eval_ms": total_ms, "latency_ms": latency})
            del sess_trt
        except Exception as e:
            print(f"  [skip] TensorRT failed: {e}")

    # ── Summary ──────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  BENCHMARK SUMMARY")
    print("=" * 65)
    print(f"  {'Backend':<20}  {'AuC':>7}  {'Eval (ms)':>10}  {'Latency':>12}")
    print(f"  {'-'*20}  {'-'*7}  {'-'*10}  {'-'*12}")
    for r in results:
        print(f"  {r['name']:<20}  {r['auc']:>6.2f}%  {r['eval_ms']:>10.0f}  "
              f"{r['latency_ms']:>9.2f} ms")
    print("=" * 65)

    # ── Numerical consistency check ──────────────────────────
    aucs = [r["auc"] for r in results]
    max_diff = max(aucs) - min(aucs)
    if max_diff < 0.5:
        print(f"  ✓ All backends match (max AuC diff: {max_diff:.3f}%)")
    else:
        print(f"  ⚠ AuC variance detected (max diff: {max_diff:.3f}%)")
    print("=" * 65)


if __name__ == "__main__":
    main()
