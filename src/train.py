"""Training loop for STG-NF model.

Usage: python -m src.train
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
from tqdm import tqdm

from .config import Config
from .dataset import get_test_loader, get_train_loader, normalize_batch
from .evaluate import evaluate
from .model import STG_NF


def train(cfg: Config) -> Path:
    """Train STG-NF model from scratch with per-epoch evaluation.

    Args:
        cfg: Configuration object.

    Returns:
        Path to the best checkpoint (highest AuC).
    """
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print(f"[train] Device: {device}")
    print(f"[train] Seed: {cfg.seed}")

    # ── Reproducibility ──────────────────────────────────────
    torch.manual_seed(cfg.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(cfg.seed)

    # Ensure CPU uses all physical cores (Windows can silently default to 1)
    import os
    n_threads = max(1, os.cpu_count() // 2)
    torch.set_num_threads(n_threads)
    print(f"[train] CPU threads: {torch.get_num_threads()}")

    # ── Data ─────────────────────────────────────────────────
    print("[train] Loading data...")
    t0 = time.perf_counter()
    train_loader = get_train_loader(cfg)
    test_loader = get_test_loader(cfg)  # created once, reused every epoch
    print(f"[train] Data loaded in {time.perf_counter() - t0:.1f}s "
          f"(train={len(train_loader.dataset)}, test={len(test_loader.dataset)})")

    # ── Model ────────────────────────────────────────────────
    model = STG_NF(cfg).to(device)
    model = torch.compile(model)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[train] Model created: {param_count} trainable parameters")

    # ── Optimizer (Adamax, matching original) ────────────────
    optimizer = torch.optim.Adamax(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    # ── Scheduler (Cosine Annealing with Warmup) ─────────────
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, total_iters=cfg.warmup_epochs
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.epochs - cfg.warmup_epochs,
        eta_min=1e-6
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[cfg.warmup_epochs]
    )

    # ── Training loop ────────────────────────────────────────
    best_auc: float = 0.0
    best_checkpoint: Path = cfg.checkpoint_dir / "stg_nf_best.pt"
    epoch_results: list[dict] = []

    for epoch in range(cfg.epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        current_lr = scheduler.get_last_lr()[0]

        t_epoch = time.perf_counter()
        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{cfg.epochs}",
            leave=True,
            file=sys.stdout,
            mininterval=1.0,
        )

        use_profiler = getattr(cfg, "profile_system", False) and epoch == 0
        if use_profiler:
            prof = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                schedule=torch.profiler.schedule(wait=2, warmup=3, active=100, repeat=1),
                record_shapes=True,
                profile_memory=True,
                with_stack=True
            )
            prof.start()

        for batch in pbar:
            pose, _label = batch
            pose = pose.to(device, non_blocking=True)
            pose = normalize_batch(pose)  # batch-level normalization

            # ── ActNorm init on very first batch ──
            if epoch == 0 and n_batches == 0:
                model.initialize_actnorms(pose)

            nll = model(pose)  # (B,)
            loss = nll.mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()
            n_batches += 1
            pbar.set_postfix(
                {"loss": f"{loss.item():.4f}", "lr": f"{current_lr:.2e}"},
                refresh=False,
            )
            
            if use_profiler:
                prof.step()

        if use_profiler:
            prof.stop()
            summary_path = cfg.checkpoint_dir / "pytorch_profiler.txt"
            with open(summary_path, "w") as f:
                f.write(prof.key_averages().table(sort_by="cpu_time_total", row_limit=30))
                f.write("\n\n" + "="*80 + "\n\n")
                f.write(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))
            print(f"\n[profiler] Human-readable PyTorch trace saved to {summary_path}")

        # Step scheduler at the end of the epoch
        scheduler.step()

        avg_loss = epoch_loss / max(n_batches, 1)
        train_time = time.perf_counter() - t_epoch

        # ── Save checkpoint ──────────────────────────────────
        ckpt_path = cfg.checkpoint_dir / f"stg_nf_epoch{epoch + 1}.pt"
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": avg_loss,
            "lr": current_lr,
        }, ckpt_path)

        # ── Per-epoch evaluation (reuses model & test_loader) ──
        t_eval = time.perf_counter()
        auc = evaluate(cfg, model=model, device=device, test_loader=test_loader,
                        silent=True)
        eval_time = time.perf_counter() - t_eval

        is_best = auc > best_auc
        if is_best:
            best_auc = auc
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_loss,
                "lr": current_lr,
                "auc": auc,
            }, best_checkpoint)

        marker = " ★" if is_best else ""
        print(f"  → Epoch {epoch + 1}: loss={avg_loss:.4f}, AuC={auc:.2f}%{marker}, "
              f"lr={current_lr:.2e}, train={train_time:.1f}s, eval={eval_time:.1f}s")

        epoch_results.append({
            "epoch": epoch + 1,
            "loss": avg_loss,
            "auc": auc,
            "lr": current_lr,
        })

    # ── Summary ──────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Training Summary")
    print(f"{'='*60}")
    print(f"  {'Epoch':>5}  {'Loss':>8}  {'AuC':>8}  {'LR':>10}")
    print(f"  {'-'*5}  {'-'*8}  {'-'*8}  {'-'*10}")
    for r in epoch_results:
        marker = " ★" if r["auc"] == best_auc else ""
        print(f"  {r['epoch']:>5}  {r['loss']:>8.4f}  {r['auc']:>7.2f}%  {r['lr']:>10.2e}{marker}")
    print(f"{'='*60}")
    print(f"  Best AuC: {best_auc:.2f}% → {best_checkpoint}")
    print(f"{'='*60}")

    return best_checkpoint


def main() -> None:
    """Entry point for training + evaluation."""
    cfg = Config()
    
    if getattr(cfg, "profile_system", False):
        from .profiler import SystemProfiler
        profile_path = cfg.checkpoint_dir / "system_profile.csv"
        with SystemProfiler(out_path=profile_path, interval=1.0):
            train(cfg)
    else:
        train(cfg)


if __name__ == "__main__":
    main()
