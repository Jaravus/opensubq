#!/usr/bin/env python3
"""
train.py — Minimal SubQ training loop.

Usage
-----
# Tiny sanity-check (synthetic data, no GPU required):
    python train.py --preset tiny --data synthetic --max-steps 200

# Tier-1 scale on a text file with a GPU:
    python train.py --preset mistral_7b --data-file corpus.txt --batch-size 4

Run ``python train.py --help`` for the full option list.

Features
--------
* bfloat16 / float16 mixed-precision via ``torch.autocast`` (auto-disabled on CPU)
* AdamW optimiser with cosine LR schedule + linear warmup
* Gradient clipping (``--grad-clip``, default 1.0)
* Periodic checkpoint save and resume (``--checkpoint-dir``)
* Eval loss on a held-out split every ``--eval-interval`` steps
* TensorBoard-compatible CSV loss log (``--log-file``)
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import math
import os
import time
from pathlib import Path
from typing import Iterator, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from opensubq import SubQConfig, SubQModel
from opensubq.data import (
    CharDataset,
    TiktokenDataset,
    make_split_loaders,
    make_synthetic_datasets,
)


# --------------------------------------------------------------------------- #
# Config presets                                                                #
# --------------------------------------------------------------------------- #

def _make_config(preset: str) -> SubQConfig:
    if preset == "tiny":
        return SubQConfig(
            vocab_size=256,
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=128,
            window_size=32,
            num_global_tokens=2,
            top_k_sparse=16,
            routing_rank=4,
            dropout=0.0,
            attention_dropout=0.0,
        )
    if preset == "mistral_7b":
        return SubQConfig.mistral_7b()
    if preset == "mimo_v2_flash":
        return SubQConfig.mimo_v2_flash()
    raise ValueError(
        f"Unknown preset '{preset}'.  Choose from: tiny, mistral_7b, mimo_v2_flash"
    )


# --------------------------------------------------------------------------- #
# LR schedule: linear warmup → cosine decay                                    #
# --------------------------------------------------------------------------- #

def _cosine_lr(
    step: int,
    warmup_steps: int,
    max_steps: int,
    max_lr: float,
    min_lr: float,
) -> float:
    """Return the learning rate for ``step`` under a cosine schedule."""
    if step < warmup_steps:
        return max_lr * step / max(1, warmup_steps)
    if step >= max_steps:
        return min_lr
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


# --------------------------------------------------------------------------- #
# Checkpoint helpers                                                            #
# --------------------------------------------------------------------------- #

def _save_checkpoint(
    model: SubQModel,
    optimizer: torch.optim.Optimizer,
    step: int,
    loss: float,
    path: Path,
) -> None:
    """Save model + optimizer state to ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "loss": loss,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": dataclasses.asdict(model.config),
        },
        path,
    )
    print(f"  [ckpt] saved → {path}")


def _load_checkpoint(
    path: Path,
    model: SubQModel,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> int:
    """Load model + optimizer state from ``path``.  Returns the step to resume from."""
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    step = ckpt["step"]
    print(f"  [ckpt] resumed from {path} at step {step}")
    return step


# --------------------------------------------------------------------------- #
# Eval helper                                                                   #
# --------------------------------------------------------------------------- #

@torch.no_grad()
def _evaluate(
    model: SubQModel,
    loader: DataLoader,
    device: torch.device,
    amp_dtype: Optional[torch.dtype],
    max_batches: int = 50,
) -> float:
    """Return mean eval loss (cross-entropy) over up to ``max_batches`` batches."""
    model.eval()
    total_loss, n_batches = 0.0, 0
    for input_ids, labels in loader:
        input_ids = input_ids.to(device)
        labels    = labels.to(device)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            loss, _ = model(input_ids, labels=labels)
        total_loss += loss.item()
        n_batches  += 1
        if n_batches >= max_batches:
            break
    model.train()
    return total_loss / max(1, n_batches)


# --------------------------------------------------------------------------- #
# Infinite DataLoader iterator                                                  #
# --------------------------------------------------------------------------- #

def _infinite(loader: DataLoader) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
    """Yield batches from ``loader`` indefinitely, reshuffling each epoch."""
    while True:
        yield from loader


# --------------------------------------------------------------------------- #
# Main training loop                                                            #
# --------------------------------------------------------------------------- #

def train(args: argparse.Namespace) -> None:
    # ------------------------------------------------------------------ #
    # Device + dtype                                                       #
    # ------------------------------------------------------------------ #
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    # bfloat16 on CUDA; float16 is not universally stable on all GPUs.
    # CPU autocast is skipped (float32 throughout on CPU).
    if device.type == "cuda":
        amp_dtype: Optional[torch.dtype] = torch.bfloat16
    else:
        amp_dtype = None  # no autocast on CPU

    print(f"Device : {device}  |  AMP dtype : {amp_dtype or 'float32 (no autocast)'}")

    # ------------------------------------------------------------------ #
    # Config + model                                                       #
    # ------------------------------------------------------------------ #
    config = _make_config(args.preset)
    model  = SubQModel(config).to(device)
    n_params = model.num_parameters()
    print(f"Model  : {args.preset}  |  {n_params:,} trainable parameters")

    # ------------------------------------------------------------------ #
    # Datasets + loaders                                                   #
    # ------------------------------------------------------------------ #
    if args.data == "synthetic":
        train_ds, val_ds = make_synthetic_datasets(
            vocab_size=config.vocab_size,
            seq_len=args.seq_len,
            total_tokens=args.synthetic_tokens,
            seed=args.seed,
        )
    elif args.data == "file":
        if args.data_file is None:
            raise ValueError("--data-file is required when --data=file")
        if args.tokeniser == "char":
            train_ds, val_ds = CharDataset.from_file(args.data_file, args.seq_len)
        else:
            train_ds, val_ds = TiktokenDataset.from_file(
                args.data_file, args.seq_len, encoding=args.tokeniser
            )
    else:
        raise ValueError(f"Unknown --data option: {args.data!r}")

    train_loader, val_loader = make_split_loaders(
        train_ds, val_ds, batch_size=args.batch_size, num_workers=0
    )
    print(
        f"Data   : {len(train_ds):,} train tokens / {len(val_ds):,} val tokens  "
        f"| seq_len={args.seq_len}  batch={args.batch_size}"
    )

    # ------------------------------------------------------------------ #
    # Optimizer                                                            #
    # ------------------------------------------------------------------ #
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
        eps=1e-8,
    )

    # ------------------------------------------------------------------ #
    # Resume from checkpoint (if requested)                               #
    # ------------------------------------------------------------------ #
    start_step = 0
    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.exists():
            start_step = _load_checkpoint(resume_path, model, optimizer, device)
        else:
            print(f"  [warn] checkpoint not found: {resume_path} — starting fresh")

    # ------------------------------------------------------------------ #
    # CSV loss log                                                         #
    # ------------------------------------------------------------------ #
    log_path: Optional[Path] = None
    log_file = None
    log_writer = None
    if args.log_file:
        log_path = Path(args.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file  = open(log_path, "w", newline="")
        log_writer = csv.writer(log_file)
        log_writer.writerow(["step", "train_loss", "val_loss", "lr", "elapsed_s"])
        log_file.flush()

    # ------------------------------------------------------------------ #
    # Training loop                                                        #
    # ------------------------------------------------------------------ #
    model.train()
    data_iter  = _infinite(train_loader)
    t0         = time.perf_counter()
    step       = start_step
    warmup     = int(args.max_steps * args.warmup_frac)
    min_lr     = args.learning_rate * args.min_lr_ratio

    print(
        f"\nTraining for {args.max_steps} steps  "
        f"(warmup={warmup}, grad_clip={args.grad_clip})\n"
        f"{'Step':>8}  {'Train loss':>12}  {'Val loss':>10}  {'LR':>10}  {'Elapsed':>9}"
    )
    print("-" * 58)

    first_loss: Optional[float] = None
    last_loss:  Optional[float] = None

    while step < args.max_steps:
        # -- LR update -------------------------------------------------- #
        lr = _cosine_lr(step, warmup, args.max_steps, args.learning_rate, min_lr)
        for g in optimizer.param_groups:
            g["lr"] = lr

        # -- Forward + backward ----------------------------------------- #
        input_ids, labels = next(data_iter)
        input_ids = input_ids.to(device)
        labels    = labels.to(device)

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            loss, _ = model(input_ids, labels=labels)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        step += 1
        train_loss = loss.item()

        if first_loss is None:
            first_loss = train_loss
        last_loss = train_loss

        # -- Logging ---------------------------------------------------- #
        elapsed = time.perf_counter() - t0
        log_val_loss = ""

        if step % args.log_interval == 0 or step == args.max_steps:
            val_loss = _evaluate(model, val_loader, device, amp_dtype)
            log_val_loss = f"{val_loss:10.4f}"
            print(
                f"{step:>8}  {train_loss:12.4f}  {log_val_loss}  {lr:10.2e}  {elapsed:8.1f}s"
            )
            if log_writer:
                log_writer.writerow([step, train_loss, val_loss, lr, f"{elapsed:.1f}"])
                log_file.flush()  # type: ignore[union-attr]
        else:
            if step % max(1, args.log_interval // 5) == 0:
                print(
                    f"{step:>8}  {train_loss:12.4f}  {'':>10}  {lr:10.2e}  {elapsed:8.1f}s"
                )
            if log_writer:
                log_writer.writerow([step, train_loss, "", lr, f"{elapsed:.1f}"])
                log_file.flush()  # type: ignore[union-attr]

        # -- Checkpoint ------------------------------------------------- #
        if args.checkpoint_dir and step % args.checkpoint_interval == 0:
            ckpt_path = Path(args.checkpoint_dir) / f"step_{step:07d}.pt"
            _save_checkpoint(model, optimizer, step, train_loss, ckpt_path)

    # ------------------------------------------------------------------ #
    # Final report                                                         #
    # ------------------------------------------------------------------ #
    print("-" * 58)
    if first_loss is not None and last_loss is not None:
        delta = first_loss - last_loss
        print(
            f"First loss : {first_loss:.4f}  →  Last loss : {last_loss:.4f}  "
            f"(Δ = {delta:+.4f})"
        )
        if delta > 0:
            print("✓ Loss decreased — training is working.")
        else:
            print("⚠ Loss did not decrease — check data and hyperparameters.")

    if args.checkpoint_dir:
        final_path = Path(args.checkpoint_dir) / "final.pt"
        _save_checkpoint(model, optimizer, step, last_loss or 0.0, final_path)

    if log_file:
        log_file.close()

    print(f"\nDone in {time.perf_counter() - t0:.1f}s")


# --------------------------------------------------------------------------- #
# CLI                                                                           #
# --------------------------------------------------------------------------- #

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SubQ minimal training loop",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model
    p.add_argument(
        "--preset", default="tiny",
        choices=["tiny", "mistral_7b", "mimo_v2_flash"],
        help="Named config preset.",
    )

    # Data
    p.add_argument(
        "--data", default="synthetic",
        choices=["synthetic", "file"],
        help="Data source: 'synthetic' (no extra deps) or 'file' (requires --data-file).",
    )
    p.add_argument("--data-file", default=None, help="Path to a plain-text corpus file.")
    p.add_argument(
        "--tokeniser", default="char",
        choices=["char", "gpt2", "cl100k_base"],
        help="Tokeniser to use when --data=file.",
    )
    p.add_argument("--seq-len", type=int, default=128, help="Context window length.")
    p.add_argument(
        "--synthetic-tokens", type=int, default=50_000,
        help="Corpus size when --data=synthetic.",
    )
    p.add_argument("--seed", type=int, default=42, help="RNG seed.")

    # Optimiser
    p.add_argument("--learning-rate", type=float, default=3e-4, help="Peak learning rate.")
    p.add_argument("--min-lr-ratio", type=float, default=0.1,
                   help="Min LR = learning-rate × min-lr-ratio.")
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clip norm.")
    p.add_argument("--warmup-frac", type=float, default=0.05,
                   help="Fraction of max-steps used for LR warmup.")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-steps", type=int, default=200, help="Total training steps.")

    # Infra
    p.add_argument("--device", default=None, help="e.g. 'cpu', 'cuda', 'cuda:1'.")
    p.add_argument("--checkpoint-dir", default=None, help="Directory for checkpoints.")
    p.add_argument("--checkpoint-interval", type=int, default=500,
                   help="Save checkpoint every N steps.")
    p.add_argument("--resume", default=None, help="Path to checkpoint to resume from.")
    p.add_argument("--log-file", default=None, help="Path to write CSV loss log.")
    p.add_argument("--log-interval", type=int, default=50,
                   help="Steps between eval + full log lines.")

    return p.parse_args()


if __name__ == "__main__":
    train(_parse_args())
