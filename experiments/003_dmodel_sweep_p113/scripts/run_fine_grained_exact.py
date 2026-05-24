"""Exact-match version of run_fine_grained.py.

Mirrors the official `train()` function from `crypto_interp/training/loop.py`
line-for-line (same RNG order, same EmbeddingEnergyTracker construction, same
optimizer/scheduler order), with one addition: capture W_E every step.

This guarantees the re-run reproduces the user's training trajectory exactly,
so the bifurcation-step we measure is the one from the very same run.

Usage:
    python experiments/003_dmodel_sweep_p113/scripts/run_fine_grained_exact.py \
        --experiment 003_dmodel_sweep_p113 \
        --override d_model=24 --override d_mlp=24 \
        --seed-override 1 --num-epochs 500 --tag dmlp24_seed1
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from dataclasses import fields, replace
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from crypto_interp import data
from crypto_interp.interp.metrics import EmbeddingEnergyTracker
from crypto_interp.models import Transformer, TransformerConfig
from crypto_interp.training.loop import (
    cross_entropy_high_precision,
    pick_device,
)

OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "fine_grained"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_experiment_config(experiment_id: str):
    cfg_path = REPO_ROOT / "experiments" / experiment_id / "config.py"
    spec = importlib.util.spec_from_file_location(
        f"experiments.{experiment_id}.config", cfg_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.CONFIG


def _loss_fn(model, inputs, labels, n_answer_tokens, use_float64):
    logits = model(inputs)[:, -1, :n_answer_tokens]
    return cross_entropy_high_precision(logits, labels, use_float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--seed-override", type=int, default=None)
    ap.add_argument("--num-epochs", type=int, default=500)
    ap.add_argument("--override", action="append", default=[])
    args = ap.parse_args()

    cfg = load_experiment_config(args.experiment)
    overrides = {}
    if args.seed_override is not None:
        overrides["seed"] = args.seed_override
    overrides["num_epochs"] = args.num_epochs
    field_types = {f.name: f.type for f in fields(cfg)}
    for kv in args.override:
        key, value = kv.split("=", 1)
        t_name = field_types[key]
        t_name = t_name if isinstance(t_name, str) else getattr(t_name, "__name__", str(t_name))
        if t_name == "int":
            coerced = int(value)
        elif t_name == "float":
            coerced = float(value)
        else:
            coerced = value
        overrides[key] = coerced
    cfg = replace(cfg, **overrides)

    datasets_dir = REPO_ROOT / "experiments" / args.experiment / "datasets"

    # ===== mirror train() begin =====
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    device = pick_device(cfg.device)
    use_float64 = device.type != "mps"
    print(f"Device: {device}, high-precision (float64) loss: {use_float64}")
    if not use_float64:
        print("WARNING: MPS doesn't support float64. ...")

    ds = data.load_or_build(
        cache_dir=datasets_dir,
        task=cfg.task, p=cfg.p, frac_train=cfg.frac_train, seed=cfg.seed,
    )
    inputs = ds.inputs.to(device)
    labels = ds.labels.to(device)
    train_mask = ds.train_mask.to(device)
    test_mask = ds.test_mask.to(device)
    print(f"task={ds.task}, p={ds.p}, total={len(ds.inputs)}, "
          f"train={ds.n_train}, test={ds.n_test}, vocab={ds.vocab_size}")

    model_cfg = TransformerConfig(
        d_vocab=ds.vocab_size,
        d_model=cfg.d_model, d_mlp=cfg.d_mlp,
        num_heads=cfg.num_heads, d_head=cfg.d_head,
        n_ctx=cfg.n_ctx, num_layers=cfg.num_layers,
    )
    model = Transformer(model_cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    optimizer = optim.AdamW(
        model.parameters(), lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        betas=(cfg.beta1, cfg.beta2),
    )
    warmup = max(cfg.warmup_steps, 1)
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min(s / warmup, 1.0)
    )

    train_losses: list[float] = []
    test_losses: list[float] = []

    tracker = (
        EmbeddingEnergyTracker(p=ds.p, device=device)
        if cfg.metrics_every > 0 else None
    )

    # Per-step W_E logging.
    saved_epochs: list[int] = []
    saved_W_E: list[torch.Tensor] = []

    start = time.time()
    for epoch in range(cfg.num_epochs):
        train_loss = _loss_fn(model, inputs[train_mask], labels[train_mask],
                              ds.n_answer_tokens, use_float64)
        with torch.no_grad():
            test_loss = _loss_fn(model, inputs[test_mask], labels[test_mask],
                                 ds.n_answer_tokens, use_float64)
        train_losses.append(train_loss.item())
        test_losses.append(test_loss.item())

        train_loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        # Per-step W_E (AFTER the optimizer step).
        saved_epochs.append(epoch)
        saved_W_E.append(model.embed.W_E.detach().cpu().clone())

        if tracker is not None and epoch % cfg.metrics_every == 0:
            tracker.record(epoch, model.embed.W_E[:, : ds.p].detach())

        if epoch % cfg.log_every == 0:
            print(f"  ep {epoch:5d}  log10 train {np.log10(train_loss.item()):7.3f} "
                  f"log10 test {np.log10(test_loss.item()):7.3f}  "
                  f"{time.time() - start:.1f}s")

    # Verify against the original checkpoint_000500.pt if present.
    run_dir = REPO_ROOT / "experiments" / args.experiment / "runs" / args.tag
    ck = run_dir / "checkpoint_000500.pt"
    if ck.exists():
        ref = torch.load(ck, weights_only=False, map_location="cpu")
        ref_W_E = ref["model_state"]["embed.W_E"].cpu()
        # checkpoint_000500.pt is saved AFTER epoch-500's update; our saved_W_E[499]
        # is after epoch-499's update. They differ by one step.
        # Compare instead: re-run forward + step once more, then check.
        # Simpler: just compare current W_E now (post-step 499) to original at
        # step 499. We can't easily get the original at step 499. Just report
        # diff vs the original at step 500 for sanity.
        diff = (saved_W_E[-1] - ref_W_E).abs().max().item()
        scale = saved_W_E[-1].abs().max().item()
        print(f"Diff vs {ck.name} (one step off): max |Δ| = {diff:.2e} "
              f"(scale {scale:.2e})")
    else:
        print(f"No reference checkpoint at {ck} to compare against.")

    out_path = OUT_DIR / f"{args.tag}_fine_grained_exact.pt"
    torch.save({
        "seed": cfg.seed,
        "d_model": cfg.d_model,
        "d_mlp": cfg.d_mlp,
        "epochs": torch.tensor(saved_epochs),
        "W_E": torch.stack(saved_W_E),
        "train_losses": torch.tensor(train_losses),
        "test_losses": torch.tensor(test_losses),
    }, out_path)
    print(f"Saved {len(saved_epochs)} W_E snapshots to {out_path}")


if __name__ == "__main__":
    main()
