"""Fill in W_E at every step in [0, 500) — the gap between checkpoint_000000.pt
and checkpoint_000500.pt of an existing run.

We re-run training from the same seed (deterministic given seed + dataset +
optimizer config), saving W_E every step. The final step (499) should match
the W_E in the existing checkpoint_000500.pt up to numerical noise; we print a
diff to verify.

Usage:
    python experiments/003_dmodel_sweep_p113/scripts/run_fine_grained.py --seed 1
    python experiments/003_dmodel_sweep_p113/scripts/run_fine_grained.py --seed 8
    # or all seeds:
    for s in 1 2 3 4 5 6 7 8 9 10 11 12 13 14; do
        python experiments/003_dmodel_sweep_p113/scripts/run_fine_grained.py --seed $s
    done
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from crypto_interp import data
from crypto_interp.interp.metrics import EmbeddingEnergyTracker
from crypto_interp.models import Transformer, TransformerConfig
from crypto_interp.training.loop import cross_entropy_high_precision

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ROOT / "datasets"
OUT_DIR = ROOT / "data" / "fine_grained"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def should_save(epoch: int) -> bool:
    return True  # every step within the (short) window


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--num-epochs", type=int, default=500)
    ap.add_argument("--d-model", type=int, default=24)
    ap.add_argument("--d-mlp", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1.0)
    ap.add_argument("--frac-train", type=float, default=0.3)
    args = ap.parse_args()

    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_float64 = device.type != "mps"
    print(f"Device: {device}")

    ds = data.load_or_build(
        cache_dir=DATASETS, task="mul", p=113,
        frac_train=args.frac_train, seed=seed,
    )
    inputs = ds.inputs.to(device)
    labels = ds.labels.to(device)
    train_mask = ds.train_mask.to(device)
    test_mask = ds.test_mask.to(device)
    print(f"task=mul p=113 train={ds.n_train} test={ds.n_test}")

    cfg = TransformerConfig(
        d_vocab=ds.vocab_size, d_model=args.d_model, d_mlp=args.d_mlp,
        num_heads=4, d_head=32, n_ctx=3, num_layers=1,
    )
    model = Transformer(cfg).to(device)

    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr,
        weight_decay=args.weight_decay, betas=(0.9, 0.98),
    )
    warmup = 10
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min(s / warmup, 1.0)
    )

    # Construct EmbeddingEnergyTracker so that any RNG state it consumes
    # (currently none, but kept here to mirror the main train() pipeline)
    # matches train.py exactly.
    _tracker = EmbeddingEnergyTracker(p=ds.p, device=device)

    n_answer = ds.n_answer_tokens
    train_in, train_lab = inputs[train_mask], labels[train_mask]
    test_in, test_lab = inputs[test_mask], labels[test_mask]

    saved_epochs: list[int] = []
    saved_W_E: list[torch.Tensor] = []
    train_losses: list[float] = []
    test_losses: list[float] = []

    t0 = time.time()
    for epoch in range(args.num_epochs):
        logits = model(train_in)[:, -1, :n_answer]
        train_loss = cross_entropy_high_precision(logits, train_lab, use_float64)
        with torch.no_grad():
            test_logits = model(test_in)[:, -1, :n_answer]
            test_loss = cross_entropy_high_precision(test_logits, test_lab, use_float64)
        train_losses.append(float(train_loss.item()))
        test_losses.append(float(test_loss.item()))

        train_loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        if should_save(epoch):
            saved_epochs.append(epoch)
            saved_W_E.append(model.embed.W_E.detach().cpu().clone())

        if epoch % 1000 == 0:
            el = time.time() - t0
            print(f"  epoch {epoch:6d} | log10 train {np.log10(train_loss.item()):7.3f} "
                  f"| log10 test {np.log10(test_loss.item()):7.3f} | {el:.1f}s")

    # Verify the last in-window W_E matches the existing epoch-500 checkpoint
    # of the matching configuration, if one exists. Baseline d_mlp=512 lives at
    # `dmodel_24_seedN`, others at `dmodel_24_dmlp_M_seedN`.
    if args.d_mlp == 512:
        existing = ROOT / "runs" / f"dmodel_{args.d_model}_seed{seed}" / "checkpoint_000500.pt"
    else:
        existing = ROOT / "runs" / f"dmodel_{args.d_model}_dmlp_{args.d_mlp}_seed{seed}" / "checkpoint_000500.pt"
    if existing.exists():
        ref = torch.load(existing, weights_only=False, map_location="cpu")
        ref_W_E = ref["model_state"]["embed.W_E"]
        diff = (saved_W_E[-1] - ref_W_E.cpu()).abs().max().item()
        scale = saved_W_E[-1].abs().max().item()
        print(f"Diff vs {existing.name}: max |Δ| = {diff:.2e} "
              f"(scale {scale:.2e}, rel {diff/scale:.2e})")
    else:
        print(f"No existing checkpoint at {existing} to compare against.")

    # Save. Include d_mlp in filename to avoid clobbering across configurations.
    if args.d_mlp == 512:
        out_path = OUT_DIR / f"seed{seed:02d}_fine_grained.pt"
    else:
        out_path = OUT_DIR / f"dmlp{args.d_mlp:03d}_seed{seed:02d}_fine_grained.pt"
    torch.save({
        "seed": seed,
        "d_model": args.d_model,
        "d_mlp": args.d_mlp,
        "epochs": torch.tensor(saved_epochs),
        "W_E": torch.stack(saved_W_E),  # (T, d_model, vocab)
        "train_losses": torch.tensor(train_losses),
        "test_losses": torch.tensor(test_losses),
    }, out_path)
    print(f"Saved {len(saved_epochs)} W_E snapshots to {out_path}")
    print(f"Total time {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
