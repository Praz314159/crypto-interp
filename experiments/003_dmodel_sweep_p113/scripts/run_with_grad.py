"""Fine-grained run with per-step recording of:
  - char_energy[t, k]   energy of character k in W_E
  - char_grad[t, k]     energy of character k in W_E.grad (post-backward, pre-step)
  - char_m[t, k]        energy of character k in Adam's first-moment m_t for W_E
  - char_v[t, k]        energy of character k in Adam's second-moment v_t for W_E

The "energy of character k in tensor X" is computed by projecting X[:, :p]
columns onto the multiplicative-Fourier character basis and summing squared
coefficients over d_model.

Goal: identify the step at which the gradient (or m_t / v_t) first becomes
character-aligned with the final K — i.e., when the dynamics first "know"
which characters to amplify.

Usage:
    python experiments/003_dmodel_sweep_p113/scripts/run_with_grad.py --seed 1
"""
from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim

from crypto_interp import data
from crypto_interp.interp.bases import multiplicative_fourier_basis
from crypto_interp.models import Transformer, TransformerConfig
from crypto_interp.training.loop import cross_entropy_high_precision

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ROOT / "datasets"
OUT_DIR = ROOT / "data" / "with_grad"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def build_char_basis(device):
    basis, names, g = multiplicative_fourier_basis(113, device=device)
    char_index = {}
    for i, nm in enumerate(names):
        m = re.match(r"mul (cos|sin) (\d+)", nm)
        if m:
            kk = int(m.group(2))
            char_index.setdefault(kk, []).append(i)
    return basis.double(), char_index


def char_energy(tensor_dp, basis, char_index, n_chars=56) -> np.ndarray:
    """tensor_dp: (d_model, p). Returns array of length n_chars."""
    coef = torch.einsum("kp,dp->kd", basis, tensor_dp.double())
    E = (coef ** 2).sum(dim=1).cpu().numpy()  # (n_basis,)
    out = np.zeros(n_chars)
    for k, rows in char_index.items():
        out[k - 1] = float(E[rows].sum())
    return out


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
    device = torch.device("cpu")

    ds = data.load_or_build(
        cache_dir=DATASETS, task="mul", p=113,
        frac_train=args.frac_train, seed=seed,
    )
    inputs = ds.inputs
    labels = ds.labels
    train_in, train_lab = inputs[ds.train_mask], labels[ds.train_mask]

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

    basis, char_index = build_char_basis(device)
    p = 113
    T = args.num_epochs

    char_E = np.zeros((T, 56))
    char_G = np.zeros((T, 56))
    char_M = np.zeros((T, 56))
    char_V = np.zeros((T, 56))
    train_losses = np.zeros(T)

    t0 = time.time()
    for epoch in range(T):
        logits = model(train_in)[:, -1, :ds.n_answer_tokens]
        loss = cross_entropy_high_precision(logits, train_lab, True)
        train_losses[epoch] = float(loss.item())

        loss.backward()
        # ---- record W_E, W_E.grad, Adam state BEFORE optimizer.step ----
        W_E = model.embed.W_E.detach()[:, :p]
        char_E[epoch] = char_energy(W_E, basis, char_index)
        grad = model.embed.W_E.grad.detach()[:, :p]
        char_G[epoch] = char_energy(grad, basis, char_index)

        # Adam state for embed.W_E param
        state = optimizer.state.get(model.embed.W_E, {})
        m = state.get("exp_avg")
        v = state.get("exp_avg_sq")
        if m is not None:
            char_M[epoch] = char_energy(m.detach()[:, :p], basis, char_index)
            char_V[epoch] = char_energy(v.detach()[:, :p], basis, char_index)

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        if epoch % 100 == 0:
            print(f"  step {epoch:4d}  log10 train {np.log10(loss.item()):.3f}  "
                  f"{time.time() - t0:.1f}s")

    out_path = OUT_DIR / f"seed{seed:02d}_with_grad.pt"
    torch.save({
        "seed": seed,
        "char_E": torch.tensor(char_E),
        "char_G": torch.tensor(char_G),
        "char_M": torch.tensor(char_M),
        "char_V": torch.tensor(char_V),
        "train_losses": torch.tensor(train_losses),
    }, out_path)
    print(f"Saved {out_path}  ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
