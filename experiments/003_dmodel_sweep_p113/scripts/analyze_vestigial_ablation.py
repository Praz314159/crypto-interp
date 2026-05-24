"""Ablate individual characters from W_E at inference time; measure test-loss impact.

For each character k ∈ K of a trained model, zero out W_E's projection onto the
cos_k and sin_k basis directions, leaving all other components untouched.
Recompute test loss with the ablated W_E. The ablation that does no harm
identifies a character that's "vestigial" — present in the embedding but doing
no algorithmic work. Ablations that do harm identify characters that are
load-bearing.

Reference (baseline) loss is the unablated test loss.

Usage:
    python experiments/003_dmodel_sweep_p113/scripts/analyze_vestigial_ablation.py \
        --run-dir experiments/003_dmodel_sweep_p113/runs/dmodel_24_dmlp_32_seed1
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import torch

from crypto_interp.interp.bases import multiplicative_fourier_basis
from crypto_interp.interp.load import load_run
from crypto_interp.training.loop import cross_entropy_high_precision


def build_basis_indexed():
    basis, names, _ = multiplicative_fourier_basis(113)
    char_idx = {}
    for i, nm in enumerate(names):
        m = re.match(r"mul (cos|sin) (\d+)", nm)
        if m:
            kk = int(m.group(2))
            char_idx.setdefault(kk, []).append(i)
    return basis.double(), names, char_idx


def char_energy(W_E_dp, basis, char_idx, n_chars=56):
    coef = torch.einsum("kp,dp->kd", basis, W_E_dp.double())
    E = (coef ** 2).sum(dim=1).cpu().numpy()
    out = np.zeros(n_chars)
    for k, rs in char_idx.items():
        out[k - 1] = float(E[rs].sum())
    return out


def ablate_character(W_E_full, basis, char_idx, k_remove, p=113):
    """Return a new W_E with the projection onto cos_k_remove and sin_k_remove
    zeroed out (on the first p columns; '=' token column at index p is left alone).
    """
    W_E_new = W_E_full.clone().double()
    W_E_v = W_E_new[:, :p].clone()
    rows = char_idx[k_remove]  # cos and sin basis-row indices for this character
    # Compute coefficients along the cos and sin rows for each d_model row.
    # coef[r, d] = sum_a basis[r, a] * W_E_v[d, a]
    for r in rows:
        b = basis[r].double()           # (p,)
        coef = (W_E_v.double() @ b)     # (d_model,) coefficient along basis row r
        W_E_v = W_E_v - coef[:, None] * b[None, :]
    W_E_new[:, :p] = W_E_v
    return W_E_new


def compute_test_loss(model, ds):
    inputs = ds.inputs
    labels = ds.labels
    mask = ds.test_mask.bool()
    with torch.no_grad():
        logits = model(inputs[mask])[:, -1, :ds.n_answer_tokens]
    return float(cross_entropy_high_precision(logits, labels[mask], True).item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args()
    run_dir = Path(args.run_dir).resolve()
    ck = sorted(run_dir.glob("checkpoint_*.pt"))
    if not ck:
        raise SystemExit(f"No checkpoint in {run_dir}")
    print(f"Loading {ck[-1]}")
    model, ds, _ = load_run(ck[-1])
    model.eval()
    basis, names, char_idx = build_basis_indexed()
    p = ds.p

    # Identify final K from W_E.
    W_E_orig = model.embed.W_E.detach().clone()
    char_E = char_energy(W_E_orig[:, :p], basis, char_idx)
    K = sorted([k + 1 for k, e in enumerate(char_E) if e >= 0.05 * char_E.max()])
    print(f"K = {K}")
    print(f"per-K energies: {[f'k={k}: {char_E[k - 1]:.3f}' for k in K]}")

    # Baseline test loss.
    base = compute_test_loss(model, ds)
    print(f"\nBaseline test loss: {base:.4e}")

    # Ablate each K character one at a time and measure.
    print(f"\n{'k':>4} {'order':>5} {'energy':>10} {'ablated test':>14} "
          f"{'Δlog10':>8}  interpretation")
    rows = []
    for k in K:
        W_E_ab = ablate_character(W_E_orig, basis, char_idx, k, p=p)
        with torch.no_grad():
            model.embed.W_E.copy_(W_E_ab.to(model.embed.W_E.dtype))
        ablated = compute_test_loss(model, ds)
        # Restore.
        with torch.no_grad():
            model.embed.W_E.copy_(W_E_orig)
        from math import gcd
        o = 112 // gcd(k, 112)
        delta_log = np.log10(ablated) - np.log10(base)
        interp = "vestigial" if delta_log < 0.5 else ("load-bearing" if delta_log < 2 else "essential")
        rows.append((k, o, char_E[k - 1], ablated, delta_log, interp))
        print(f"{k:>4} {o:>5} {char_E[k - 1]:>10.3f} {ablated:>14.4e} "
              f"{delta_log:>+8.3f}  {interp}")

    # Control: ablate a random non-K character (one of the strongest non-K).
    nonK_top = int(np.argmax([char_E[i] if (i + 1) not in K else -1 for i in range(56)])) + 1
    W_E_ab = ablate_character(W_E_orig, basis, char_idx, nonK_top, p=p)
    with torch.no_grad():
        model.embed.W_E.copy_(W_E_ab.to(model.embed.W_E.dtype))
    ctrl = compute_test_loss(model, ds)
    with torch.no_grad():
        model.embed.W_E.copy_(W_E_orig)
    print(f"\nControl: ablate top non-K char k={nonK_top} "
          f"(energy {char_E[nonK_top - 1]:.3f}): test loss {ctrl:.4e}  "
          f"Δlog10={np.log10(ctrl) - np.log10(base):+.3f}")


if __name__ == "__main__":
    main()
