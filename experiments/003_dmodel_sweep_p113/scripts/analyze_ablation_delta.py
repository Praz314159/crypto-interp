"""For each character k ∈ K, compute the direct ablation contribution:
    Δ_k(a, b, c) = logit_full(c|a, b) - logit_ablated(c|a, b)
where ablated removes χ_k from W_E. This is the EXACT contribution of
character k in W_E to each logit, with no projection assumptions.

If k's contribution is a clean character product χ_k(ab) · χ_k(c)^*, then
Δ_k(a, b, c) should depend on (a, b) only through ab, and should reduce to a
1D function of (log c - log ab) mod (p-1): a cosine at frequency k.

For each k, we visualize:
  - Δ_k(ab, c) heatmap, dlog-sorted on both axes. Character product = anti-
    diagonal stripes; peak on c=ab line.
  - 1D reduction: average Δ_k(ab, c) along constant (log c - log ab). Should
    be ~ cos[2π k Δlog / (p-1)] if k is a pure character product.

Usage:
    python experiments/003_dmodel_sweep_p113/scripts/analyze_ablation_delta.py \
        --run-dir experiments/003_dmodel_sweep_p113/runs/dmodel_24_dmlp_32_seed1 \
        --ks 3,10,20,51,6
"""
from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from crypto_interp.interp.bases import (
    multiplicative_fourier_basis,
    discrete_log_table,
)
from crypto_interp.interp.load import load_run


def order_of(k, n=112):
    return n // math.gcd(k, n)


def build_basis_indexed():
    basis, names, _ = multiplicative_fourier_basis(113)
    char_idx = {}
    for i, nm in enumerate(names):
        m = re.match(r"mul (cos|sin) (\d+)", nm)
        if m:
            kk = int(m.group(2))
            char_idx.setdefault(kk, []).append(i)
    return basis.double(), char_idx


def ablate_W_E(W_E, k, basis, char_idx, p=113):
    W_E_new = W_E.clone().double()
    W_E_v = W_E_new[:, :p].clone()
    for r in char_idx[k]:
        b = basis[r].double()
        coef = (W_E_v @ b)
        W_E_v = W_E_v - coef[:, None] * b[None, :]
    W_E_new[:, :p] = W_E_v
    return W_E_new


def compute_logits_grid(model, ds, W_E_override=None):
    p = ds.p
    aa, bb = torch.meshgrid(torch.arange(1, p), torch.arange(1, p), indexing="ij")
    eq = torch.full_like(aa, ds.eq_token)
    inputs = torch.stack([aa, bb, eq], dim=-1).reshape(-1, 3)
    if W_E_override is not None:
        orig = model.embed.W_E.detach().clone()
        with torch.no_grad():
            model.embed.W_E.copy_(W_E_override.to(model.embed.W_E.dtype))
    with torch.no_grad():
        logits = model(inputs)[:, -1, :].double()
    if W_E_override is not None:
        with torch.no_grad():
            model.embed.W_E.copy_(orig)
    return logits.reshape(p - 1, p - 1, -1)


def reduce_to_ab(delta_abc, p):
    """delta_abc: (p-1, p-1, vocab) indexed by (a-1, b-1, c).
    Returns f(ab, c) of shape (p-1, vocab) averaged over (a, b) with the same ab."""
    # Compute ab for each (a, b) pair (a, b ∈ 1..p-1).
    aa = np.arange(1, p)
    bb = np.arange(1, p)
    ab_grid = (aa[:, None] * bb[None, :]) % p   # (p-1, p-1)
    out = np.zeros((p - 1, delta_abc.shape[-1]))
    counts = np.zeros(p - 1)
    arr = delta_abc.cpu().numpy()
    for a_i in range(p - 1):
        for b_i in range(p - 1):
            ab = ab_grid[a_i, b_i]
            if ab == 0:
                continue
            out[ab - 1] += arr[a_i, b_i]
            counts[ab - 1] += 1
    out /= np.where(counts > 0, counts, 1.0)[:, None]
    return out


def reduce_to_diff(f_ab_c, p):
    """f_ab_c: (p-1, p-1). Reduce to 1D function of (log c - log ab) mod (p-1)
    by averaging over (ab, c) pairs with constant Δlog.
    Assumes columns are c=1..p-1 (already restricted to value tokens).
    Returns: array of length p-1, indexed by Δlog ∈ {0, ..., p-2}."""
    _, dlog = discrete_log_table(p)
    n_diff = p - 1
    out = np.zeros(n_diff)
    counts = np.zeros(n_diff)
    for ab in range(1, p):
        j_ab = dlog[ab]
        for c in range(1, p):
            j_c = dlog[c]
            delta = (j_c - j_ab) % n_diff
            out[delta] += f_ab_c[ab - 1, c - 1]
            counts[delta] += 1
    out /= np.where(counts > 0, counts, 1.0)
    return out


def correlate(x, y):
    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()
    x = x - x.mean(); y = y - y.mean()
    return float(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--ks", required=True)
    args = ap.parse_args()
    run_dir = Path(args.run_dir).resolve()
    ck = sorted(run_dir.glob("checkpoint_*.pt"))
    print(f"Loading {ck[-1]}")
    model, ds, _ = load_run(ck[-1])
    model.eval()
    basis, char_idx = build_basis_indexed()
    p = ds.p
    W_E_orig = model.embed.W_E.detach().clone()

    # Full logits.
    logits_full = compute_logits_grid(model, ds)
    print(f"Full logits shape: {tuple(logits_full.shape)}")

    _, dlog = discrete_log_table(p)
    order_idx = np.argsort([dlog[a] for a in range(1, p)])

    ks = [int(x) for x in args.ks.split(",")]
    print(f"\n{'k':>4} {'order':>5}  {'1D-corr w/ ref cos':>20}  "
          f"{'||Δ_k|| total':>14}")
    for k in ks:
        W_E_ab = ablate_W_E(W_E_orig, k, basis, char_idx, p=p)
        logits_ab = compute_logits_grid(model, ds, W_E_override=W_E_ab)
        delta = logits_full - logits_ab               # (p-1, p-1, vocab)

        # Reduce to f(ab, c) by averaging over (a, b) with constant ab.
        f_ab_c = reduce_to_ab(delta, p)               # (p-1, vocab)

        # Restrict to value tokens (c ∈ 1..p-1) for visualization.
        f_vis = f_ab_c[:, 1:p]                        # (p-1, p-1)

        # 1D reduction.
        f_diff = reduce_to_diff(f_vis, p)             # (p-1,) indexed by Δlog

        # Reference: cos[2π k Δlog / (p-1)]
        delta_log = np.arange(p - 1)
        ref_1d = np.cos(2 * np.pi * k * delta_log / (p - 1))
        corr_1d = correlate(f_diff, ref_1d)
        total_norm = float(np.linalg.norm(delta.cpu().numpy()))

        print(f"{k:>4} {order_of(k):>5}  {corr_1d:>+20.3f}  {total_norm:>14.3e}")

        # Plot.
        f_vis_sorted = f_vis[order_idx][:, order_idx]
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        ax = axes[0]
        vmax = max(abs(f_vis.max()), abs(f_vis.min()))
        ax.imshow(f_vis, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                  aspect="equal", origin="upper")
        ax.set_xlabel("c"); ax.set_ylabel("ab")
        ax.set_title(f"Δ_{k}(ab, c) natural index")

        ax = axes[1]
        ax.imshow(f_vis_sorted, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                  aspect="equal", origin="upper")
        ax.set_xlabel("c (dlog-sorted)"); ax.set_ylabel("ab (dlog-sorted)")
        ax.set_title(f"Δ_{k}(ab, c) dlog-sorted — character-product ⇒ anti-diagonal stripes")

        ax = axes[2]
        ax.plot(delta_log, f_diff, "b-", lw=1.6, label=f"empirical Δ_{k}(Δlog)")
        # Scale ref to match.
        scale = (f_diff @ ref_1d) / (ref_1d @ ref_1d + 1e-12)
        ax.plot(delta_log, scale * ref_1d, "r--", lw=1.4,
                label=f"{scale:.2e} · cos[2π·{k}·Δlog/(p-1)]")
        ax.set_xlabel("Δlog = (log c - log ab) mod (p-1)")
        ax.set_ylabel(f"avg Δ_{k}")
        ax.set_title(f"(c) 1D reduction; corr w/ ref = {corr_1d:+.3f}")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

        fig.suptitle(f"Direct ablation contribution Δ_{k}(ab, c) for k={k} "
                     f"(order {order_of(k)})", fontsize=11)
        fig.tight_layout()
        out = run_dir / f"delta_k{k}.png"
        fig.savefig(out, dpi=130, bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    main()
