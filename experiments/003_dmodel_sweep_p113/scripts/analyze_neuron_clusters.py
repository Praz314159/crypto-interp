"""For a chosen model, identify the cluster of MLP neurons specialized to each
character k ∈ K, then reconstruct the cluster's contribution to the residual
stream as a function of the input pair (a, b). Compare against the algebraic
reference χ_k(a)·χ_k(b) = χ_k(ab) (real part = cos(θ_k(a)+θ_k(b))).

For each character k in K, produces four panels:
  (top-left)  reconstructed cluster signal, natural integer indexing
  (top-right) same signal, rows/cols sorted by discrete log (CRT-aligned)
  (bot-left)  reference cos(θ_k(a)+θ_k(b)), natural integer indexing
  (bot-right) reference, dlog-sorted

In the dlog-sorted view, the signal is a function of (log_g a + log_g b) mod
(p-1), so it should look like diagonal stripes of period order(k).

Usage:
    python experiments/003_dmodel_sweep_p113/scripts/analyze_neuron_clusters.py \
        --run-dir experiments/003_dmodel_sweep_p113/runs/dmodel_24_dmlp_32_seed1
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

ROOT = Path(__file__).resolve().parents[1]


def build_char_basis():
    basis, names, g = multiplicative_fourier_basis(113)
    char_idx = {}
    for i, nm in enumerate(names):
        m = re.match(r"mul (cos|sin) (\d+)", nm)
        if m:
            kk = int(m.group(2))
            char_idx.setdefault(kk, []).append(i)
    return basis.double(), names, char_idx, g


def order_of(k, n=112):
    return n // math.gcd(k, n)


def per_neuron_dominant_char(W_U, W_out, basis, char_idx):
    """Return (char_E [d_mlp, 56], dominant_char [d_mlp])."""
    p = 113
    V = W_U[:, :p].double().T @ W_out.double()   # (p, d_mlp)
    coef = basis @ V                              # (n_basis, d_mlp)
    E = (coef ** 2)
    d_mlp = W_out.shape[1]
    char_E = np.zeros((d_mlp, 56))
    for k, rs in char_idx.items():
        char_E[:, k - 1] = E[rs].sum(dim=0).cpu().numpy()
    dom = char_E.argmax(axis=1) + 1  # 1-based char index
    return char_E, dom


def cluster_signal(model, ds, cluster_neurons, W_U_k):
    """For neurons in cluster_neurons, compute the cluster's projection onto
    the unembed's character-k direction over the full (a,b) grid.

    Returns: (p-1) x (p-1) array indexed by (a, b) ∈ {1, ..., p-1}².
    """
    p = ds.p
    # Build all (a, b, =) input triples for a, b ∈ {1, ..., p-1}.
    a_grid = torch.arange(1, p)
    b_grid = torch.arange(1, p)
    aa, bb = torch.meshgrid(a_grid, b_grid, indexing="ij")
    eq = torch.full_like(aa, ds.eq_token)
    inputs = torch.stack([aa, bb, eq], dim=-1).reshape(-1, 3)
    # Forward pass with hooks to get MLP post-activations.
    cache: dict[str, torch.Tensor] = {}
    model.remove_all_hooks()
    model.cache_all(cache)
    with torch.no_grad():
        _ = model(inputs)
    model.remove_all_hooks()
    # post-MLP activations at final position: (batch, seq, d_mlp).
    h = cache["blocks.0.mlp.hook_post"][:, -1, :].double()  # (N, d_mlp)
    # cluster sum: (N, d_model) = sum_{j in cluster} W_out[:, j] * h[:, j]
    W_out = model.blocks[0].mlp.W_out.detach().double()
    cluster_W_out = W_out[:, cluster_neurons]
    cluster_acts = h[:, cluster_neurons]
    cluster_resid = cluster_acts @ cluster_W_out.T  # (N, d_model)
    # Project onto W_U_k direction (d_model -> scalar via dot product).
    sig = cluster_resid @ W_U_k.double()   # (N,)
    return sig.reshape(p - 1, p - 1).cpu().numpy()


def reference_cos_signal(k, p=113):
    """The mathematical reference: cos(θ_k(a) + θ_k(b)) for a,b ∈ {1,...,p-1}.
    Returns (p-1, p-1) array."""
    _, dlog = discrete_log_table(p)
    a_dlog = np.array([dlog[a] for a in range(1, p)])
    b_dlog = np.array([dlog[b] for b in range(1, p)])
    theta_a = 2 * np.pi * k * a_dlog / (p - 1)
    theta_b = 2 * np.pi * k * b_dlog / (p - 1)
    return np.cos(theta_a[:, None] + theta_b[None, :])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=str, required=True)
    ap.add_argument("--out-prefix", type=str, default="neuron_clusters")
    args = ap.parse_args()
    run_dir = Path(args.run_dir).resolve()
    ck = sorted(run_dir.glob("checkpoint_*.pt"))
    if not ck:
        raise SystemExit(f"No checkpoint in {run_dir}")
    print(f"Loading {ck[-1]}")
    model, ds, _ = load_run(ck[-1])
    model.eval()

    basis, names, char_idx, g = build_char_basis()
    p = ds.p
    W_E = model.embed.W_E.detach()[:, :p].double()
    W_U = model.unembed.W_U.detach()
    W_out = model.blocks[0].mlp.W_out.detach()

    # K from W_E.
    coef = torch.einsum("kp,dp->kd", basis, W_E)
    char_E_WE = np.zeros(56)
    for k, rs in char_idx.items():
        char_E_WE[k - 1] = float((coef[rs] ** 2).sum())
    K = sorted([k + 1 for k, e in enumerate(char_E_WE) if e >= 0.05 * char_E_WE.max()])
    print(f"K = {K} (orders {[order_of(k) for k in K]}) — primitive root g={g}")

    # Per-neuron dominant character.
    char_E_neur, dom = per_neuron_dominant_char(W_U, W_out, basis, char_idx)

    fig_dir = run_dir
    for k in K:
        cluster = np.where(dom == k)[0]
        n_cluster = len(cluster)
        # W_U_k: the unembed's character-k read direction in d_model.
        # = sum of W_U[:, c] * χ_k(c) for c ∈ Z/p* (via the basis row for cos_k).
        # Build the cos_k basis vector then project.
        # We just use the basis row directly: basis[rows, c] gives χ_k cos/sin
        # over c. The read direction is W_U @ (basis[rows].T pooled).
        cos_k_idx, sin_k_idx = None, None
        for i, nm in enumerate(names):
            m = re.match(r"mul (cos|sin) (\d+)", nm)
            if m and int(m.group(2)) == k:
                if m.group(1) == "cos":
                    cos_k_idx = i
                else:
                    sin_k_idx = i
        # Use the cos_k basis vector (real part of χ_k); for k=56 there's no sin.
        cos_k = basis[cos_k_idx]                          # (p,)
        # Read direction in d_model: W_U @ cos_k.
        W_U_k = (W_U[:, :p].double() @ cos_k.double())    # (d_model,)
        W_U_k = W_U_k / (W_U_k.norm() + 1e-12)

        if n_cluster == 0:
            print(f"  k={k}: no neurons in cluster (skip)")
            continue
        print(f"  k={k} (order {order_of(k)}): {n_cluster} neurons in cluster")
        sig = cluster_signal(model, ds, cluster, W_U_k)
        ref = reference_cos_signal(k, p)
        # Discrete-log sort: order rows/cols by log_g.
        _, dlog = discrete_log_table(p)
        order_idx = np.argsort([dlog[a] for a in range(1, p)])
        sig_sorted = sig[order_idx][:, order_idx]
        ref_sorted = ref[order_idx][:, order_idx]
        # Plot.
        fig, axes = plt.subplots(2, 2, figsize=(11, 10))
        for ax, data, title in [
            (axes[0, 0], sig, f"cluster signal (k={k}), natural index"),
            (axes[0, 1], sig_sorted, f"cluster signal, dlog-sorted"),
            (axes[1, 0], ref, f"reference cos(θ_k(a)+θ_k(b)), natural"),
            (axes[1, 1], ref_sorted, "reference, dlog-sorted"),
        ]:
            vmax = max(abs(data.max()), abs(data.min()))
            ax.imshow(data, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                      aspect="equal", origin="upper")
            ax.set_title(title, fontsize=9)
            ax.set_xlabel("b" + (" (dlog)" if "sorted" in title else ""))
            ax.set_ylabel("a" + (" (dlog)" if "sorted" in title else ""))
        # Check alignment quality.
        # Normalize both, compute correlation.
        a = sig.flatten(); a = a - a.mean()
        b = ref.flatten(); b = b - b.mean()
        corr = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
        fig.suptitle(f"Neuron cluster for character k={k} (order {order_of(k)}), "
                     f"|cluster|={n_cluster}; correlation w/ reference = {corr:+.3f}",
                     fontsize=11)
        fig.tight_layout()
        out = fig_dir / f"{args.out_prefix}_k{k}.png"
        fig.savefig(out, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"    saved {out}  corr={corr:+.3f}")


if __name__ == "__main__":
    main()
