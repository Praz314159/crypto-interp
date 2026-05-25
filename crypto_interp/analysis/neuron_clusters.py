"""Identify the cluster of MLP neurons specialized to each character k in K,
reconstruct the cluster's contribution to the residual stream as a function of
(a, b), and compare against the algebraic reference cos(θ_k(a)+θ_k(b)).

Four panels per character: cluster signal (natural + dlog-sorted) and reference
(natural + dlog-sorted). Uses crypto_interp.interp (prime-parametric).

Usage:
    python experiments/003_dmodel_sweep_p113/scripts/analyze_neuron_clusters.py \
        --run-dir experiments/003_dmodel_sweep_p113/runs/dmodel_24_dmlp_32_seed1
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from crypto_interp.interp import (
    char_energy,
    char_index,
    correlate,
    discrete_log_table,
    load_run,
    order_of,
)


def per_neuron_dominant_char(W_U, W_out, basis, ci, p):
    """Return (char_E [d_mlp, n_chars], dominant_char [d_mlp]) (1-based ids)."""
    V = W_U[:, :p].double().T @ W_out.double()       # (p, d_mlp)
    coef = basis @ V                                  # (n_basis, d_mlp)
    E = coef ** 2
    nch = max(ci.freqs)
    char_E = np.zeros((W_out.shape[1], nch))
    for k, rs in ci.by_char.items():
        char_E[:, k - 1] = E[rs].sum(dim=0).cpu().numpy()
    return char_E, char_E.argmax(axis=1) + 1


def cluster_signal(model, ds, cluster_neurons, W_U_k):
    """Cluster's projection onto the unembed's character-k direction over the
    full (a, b) grid. Returns (p-1, p-1)."""
    p = ds.p
    aa, bb = torch.meshgrid(torch.arange(1, p), torch.arange(1, p), indexing="ij")
    eq = torch.full_like(aa, ds.eq_token)
    inputs = torch.stack([aa, bb, eq], dim=-1).reshape(-1, 3)
    cache: dict[str, torch.Tensor] = {}
    model.remove_all_hooks()
    model.cache_all(cache)
    with torch.no_grad():
        _ = model(inputs)
    model.remove_all_hooks()
    h = cache["blocks.0.mlp.hook_post"][:, -1, :].double()
    W_out = model.blocks[0].mlp.W_out.detach().double()
    cluster_resid = h[:, cluster_neurons] @ W_out[:, cluster_neurons].T
    sig = cluster_resid @ W_U_k.double()
    return sig.reshape(p - 1, p - 1).cpu().numpy()


def reference_cos_signal(k, p):
    _, dlog = discrete_log_table(p)
    a_dlog = np.array([dlog[a] for a in range(1, p)])
    theta = 2 * np.pi * k * a_dlog / (p - 1)
    return np.cos(theta[:, None] + theta[None, :])


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

    p = ds.p
    basis, ci = char_index(p)
    W_E = model.embed.W_E.detach()[:, :p].double()
    W_U = model.unembed.W_U.detach()
    W_out = model.blocks[0].mlp.W_out.detach()

    char_E_WE = char_energy(W_E, basis, ci)
    K = sorted(k for k in ci.freqs if char_E_WE[k - 1] >= 0.05 * char_E_WE.max())
    print(f"K = {K} (orders {[order_of(k, p) for k in K]}) — primitive root g={ci.g}")

    _, dom = per_neuron_dominant_char(W_U, W_out, basis, ci, p)
    _, dlog = discrete_log_table(p)
    order_idx = np.argsort([dlog[a] for a in range(1, p)])

    for k in K:
        cluster = np.where(dom == k)[0]
        if len(cluster) == 0:
            print(f"  k={k}: no neurons in cluster (skip)")
            continue
        cos_k = basis[ci.cos[k]]
        W_U_k = W_U[:, :p].double() @ cos_k.double()
        W_U_k = W_U_k / (W_U_k.norm() + 1e-12)
        print(f"  k={k} (order {order_of(k, p)}): {len(cluster)} neurons in cluster")

        sig = cluster_signal(model, ds, cluster, W_U_k)
        ref = reference_cos_signal(k, p)
        sig_sorted = sig[order_idx][:, order_idx]
        ref_sorted = ref[order_idx][:, order_idx]

        fig, axes = plt.subplots(2, 2, figsize=(11, 10))
        for ax, data, title in [
            (axes[0, 0], sig, f"cluster signal (k={k}), natural index"),
            (axes[0, 1], sig_sorted, "cluster signal, dlog-sorted"),
            (axes[1, 0], ref, "reference cos(θ_k(a)+θ_k(b)), natural"),
            (axes[1, 1], ref_sorted, "reference, dlog-sorted"),
        ]:
            vmax = max(abs(data.max()), abs(data.min()))
            ax.imshow(data, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="equal", origin="upper")
            ax.set_title(title, fontsize=9)
            ax.set_xlabel("b" + (" (dlog)" if "sorted" in title else ""))
            ax.set_ylabel("a" + (" (dlog)" if "sorted" in title else ""))
        corr = correlate(sig, ref)
        fig.suptitle(f"Neuron cluster for character k={k} (order {order_of(k, p)}), "
                     f"|cluster|={len(cluster)}; correlation w/ reference = {corr:+.3f}",
                     fontsize=11)
        fig.tight_layout()
        out = run_dir / f"{args.out_prefix}_k{k}.png"
        fig.savefig(out, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"    saved {out}  corr={corr:+.3f}")


if __name__ == "__main__":
    main()
