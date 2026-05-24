"""For a chosen character k, compute the FULL summed MLP contribution to the
residual-stream direction that the unembed reads as character k. This is what
the model "sees" for that character — regardless of which individual neurons
dominantly specialize on it.

For piggybacking characters (essential but no dedicated cluster), this should
reveal that the summed signal is still a structured χ_k(a)χ_k(b) function,
even though no single neuron carries it.

For each requested k, plot:
  (top-left)  full summed signal, natural-integer indexing of (a, b)
  (top-right) full summed signal, rows/cols sorted by discrete log
  (bot-left)  algebraic reference cos(θ_k(a) + θ_k(b)), natural indexing
  (bot-right) reference, dlog-sorted

Usage:
    python experiments/003_dmodel_sweep_p113/scripts/analyze_full_signal.py \
        --run-dir experiments/003_dmodel_sweep_p113/runs/dmodel_24_dmlp_32_seed1 \
        --ks 3,10,20,51,6,30,39
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


def build_basis_indexed():
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


def full_signal(model, ds, k, basis, names, char_idx):
    """Compute the full summed MLP contribution at each (a, b) projected onto
    the unembed's character-k cos direction (and sin direction).

    Returns: real_part (p-1, p-1), imag_part (p-1, p-1)  where indexing is
    a, b ∈ {1, ..., p-1}.
    """
    p = ds.p
    # build (a, b, =) input triples
    a_grid = torch.arange(1, p)
    b_grid = torch.arange(1, p)
    aa, bb = torch.meshgrid(a_grid, b_grid, indexing="ij")
    eq = torch.full_like(aa, ds.eq_token)
    inputs = torch.stack([aa, bb, eq], dim=-1).reshape(-1, 3)

    cache: dict[str, torch.Tensor] = {}
    model.remove_all_hooks()
    model.cache_all(cache)
    with torch.no_grad():
        _ = model(inputs)
    model.remove_all_hooks()

    # post-ReLU activations at final position: (N, d_mlp)
    h = cache["blocks.0.mlp.hook_post"][:, -1, :].double()
    W_out = model.blocks[0].mlp.W_out.detach().double()             # (d_model, d_mlp)
    W_U = model.unembed.W_U.detach().double()[:, :p]                # (d_model, p)

    # Find cos_k and sin_k basis rows
    cos_idx = sin_idx = None
    for i, nm in enumerate(names):
        m = re.match(r"mul (cos|sin) (\d+)", nm)
        if m and int(m.group(2)) == k:
            if m.group(1) == "cos":
                cos_idx = i
            else:
                sin_idx = i
    cos_k = basis[cos_idx].double()                                 # (p,)
    # cosine read direction in d_model
    W_U_k_cos = W_U @ cos_k
    W_U_k_cos = W_U_k_cos / (W_U_k_cos.norm() + 1e-12)
    parts = {}
    # full residual contribution per (a, b): sum_j h[:,j] * W_out[:,j]
    cluster_resid = h @ W_out.T                                     # (N, d_model)
    sig_cos = (cluster_resid @ W_U_k_cos).reshape(p - 1, p - 1).cpu().numpy()
    parts["cos"] = sig_cos
    if sin_idx is not None:
        sin_k = basis[sin_idx].double()
        W_U_k_sin = W_U @ sin_k
        W_U_k_sin = W_U_k_sin / (W_U_k_sin.norm() + 1e-12)
        sig_sin = (cluster_resid @ W_U_k_sin).reshape(p - 1, p - 1).cpu().numpy()
        parts["sin"] = sig_sin
    return parts


def reference_cos_signal(k, p=113):
    _, dlog = discrete_log_table(p)
    j_a = np.array([dlog[a] for a in range(1, p)])
    theta_a = 2 * np.pi * k * j_a / (p - 1)
    return np.cos(theta_a[:, None] + theta_a[None, :])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--ks", type=str, required=True,
                    help="comma-separated character indices, e.g. '20,6,30'")
    args = ap.parse_args()
    run_dir = Path(args.run_dir).resolve()
    ck = sorted(run_dir.glob("checkpoint_*.pt"))
    print(f"Loading {ck[-1]}")
    model, ds, _ = load_run(ck[-1])
    model.eval()
    basis, names, char_idx, g = build_basis_indexed()
    print(f"primitive root g = {g}")

    p = ds.p
    _, dlog = discrete_log_table(p)
    order_idx = np.argsort([dlog[a] for a in range(1, p)])

    for k_str in args.ks.split(","):
        k = int(k_str)
        parts = full_signal(model, ds, k, basis, names, char_idx)
        sig = parts["cos"]
        ref = reference_cos_signal(k, p)
        sig_sorted = sig[order_idx][:, order_idx]
        ref_sorted = ref[order_idx][:, order_idx]
        # correlation with reference (mean-subtracted)
        a = sig.flatten(); a = a - a.mean()
        b = ref.flatten(); b = b - b.mean()
        corr = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

        fig, axes = plt.subplots(2, 2, figsize=(11, 10))
        for ax, data, title in [
            (axes[0, 0], sig, f"full MLP cos-{k} signal, natural index"),
            (axes[0, 1], sig_sorted, "same, dlog-sorted"),
            (axes[1, 0], ref, f"reference cos[θ_{k}(a)+θ_{k}(b)], natural"),
            (axes[1, 1], ref_sorted, "reference, dlog-sorted"),
        ]:
            vmax = max(abs(data.max()), abs(data.min()))
            ax.imshow(data, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                      aspect="equal", origin="upper")
            ax.set_title(title, fontsize=9)
            ax.set_xlabel("b" + (" (dlog)" if "sorted" in title else ""))
            ax.set_ylabel("a" + (" (dlog)" if "sorted" in title else ""))
        fig.suptitle(
            f"Full MLP→logit signal for character k={k} "
            f"(order {order_of(k)}); correlation w/ reference = {corr:+.3f}",
            fontsize=11,
        )
        fig.tight_layout()
        out = run_dir / f"full_signal_k{k}.png"
        fig.savefig(out, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"  k={k} (o={order_of(k)}): corr = {corr:+.3f}  saved {out.name}")


if __name__ == "__main__":
    main()
