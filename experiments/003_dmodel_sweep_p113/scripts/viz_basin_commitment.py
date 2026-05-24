"""Basin-commitment visualization for a fine-grained early-phase run.

Two views:
  (a) per-character energy trajectories, with the final K highlighted
  (b) K-vs-non-K mean energy over time, with the bifurcation step marked

This is the right plot to answer "how quickly does the model commit to a basis."

Usage:
    python experiments/003_dmodel_sweep_p113/scripts/viz_basin_commitment.py \
        --fg-file experiments/003_dmodel_sweep_p113/data/fine_grained/dmlp024_seed01_fine_grained.pt \
        --run-dir experiments/003_dmodel_sweep_p113/runs/dmodel_24_dmlp_24_seed1 \
        --tag dmlp24_seed1
"""
from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from crypto_interp.interp.bases import multiplicative_fourier_basis

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures" / "basis_dynamics"


def order_of(k, n=112):
    return n // math.gcd(k, n)


ORDER_COLOR = {
    2: "#9467bd", 4: "#8c564b", 7: "#bcbd22", 8: "#17becf",
    14: "#e377c2", 16: "#1f77b4", 28: "#2ca02c", 56: "#ff7f0e", 112: "#d62728",
}


def build_char_basis():
    basis, names, _ = multiplicative_fourier_basis(113)
    char_idx = {}
    for i, nm in enumerate(names):
        m = re.match(r"mul (cos|sin) (\d+)", nm)
        if m:
            kk = int(m.group(2))
            char_idx.setdefault(kk, []).append(i)
    return basis.double(), char_idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fg-file", required=True)
    ap.add_argument("--run-dir", default=None,
                    help="Path to the full-training run dir; used to identify K. "
                         "If omitted, --k must be given explicitly.")
    ap.add_argument("--k", default=None,
                    help="Explicit K as comma-separated ints, e.g. '12,17,18,24'. "
                         "Overrides run-dir-based identification.")
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    basis, char_idx = build_char_basis()
    p = 113

    if args.k is not None:
        K = sorted([int(x) for x in args.k.split(",")])
    else:
        run_dir = Path(args.run_dir)
        ck_list = sorted(run_dir.glob("checkpoint_*.pt"))
        final_ck = ck_list[-1]
        state = torch.load(final_ck, weights_only=False, map_location="cpu")
        W_E_final = state["model_state"]["embed.W_E"][:, :p].double()
        coef = torch.einsum("kp,dp->kd", basis, W_E_final)
        final_E = np.zeros(56)
        for k, rs in char_idx.items():
            final_E[k - 1] = float((coef[rs] ** 2).sum())
        K = sorted([k + 1 for k, e in enumerate(final_E) if e >= 0.05 * final_E.max()])
    nonK = [k for k in range(1, 57) if k not in K]
    print(f"K = {K} (orders {[order_of(k) for k in K]})")

    # Per-step character energies from fine-grained data.
    d = torch.load(args.fg_file, weights_only=False)
    epochs = d["epochs"].numpy()
    W_E_stack = d["W_E"]
    W_v = W_E_stack[:, :, :p].double()
    coef_t = torch.einsum("kp,tdp->tkd", basis, W_v)
    E_t = (coef_t ** 2).sum(dim=2)  # (T, n_basis)
    T = E_t.shape[0]
    char_E = np.zeros((T, 56))
    for k, rs in char_idx.items():
        char_E[:, k - 1] = E_t[:, rs].sum(dim=1).cpu().numpy()

    # K-vs-non-K mean energies.
    K_mean = char_E[:, [k - 1 for k in K]].mean(axis=1)
    nonK_mean = char_E[:, [k - 1 for k in nonK]].mean(axis=1)
    ratio = K_mean / np.where(nonK_mean > 0, nonK_mean, 1.0)
    ratio_norm = ratio / ratio[0]
    above = np.where(ratio_norm > 1.5)[0]
    bif = int(epochs[above[0]]) if len(above) else None

    # Fraction of W_E energy in K.
    total = char_E.sum(axis=1)
    K_total = char_E[:, [k - 1 for k in K]].sum(axis=1)
    K_frac = K_total / np.where(total > 0, total, 1.0)

    # Three-panel figure.
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # (a) Per-character trajectories, K highlighted.
    ax = axes[0]
    for k in range(1, 57):
        o = order_of(k)
        color = ORDER_COLOR.get(o, "#aaaaaa")
        is_K = k in K
        ax.plot(epochs, char_E[:, k - 1],
                color=color, alpha=0.95 if is_K else 0.10,
                lw=1.5 if is_K else 0.4,
                label=f"k={k} (o={o})" if is_K else None)
    if bif is not None:
        ax.axvline(bif, color="black", ls="--", lw=0.8, alpha=0.7,
                   label=f"bifurcation step ({bif})")
    ax.set_yscale("log")
    ax.set_xlabel("step")
    ax.set_ylabel("W_E energy in character k")
    ax.set_title("(a) per-character energy (K highlighted)")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)

    # (b) K mean vs non-K mean.
    ax = axes[1]
    ax.plot(epochs, K_mean, color="#d62728", lw=1.8, label=f"mean K energy ({len(K)} chars)")
    ax.plot(epochs, nonK_mean, color="#1f77b4", lw=1.8, label=f"mean non-K energy ({len(nonK)} chars)")
    if bif is not None:
        ax.axvline(bif, color="black", ls="--", lw=0.8, alpha=0.7,
                   label=f"bif = {bif}")
    ax.set_yscale("log")
    ax.set_xlabel("step")
    ax.set_ylabel("mean energy per character")
    ax.set_title("(b) K mean vs non-K mean")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3)

    # (c) K/non-K ratio (normalized to init) — the bifurcation signal.
    ax = axes[2]
    ax.plot(epochs, ratio_norm, color="#2ca02c", lw=1.8,
            label=f"K/non-K ratio (norm to t=0)")
    ax.axhline(1.5, color="red", ls="--", lw=0.8, alpha=0.6,
               label="bifurcation threshold (1.5×)")
    if bif is not None:
        ax.axvline(bif, color="black", ls="--", lw=0.8, alpha=0.7,
                   label=f"bif = {bif}")
    ax.set_yscale("log")
    ax.set_xlabel("step")
    ax.set_ylabel("ratio(t) / ratio(0)")
    ax.set_title(f"(c) basin commitment signal — bif step = {bif}")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"Basin commitment ({args.tag}) — K = {K}", fontsize=12)
    fig.tight_layout()
    out = FIG_DIR / f"basin_commitment_{args.tag}.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
