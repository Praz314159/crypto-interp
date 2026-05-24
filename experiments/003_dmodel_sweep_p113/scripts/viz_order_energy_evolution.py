"""Plot the distribution of W_E energy across character ORDERS over the
fine-grained early phase.

For each step in [0, 500), compute the per-character energy of W_E and group
characters by their order (divisors of p-1 = 112). Plot the total energy in
each order class over time, both as absolute values and as fractions.

Outputs:
  figures/basis_dynamics/order_energy_evolution_<tag>.png

Usage:
    python experiments/003_dmodel_sweep_p113/scripts/viz_order_energy_evolution.py \
        --fg-file experiments/003_dmodel_sweep_p113/data/fine_grained/dmlp024_seed01_fine_grained.pt
"""
from __future__ import annotations

import argparse
import math
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from crypto_interp.interp.bases import multiplicative_fourier_basis

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures" / "basis_dynamics"

ORDER_COLOR = {
    2: "#9467bd",   # purple - Legendre
    4: "#8c564b",
    7: "#bcbd22",
    8: "#17becf",
    14: "#e377c2",
    16: "#1f77b4",
    28: "#2ca02c",
    56: "#ff7f0e",
    112: "#d62728",
}
ORDER_LIST = [2, 4, 7, 8, 14, 16, 28, 56, 112]


def build_char_basis():
    basis, names, _ = multiplicative_fourier_basis(113)
    char_idx = {}
    for i, nm in enumerate(names):
        m = re.match(r"mul (cos|sin) (\d+)", nm)
        if m:
            kk = int(m.group(2))
            char_idx.setdefault(kk, []).append(i)
    return basis.double(), char_idx


def order_of(k, n=112):
    return n // math.gcd(k, n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fg-file", required=True)
    ap.add_argument("--tag", default=None,
                    help="optional tag for figure filename")
    args = ap.parse_args()

    fg_path = Path(args.fg_file)
    tag = args.tag or fg_path.stem  # default to filename stem
    print(f"Loading {fg_path}")
    d = torch.load(fg_path, weights_only=False)
    epochs = d["epochs"].numpy()
    W_E_stack = d["W_E"]  # (T, d_model, vocab)
    T = W_E_stack.shape[0]
    print(f"T={T} steps, d_model={W_E_stack.shape[1]}")

    basis, char_idx = build_char_basis()
    # Per-char energy at each step.
    W_v = W_E_stack[:, :, :113].double()
    coef = torch.einsum("kp,tdp->tkd", basis, W_v)
    E = (coef ** 2).sum(dim=2)  # (T, n_basis)
    char_E = np.zeros((T, 56))
    for k, rs in char_idx.items():
        char_E[:, k - 1] = E[:, rs].sum(dim=1).cpu().numpy()

    # Aggregate by order.
    chars_by_order = defaultdict(list)
    for k in range(1, 57):
        chars_by_order[order_of(k)].append(k)
    order_E = np.zeros((T, len(ORDER_LIST)))
    order_count = np.zeros(len(ORDER_LIST), dtype=int)
    for i, o in enumerate(ORDER_LIST):
        idx = [k - 1 for k in chars_by_order.get(o, [])]
        order_count[i] = len(idx)
        if idx:
            order_E[:, i] = char_E[:, idx].sum(axis=1)

    total = order_E.sum(axis=1, keepdims=True)
    order_frac = order_E / np.where(total > 0, total, 1.0)
    # Mean-per-char in each order class (controls for # of chars per order).
    order_E_mean = order_E / np.where(order_count > 0, order_count, 1.0)

    # ---- Plot 1: stacked area (fractional energy by order) ----
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    ax = axes[0]
    stacks = []
    labels = []
    colors = []
    for i, o in enumerate(ORDER_LIST):
        if order_count[i] == 0:
            continue
        stacks.append(order_frac[:, i])
        labels.append(f"order {o} ({order_count[i]} chars)")
        colors.append(ORDER_COLOR[o])
    ax.stackplot(epochs, stacks, labels=labels, colors=colors, alpha=0.85)
    ax.set_xlabel("step")
    ax.set_ylabel("fraction of W_E energy")
    ax.set_title("(a) energy fraction by order")
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)

    # ---- Plot 2: absolute energy per order (sum across chars in that order) ----
    ax = axes[1]
    for i, o in enumerate(ORDER_LIST):
        if order_count[i] == 0:
            continue
        ax.plot(epochs, order_E[:, i], color=ORDER_COLOR[o], lw=1.6,
                label=f"order {o} (×{order_count[i]} chars)")
    ax.set_yscale("log")
    ax.set_xlabel("step")
    ax.set_ylabel("total energy across chars of that order")
    ax.set_title("(b) total energy by order")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)

    # ---- Plot 3: mean energy per character within each order ----
    ax = axes[2]
    for i, o in enumerate(ORDER_LIST):
        if order_count[i] == 0:
            continue
        ax.plot(epochs, order_E_mean[:, i], color=ORDER_COLOR[o], lw=1.6,
                label=f"order {o}")
    ax.set_yscale("log")
    ax.set_xlabel("step")
    ax.set_ylabel("mean energy per character (in that order class)")
    ax.set_title("(c) per-character energy by order")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"Order-class energy evolution ({tag})", fontsize=12)
    fig.tight_layout()
    out = FIG_DIR / f"order_energy_evolution_{tag}.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
