"""For each character k, compute how concentrated its MLP contribution is
across neurons. This is the dual of the neuron-specialization analysis:
  - "Are neurons specialized?" — participation ratio over k for a fixed neuron.
  - "Are characters concentrated on a few neurons?" — participation ratio over
    neurons for a fixed character k.

Compute, for each k:
  P_neur(k) = (Σ_j |v_j[k]|^2)^2 / Σ_j |v_j[k]|^4
where v_j is neuron j's output spectrum at character k. Small → few neurons
carry character k. Large → contribution spread across many.

Then compare against the ablation impact (Δlog10 test loss) from
analyze_ablation_full.py to see whether piggybacking characters (essential but
no dedicated cluster) are concentrated or distributed.

Usage:
    python experiments/003_dmodel_sweep_p113/scripts/analyze_char_concentration.py \
        --run-dir experiments/003_dmodel_sweep_p113/runs/dmodel_24_dmlp_32_seed1 \
        --ablation-csv experiments/003_dmodel_sweep_p113/data/basis_dynamics/ablation_full_dmlp32_seed1.csv \
        --tag dmlp32_seed1
"""
from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from crypto_interp.interp.bases import multiplicative_fourier_basis

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures" / "basis_dynamics"

ORDER_COLOR = {
    2: "#9467bd", 4: "#8c564b", 7: "#bcbd22", 8: "#17becf",
    14: "#e377c2", 16: "#1f77b4", 28: "#2ca02c", 56: "#ff7f0e", 112: "#d62728",
}


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


def participation(arr, axis=None):
    """(sum²) / sum_of_squares — effective count of dominant entries."""
    s = arr.sum(axis=axis)
    s2 = (arr ** 2).sum(axis=axis)
    return (s ** 2) / np.where(s2 > 1e-30, s2, 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--ablation-csv", required=True)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    run_dir = Path(args.run_dir).resolve()
    ck = sorted(run_dir.glob("checkpoint_*.pt"))
    state = torch.load(ck[-1], weights_only=False, map_location="cpu")
    W_U = state["model_state"]["unembed.W_U"]
    W_out = state["model_state"]["blocks.0.mlp.W_out"]
    d_mlp = W_out.shape[1]
    basis, char_idx = build_basis_indexed()
    p = 113

    # Per-neuron, per-character output energy.
    V = W_U[:, :p].double().T @ W_out.double()        # (p, d_mlp)
    coef = basis @ V                                   # (n_basis, d_mlp)
    E = (coef ** 2).cpu().numpy()                      # (n_basis, d_mlp)
    char_neur = np.zeros((56, d_mlp))
    for k, rs in char_idx.items():
        char_neur[k - 1] = E[rs].sum(axis=0)

    # Per-character participation across neurons.
    P = np.array([participation(char_neur[k - 1]) for k in range(1, 57)])  # length 56
    # Per-neuron participation across characters.
    Pn = np.array([participation(char_neur[:, j]) for j in range(d_mlp)])

    # Read ablation table.
    abl = {}
    with open(args.ablation_csv) as f:
        rd = csv.DictReader(f)
        for row in rd:
            abl[int(row["k"])] = dict(
                delta_log=float(row["delta_log"]),
                energy=float(row["energy"]),
                in_top_K=row["in_top_K"] == "Y",
            )

    # Print.
    print(f"d_mlp = {d_mlp}")
    print(f"\nPer-neuron participation across characters (specialization view):")
    print(f"  min/median/mean/max:  "
          f"{Pn.min():.2f} / {np.median(Pn):.2f} / {Pn.mean():.2f} / {Pn.max():.2f}")
    print(f"  (1 ≈ pure specialist, larger ≈ more characters per neuron)")

    print(f"\nPer-character concentration (effective # neurons carrying char k):")
    print(f"{'k':>4} {'o':>4} {'P_neur':>7} {'energy_WE':>10} "
          f"{'Δlog10':>8} {'topK':>5}")
    by_impact = sorted(range(56), key=lambda i: -abl[i + 1]["delta_log"])
    for i in by_impact[:15]:
        k = i + 1
        print(f"{k:>4} {order_of(k):>4} {P[i]:>7.2f} {abl[k]['energy']:>10.3f} "
              f"{abl[k]['delta_log']:>+8.3f} "
              f"{'Y' if abl[k]['in_top_K'] else '.':>5}")

    # Plot.
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

    # (a) participation per character vs ablation impact.
    ax = axes[0]
    for k in range(1, 57):
        c = ORDER_COLOR.get(order_of(k), "#aaaaaa")
        marker = "*" if abl[k]["in_top_K"] else "o"
        ms = 14 if abl[k]["in_top_K"] else 7
        ax.scatter(P[k - 1], abl[k]["delta_log"], color=c,
                   marker=marker, s=ms ** 2, alpha=0.85,
                   edgecolor="black", linewidth=0.5)
        if abl[k]["delta_log"] > 1.5:
            ax.annotate(f"k={k}", (P[k - 1], abl[k]["delta_log"]),
                        xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.axvline(1, color="gray", ls=":", lw=0.6, alpha=0.5)
    ax.axhline(0.5, color="red", ls="--", lw=0.6, alpha=0.5,
               label="ablation Δlog10 = 0.5")
    ax.set_xlabel("effective # neurons carrying character k  (participation)")
    ax.set_ylabel("Δlog10(test loss) after ablating k")
    ax.set_title(f"(a) per-character concentration vs essentialness")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (b) per-neuron participation histogram (specialization).
    ax = axes[1]
    ax.hist(Pn, bins=20, color="#1f77b4", alpha=0.85, edgecolor="black")
    ax.axvline(np.median(Pn), color="red", ls="--", lw=1.0,
               label=f"median = {np.median(Pn):.2f}")
    ax.set_xlabel("participation ratio across characters")
    ax.set_ylabel("# neurons")
    ax.set_title(f"(b) per-neuron specialization (across {d_mlp} neurons)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"Character concentration vs neuron specialization ({args.tag})",
                 fontsize=11)
    fig.tight_layout()
    out = FIG_DIR / f"char_concentration_{args.tag}.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
