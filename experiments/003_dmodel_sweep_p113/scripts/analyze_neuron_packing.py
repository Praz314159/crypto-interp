"""Per-neuron output-spectrum analysis — Analysis (A) from the packing discussion.

For each MLP neuron j in each completed run, compute its direct contribution to
the c-axis logits:
    v_j = W_U^T @ W_out[:, j]   in R^p
Then project v_j onto the multiplicative-Fourier basis on (Z/p)^*, and compute
the participation ratio (effective number of characters this neuron contributes
to).

Outputs:
  data/basis_dynamics/neuron_packing_table.csv
  figures/basis_dynamics/neuron_packing_distributions.png
  figures/basis_dynamics/neuron_packing_summary.png
"""
from __future__ import annotations

import csv
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from crypto_interp.interp.bases import multiplicative_fourier_basis

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
FIG_DIR = ROOT / "figures" / "basis_dynamics"
OUT_DIR = ROOT / "data" / "basis_dynamics"


def order_of(k, n=112):
    return n // math.gcd(k, n)


def build_char_basis():
    basis, names, _ = multiplicative_fourier_basis(113)
    char_idx = {}
    for i, nm in enumerate(names):
        m = re.match(r"mul (cos|sin) (\d+)", nm)
        if m:
            kk = int(m.group(2))
            char_idx.setdefault(kk, []).append(i)
    return basis.double(), char_idx


def per_neuron_char_energy(W_U, W_out, basis, char_idx):
    """W_U shape: (d_model, d_vocab); W_out shape: (d_model, d_mlp).
    Returns char_E shape (d_mlp, 56): per-neuron per-character energy."""
    p = 113
    W_U_v = W_U[:, :p].double()             # (d_model, p)
    W_out = W_out.double()                  # (d_model, d_mlp)
    # v_j[a] = sum_d W_U[d, a] * W_out[d, j]
    V = W_U_v.T @ W_out                     # (p, d_mlp)
    # Project: coef[k, j] = sum_a basis[k, a] V[a, j]
    coef = basis @ V                        # (n_basis, d_mlp)
    E = (coef ** 2)                         # (n_basis, d_mlp)
    d_mlp = W_out.shape[1]
    out = np.zeros((d_mlp, 56))
    for k, rows in char_idx.items():
        out[:, k - 1] = E[rows].sum(dim=0).cpu().numpy()
    return out


def participation_ratio(row, eps=1e-12):
    s = row.sum()
    if s < eps:
        return 0.0
    s2 = (row ** 2).sum()
    return float(s ** 2 / s2)


def list_runs():
    """Discover (d_mlp, seed, run_dir, final_ckpt) for all completed runs."""
    out = []
    # d_mlp sweep runs.
    pat_a = re.compile(r"dmodel_24_dmlp_(\d+)_seed(\d+)$")
    for d in sorted(RUNS.iterdir()):
        m = pat_a.match(d.name)
        if m and d.is_dir():
            d_mlp = int(m.group(1))
            seed = int(m.group(2))
            ck = sorted(d.glob("checkpoint_*.pt"))
            if ck and (d / "losses.pt").exists():  # only completed
                out.append((d_mlp, seed, d, ck[-1]))
    # Baseline 512 runs.
    pat_b = re.compile(r"dmodel_24_seed(\d+)$")
    for d in sorted(RUNS.iterdir()):
        m = pat_b.match(d.name)
        if m and d.is_dir():
            seed = int(m.group(1))
            ck = sorted(d.glob("checkpoint_*.pt"))
            if ck and (d / "losses.pt").exists():
                out.append((512, seed, d, ck[-1]))
    return sorted(out)


def main():
    basis, char_idx = build_char_basis()
    runs = list_runs()
    print(f"Analyzing {len(runs)} completed runs")

    rows = []
    per_neuron_eff = {}  # (d_mlp, seed) -> array of participation ratios
    for d_mlp, seed, run_dir, ck_path in runs:
        state = torch.load(ck_path, weights_only=False, map_location="cpu")
        W_U = state["model_state"]["unembed.W_U"]   # (d_model, d_vocab)
        W_out = state["model_state"]["blocks.0.mlp.W_out"]  # (d_model, d_mlp)
        if W_out.shape[1] != d_mlp:
            print(f"  SKIP: shape mismatch d_mlp tag={d_mlp}, W_out.shape={W_out.shape}")
            continue
        char_E = per_neuron_char_energy(W_U, W_out, basis, char_idx)  # (d_mlp, 56)
        # Participation ratio per neuron.
        eff = np.array([participation_ratio(char_E[j]) for j in range(d_mlp)])
        # Total energy per neuron.
        E_total = char_E.sum(axis=1)
        # Filter to "active" neurons (energy at least 1% of max).
        thresh = 0.01 * E_total.max()
        active = E_total >= thresh
        n_active = int(active.sum())
        eff_active = eff[active]
        per_neuron_eff[(d_mlp, seed)] = eff_active

        # Final K identification (same as analyze_dmlp_sweep).
        W_E = state["model_state"]["embed.W_E"][:, :113].double()
        coef = torch.einsum("kp,dp->kd", basis, W_E)
        char_E_WE = np.zeros(56)
        for k, rs in char_idx.items():
            char_E_WE[k - 1] = float((coef[rs] ** 2).sum())
        K = sorted([int(k + 1) for k, e in enumerate(char_E_WE)
                    if e >= 0.05 * char_E_WE.max()])

        rows.append(dict(
            d_mlp=d_mlp, seed=seed, K=K, K_size=len(K),
            n_active_neurons=n_active,
            median_eff=float(np.median(eff_active)) if n_active else 0.0,
            mean_eff=float(np.mean(eff_active)) if n_active else 0.0,
            max_eff=float(np.max(eff_active)) if n_active else 0.0,
            min_eff=float(np.min(eff_active)) if n_active else 0.0,
        ))
        print(f"  d_mlp={d_mlp:>3} seed={seed:>2}: |K|={len(K)}  active={n_active:>3}/{d_mlp}  "
              f"med_eff={np.median(eff_active):>5.2f}  "
              f"mean_eff={np.mean(eff_active):>5.2f}")

    # CSV.
    csv_path = OUT_DIR / "neuron_packing_table.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            r["K"] = "|".join(map(str, r["K"]))
            w.writerow(r)
    print(f"\nSaved {csv_path}")

    # --- Plot 1: per-(d_mlp,seed) histograms of participation ratio ---
    dmlps = sorted(set(d for d, _ in per_neuron_eff))
    fig, axes = plt.subplots(1, len(dmlps), figsize=(4 * len(dmlps), 4.5),
                              sharey=True)
    if len(dmlps) == 1:
        axes = [axes]
    bins = np.linspace(0, 8, 41)
    for ax, d in zip(axes, dmlps):
        seeds_here = sorted(s for (dm, s) in per_neuron_eff if dm == d)
        for s in seeds_here:
            arr = per_neuron_eff[(d, s)]
            ax.hist(arr, bins=bins, alpha=0.4,
                    label=f"seed {s} (n={len(arr)}, med={np.median(arr):.2f})")
        ax.set_xlabel("participation ratio (≈ # chars/neuron)")
        ax.set_title(f"d_mlp = {d}")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("# neurons")
    fig.suptitle("Per-neuron character participation ratio across d_mlp budgets",
                 fontsize=11)
    fig.tight_layout()
    out1 = FIG_DIR / "neuron_packing_distributions.png"
    fig.savefig(out1, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out1}")

    # --- Plot 2: summary — median eff vs d_mlp ---
    fig, ax = plt.subplots(figsize=(7, 5))
    xs, ys, colors = [], [], []
    cmap = {64: "#d62728", 128: "#ff7f0e", 256: "#9467bd", 512: "#1f77b4"}
    for (d, s), arr in per_neuron_eff.items():
        ax.scatter([d], [np.median(arr)], s=80, color=cmap.get(d, "#7f7f7f"),
                   alpha=0.85, edgecolor="black", linewidth=0.4)
        ax.annotate(f"s{s}", (d, np.median(arr)),
                    xytext=(5, 5), textcoords="offset points", fontsize=7)
    # Connect medians by d_mlp budget
    by_d = {}
    for (d, _), arr in per_neuron_eff.items():
        by_d.setdefault(d, []).append(np.median(arr))
    xs_line = sorted(by_d)
    ys_line = [np.median(by_d[d]) for d in xs_line]
    ax.plot(xs_line, ys_line, "k--", lw=0.8, alpha=0.6, label="median over seeds")
    ax.set_xscale("log", base=2)
    ax.set_xticks(xs_line)
    ax.set_xticklabels([str(d) for d in xs_line])
    ax.set_xlabel("d_mlp budget")
    ax.set_ylabel("median participation ratio across active neurons")
    ax.set_title(f"Per-neuron packing across d_mlp ({len(per_neuron_eff)} runs)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out2 = FIG_DIR / "neuron_packing_summary.png"
    fig.savefig(out2, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out2}")


if __name__ == "__main__":
    main()
