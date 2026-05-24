"""Quantify the bifurcation: when do K characters start outgrowing non-K?

For each seed, compute mean(K energy) / mean(non-K energy) at every step in
[0, 500). Define commit step as the first step at which the ratio exceeds 1.5.

Produces:
  figures/basis_dynamics/bifurcation_overlay.png       (all seeds, ratio vs step)
  figures/basis_dynamics/bifurcation_summary.png       (histogram of commit steps)
  figures/basis_dynamics/bifurcation_summary.csv
"""
from __future__ import annotations

import math
import pickle
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from crypto_interp.interp.bases import multiplicative_fourier_basis

ROOT = Path(__file__).resolve().parents[1]
IN_DIR = ROOT / "data" / "fine_grained"
TRAJ_FILE = ROOT / "data" / "basis_dynamics" / "trajectories.pkl"
FIG_DIR = ROOT / "figures" / "basis_dynamics"


def build_char_basis():
    p = 113
    basis, names, g = multiplicative_fourier_basis(p)
    char_index = {}
    for i, nm in enumerate(names):
        m = re.match(r"mul (cos|sin) (\d+)", nm)
        if m:
            kk = int(m.group(2))
            char_index.setdefault(kk, []).append(i)
    return basis, char_index


def char_energies_batch(W_E_stack, basis, char_index):
    W_v = W_E_stack[:, :, :113].double()
    coef = torch.einsum("kp,tdp->tkd", basis.double(), W_v)
    E = (coef ** 2).sum(dim=2)
    T = E.shape[0]
    out = np.zeros((T, 56))
    for k, rows in char_index.items():
        out[:, k - 1] = E[:, rows].sum(dim=1).cpu().numpy()
    return out


def main():
    basis, char_index = build_char_basis()
    with open(TRAJ_FILE, "rb") as f:
        trajectories = pickle.load(f)

    fig_overlay, ax = plt.subplots(figsize=(11, 5.5))
    fig_normed, ax_n = plt.subplots(figsize=(11, 5.5))
    commit_steps = {}
    threshold = 1.5

    cmap = plt.cm.viridis
    fg_files = sorted(IN_DIR.glob("seed*_fine_grained.pt"))
    for i, fg_path in enumerate(fg_files):
        seed = int(re.match(r"seed(\d+)", fg_path.name).group(1))
        d = torch.load(fg_path, weights_only=False)
        W_E = d["W_E"]
        epochs = d["epochs"].numpy()
        char_E = char_energies_batch(W_E, basis, char_index)

        # Final K from full trajectory.
        final_ce = trajectories[seed]["char_energy"][-1]
        thresh_e = 0.05 * final_ce.max()
        final_K = [k + 1 for k, e in enumerate(final_ce) if e >= thresh_e]
        non_K = [k for k in range(1, 57) if k not in final_K]

        K_mean = char_E[:, [k - 1 for k in final_K]].mean(axis=1)
        nonK_mean = char_E[:, [k - 1 for k in non_K]].mean(axis=1)
        ratio = K_mean / np.where(nonK_mean > 0, nonK_mean, 1.0)
        # initial-normalized ratio: ratio(t) / ratio(0)
        ratio_norm = ratio / ratio[0]

        above = np.where(ratio_norm > threshold)[0]
        commit = int(epochs[above[0]]) if len(above) else None
        commit_steps[seed] = commit

        color = cmap(i / max(1, len(fg_files) - 1))
        ax.plot(epochs, ratio, color=color, lw=1.0, alpha=0.85,
                label=f"seed {seed} ({commit})")
        ax_n.plot(epochs, ratio_norm, color=color, lw=1.0, alpha=0.85,
                  label=f"seed {seed} ({commit})")
        if commit is not None:
            ax_n.scatter([commit], [ratio_norm[above[0]]], color=color, s=22, zorder=5)

    ax.set_xlabel("step")
    ax.set_ylabel("mean(K energy) / mean(non-K energy)")
    ax.set_title("Bifurcation: K vs non-K energy ratio, all seeds (per-step)")
    ax.set_yscale("log")
    ax.axhline(1.0, color="black", ls="--", lw=0.6, alpha=0.4)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, ncol=2, loc="lower right")
    fig_overlay.tight_layout()
    out1 = FIG_DIR / "bifurcation_overlay.png"
    fig_overlay.savefig(out1, dpi=130, bbox_inches="tight")
    plt.close(fig_overlay)

    ax_n.set_xlabel("step")
    ax_n.set_ylabel("ratio(t) / ratio(0)")
    ax_n.set_title(f"Bifurcation normed to init. Marker = commit step (ratio > {threshold}× init).")
    ax_n.set_yscale("log")
    ax_n.axhline(threshold, color="red", ls="--", lw=0.7, alpha=0.5,
                 label=f"threshold = {threshold}×")
    ax_n.grid(True, alpha=0.3)
    ax_n.legend(fontsize=7, ncol=2, loc="lower right")
    fig_normed.tight_layout()
    out2 = FIG_DIR / "bifurcation_overlay_normed.png"
    fig_normed.savefig(out2, dpi=130, bbox_inches="tight")
    plt.close(fig_normed)

    # Histogram.
    fig, ax = plt.subplots(figsize=(8, 4.5))
    finite = [c for c in commit_steps.values() if c is not None]
    if finite:
        ax.hist(finite, bins=15, color="#2ca02c", alpha=0.8)
    ax.set_xlabel(f"commit step (ratio exceeds {threshold}× init)")
    ax.set_ylabel("seeds")
    ax.set_title(
        f"Distribution of bifurcation step across {len(finite)} / {len(commit_steps)} seeds. "
        f"median = {int(np.median(finite)) if finite else 'n/a'}"
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out3 = FIG_DIR / "bifurcation_summary.png"
    fig.savefig(out3, dpi=130, bbox_inches="tight")
    plt.close(fig)

    print(f"\nCommit steps (threshold = {threshold}× init ratio):")
    for s in sorted(commit_steps):
        print(f"  seed {s:>2}: {commit_steps[s]}")
    if finite:
        print(f"\n  median: {int(np.median(finite))}, mean: {int(np.mean(finite))}, "
              f"min: {min(finite)}, max: {max(finite)}")
    print(f"\nSaved {out1}\n      {out2}\n      {out3}")


if __name__ == "__main__":
    main()
