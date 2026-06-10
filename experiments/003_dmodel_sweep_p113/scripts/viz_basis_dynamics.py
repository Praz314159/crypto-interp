"""Visualize and summarize the basis-dynamics analysis.

Loads:
  data/basis_dynamics/trajectories.pkl   (G1 + G2 per-seed time series)
  data/basis_dynamics/b4_init_gradient.pkl (B4 init-gradient measurement)

Produces:
  figures/basis_dynamics/per_character_<seed>.png    (G1: 56 char trajectories)
  figures/basis_dynamics/eigvals_<seed>.png          (G2: top eigvals over time)
  figures/basis_dynamics/eig_identity_<seed>.png     (G2: char composition of eig 1 over time)
  figures/basis_dynamics/b4_predictions.png          (B4: predicted vs observed K)
  figures/basis_dynamics/b4_summary.csv              (B4 quantitative)
  figures/basis_dynamics/joint_summary.png           (overlay: commit-time vs cliff)

Also prints a summary table.
"""
from __future__ import annotations

import math
import pickle
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
IN_DIR = ROOT / "data" / "basis_dynamics"
FIG_DIR = ROOT / "figures" / "basis_dynamics"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def order_of(k: int, n: int = 112) -> int:
    return n // math.gcd(k, n)


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


def identify_observed_K(char_E_final: np.ndarray, frac: float = 0.05) -> list[int]:
    """Characters with energy >= frac * max(energy) at the final checkpoint."""
    thresh = frac * char_E_final.max()
    return sorted([int(k + 1) for k, e in enumerate(char_E_final) if e >= thresh])


def cliff_time(trajectories_seed) -> int | None:
    """Heuristic: the first epoch at which the top-K energy concentration exceeds
    50% of the total. Returns the epoch (int) or None if not reached."""
    epochs = trajectories_seed["epochs"]
    ce = trajectories_seed["char_energy"]
    final = ce[-1]
    K = identify_observed_K(final)
    total = ce.sum(axis=1)
    inK = ce[:, [k - 1 for k in K]].sum(axis=1)
    frac = inK / np.where(total > 0, total, 1.0)
    above = np.where(frac > 0.5)[0]
    return int(epochs[above[0]]) if len(above) else None


def commit_time(trajectories_seed, eigvec_top_k_chars=5) -> int | None:
    """The earliest epoch after which the top-K-character set is stable (= terminal set)
    for all subsequent checkpoints. Uses top-5 by character energy."""
    epochs = trajectories_seed["epochs"]
    ce = trajectories_seed["char_energy"]
    terminal_top = set(np.argsort(ce[-1])[-eigvec_top_k_chars:])
    # Walk backwards, find first checkpoint where top-5 stays terminal forever after.
    for i in range(len(epochs) - 1, -1, -1):
        topi = set(np.argsort(ce[i])[-eigvec_top_k_chars:])
        if topi != terminal_top:
            return int(epochs[i + 1]) if i + 1 < len(epochs) else None
    return int(epochs[0])


# ---------- per-seed plots ----------

def plot_G1_per_character(seed, traj):
    epochs = traj["epochs"]
    ce = traj["char_energy"]  # (T, 56)
    K = identify_observed_K(ce[-1])
    fig, ax = plt.subplots(figsize=(11, 6))
    for k in range(1, 57):
        o = order_of(k)
        color = ORDER_COLOR.get(o, "#aaaaaa")
        is_K = k in K
        ax.plot(
            epochs, ce[:, k - 1],
            color=color, alpha=0.95 if is_K else 0.18,
            lw=1.5 if is_K else 0.8,
            label=(f"k={k} (o={o})" if is_K else None),
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("energy in character k")
    ct = commit_time(traj)
    cl = cliff_time(traj)
    title = f"seed {seed}: per-character W_E energy trajectory. K={K}"
    if ct is not None:
        ax.axvline(ct, color="black", ls="--", lw=1, alpha=0.6, label=f"commit ≈ {ct}")
    if cl is not None:
        ax.axvline(cl, color="red", ls="--", lw=1, alpha=0.6, label=f"cliff ≈ {cl}")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7, loc="lower right", ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = FIG_DIR / f"per_character_seed{seed:02d}.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_G2_eigvals(seed, traj):
    epochs = traj["epochs"]
    eigvals = traj["eigvals"]  # (T, 24)
    fig, ax = plt.subplots(figsize=(10, 5))
    for j in range(eigvals.shape[1]):
        ax.plot(epochs, eigvals[:, j], lw=1.0, alpha=0.7)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("eigenvalue of W_E_v^T W_E_v")
    ax.set_title(f"seed {seed}: eigenspectrum of W_E over training (24 eigvals)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = FIG_DIR / f"eigvals_seed{seed:02d}.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_G2_eig_identity(seed, traj):
    """Heatmap: for each checkpoint, show the dominant character of the top-1, -2, -3 eigvecs."""
    epochs = traj["epochs"]
    eigvec_comp = traj["eigvec_char_composition"]  # (T, 24, 56)
    # For each (t, eig_index), pick top-character.
    top_char = eigvec_comp.argmax(axis=2) + 1  # (T, 24)
    fig, ax = plt.subplots(figsize=(11, 5))
    n_show = 6
    for j in range(n_show):
        ax.scatter(epochs, top_char[:, j], s=18, alpha=0.7, label=f"eig {j+1}")
    ax.set_xscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("dominant character k")
    ax.set_title(f"seed {seed}: identity (dominant character) of top-6 eigenvectors over training")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = FIG_DIR / f"eig_identity_seed{seed:02d}.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ---------- B4 summary ----------

def summarize_b4(b4, trajectories):
    """For each seed, compare top-amp characters at init to observed K at end."""
    rows = []
    for seed, b in b4.items():
        traj = trajectories.get(seed)
        if traj is None:
            continue
        observed_K = identify_observed_K(traj["char_energy"][-1])
        # Rankings.
        rank_init = np.argsort(b["init_energy"])[::-1] + 1
        rank_grad = np.argsort(b["grad_energy"])[::-1] + 1
        rank_amp = np.argsort(b["amp"])[::-1] + 1

        def topn_recall(rank, n):
            top = set(rank[:n].tolist())
            return len(top & set(observed_K)) / max(1, len(observed_K))

        rows.append(dict(
            seed=seed,
            K=observed_K,
            len_K=len(observed_K),
            init_top5_recall=topn_recall(rank_init, 5),
            grad_top5_recall=topn_recall(rank_grad, 5),
            amp_top5_recall=topn_recall(rank_amp, 5),
            init_topK_recall=topn_recall(rank_init, len(observed_K)),
            grad_topK_recall=topn_recall(rank_grad, len(observed_K)),
            amp_topK_recall=topn_recall(rank_amp, len(observed_K)),
            init_top2K_recall=topn_recall(rank_init, 2 * len(observed_K)),
            grad_top2K_recall=topn_recall(rank_grad, 2 * len(observed_K)),
            amp_top2K_recall=topn_recall(rank_amp, 2 * len(observed_K)),
            commit_time=commit_time(traj),
            cliff_time=cliff_time(traj),
        ))
    return rows


def print_summary(rows):
    print("\n" + "=" * 100)
    print("B4 SUMMARY: how well does init-time character ranking predict observed K?")
    print("=" * 100)
    print(f"{'seed':>4}  {'|K|':>3}  {'commit':>7}  {'cliff':>7}  "
          f"{'i@5':>5} {'g@5':>5} {'a@5':>5}  "
          f"{'i@K':>5} {'g@K':>5} {'a@K':>5}  "
          f"{'i@2K':>5} {'g@2K':>5} {'a@2K':>5}   K")
    for r in rows:
        ct = "-" if r["commit_time"] is None else f"{r['commit_time']:>7}"
        cl = "-" if r["cliff_time"] is None else f"{r['cliff_time']:>7}"
        print(f"{r['seed']:>4}  {r['len_K']:>3}  {ct}  {cl}  "
              f"{r['init_top5_recall']:>5.2f} {r['grad_top5_recall']:>5.2f} {r['amp_top5_recall']:>5.2f}  "
              f"{r['init_topK_recall']:>5.2f} {r['grad_topK_recall']:>5.2f} {r['amp_topK_recall']:>5.2f}  "
              f"{r['init_top2K_recall']:>5.2f} {r['grad_top2K_recall']:>5.2f} {r['amp_top2K_recall']:>5.2f}   {r['K']}")
    print()
    # Aggregate.
    for col in ("init", "grad", "amp"):
        for n in ("5", "K", "2K"):
            key = f"{col}_top{n}_recall"
            mean = np.mean([r[key] for r in rows])
            print(f"  mean {key}: {mean:.3f}")


def plot_b4_predictions(b4, trajectories):
    """One panel per seed: bar chart of init/grad/amp ranking, color-coded by whether
    the character is in observed K."""
    seeds = sorted(b4.keys())
    n = len(seeds)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 3.2 * nrows))
    axes = np.atleast_2d(axes).ravel()
    for i, seed in enumerate(seeds):
        ax = axes[i]
        b = b4[seed]
        traj = trajectories[seed]
        observed_K = set(identify_observed_K(traj["char_energy"][-1]))
        order_top = np.argsort(b["amp"])[::-1][:15]
        ks = order_top + 1
        amps = b["amp"][order_top]
        colors = ["#d62728" if int(k) in observed_K else "#1f77b4" for k in ks]
        ax.bar(range(len(ks)), amps, color=colors)
        ax.set_xticks(range(len(ks)))
        ax.set_xticklabels([str(k) for k in ks], fontsize=7)
        ax.set_title(f"seed {seed}  K={sorted(observed_K)}", fontsize=8)
        ax.set_ylabel("amp = √(init·grad)")
        ax.grid(True, alpha=0.3, axis="y")
    for j in range(len(seeds), len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("B4: top-15 characters by init-time amplification √(init_E · grad_E).\n"
                 "Red = in observed K. Blue = not.", fontsize=11)
    fig.tight_layout()
    out = FIG_DIR / "b4_predictions.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_commit_vs_cliff(rows):
    fig, ax = plt.subplots(figsize=(7, 5))
    for r in rows:
        if r["commit_time"] is None or r["cliff_time"] is None:
            continue
        ax.scatter(r["commit_time"], r["cliff_time"], s=60, alpha=0.8)
        ax.annotate(str(r["seed"]),
                    (r["commit_time"], r["cliff_time"]),
                    xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.set_xlabel("commit time (epoch at which top-5 K stabilizes)")
    ax.set_ylabel("cliff time (epoch at which K-fraction > 0.5)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.set_title("Commit time vs cliff time")
    # y=x diagonal
    lim = max(max([r["commit_time"] for r in rows if r["commit_time"]], default=1),
              max([r["cliff_time"] for r in rows if r["cliff_time"]], default=1))
    ax.plot([1, lim], [1, lim], "k--", lw=0.7, alpha=0.5)
    fig.tight_layout()
    out = FIG_DIR / "commit_vs_cliff.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def main():
    with open(IN_DIR / "trajectories.pkl", "rb") as f:
        trajectories = pickle.load(f)
    with open(IN_DIR / "b4_init_gradient.pkl", "rb") as f:
        b4 = pickle.load(f)

    for seed, traj in trajectories.items():
        if len(traj["epochs"]) < 5:
            continue
        plot_G1_per_character(seed, traj)
        plot_G2_eigvals(seed, traj)
        plot_G2_eig_identity(seed, traj)
    print(f"Per-seed plots → {FIG_DIR}")

    rows = summarize_b4(b4, trajectories)
    print_summary(rows)
    plot_b4_predictions(b4, trajectories)
    plot_commit_vs_cliff(rows)


if __name__ == "__main__":
    main()
