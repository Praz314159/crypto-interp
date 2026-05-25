"""Quantify the bifurcation: when do K characters start outgrowing non-K?

For each seed, compute mean(K energy) / mean(non-K energy) at every fine-grained
step and find the commit step (ratio first exceeds 1.5x its init). Uses
crypto_interp.interp (prime-parametric: prime inferred from W_E width).

Produces:
  figures/basis_dynamics/bifurcation_overlay.png
  figures/basis_dynamics/bifurcation_overlay_normed.png
  figures/basis_dynamics/bifurcation_summary.png
"""
from __future__ import annotations

import argparse
import pickle
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from crypto_interp.interp import bifurcation_step, char_energy_batch, char_index

THRESHOLD = 1.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True,
                    help="Experiment data/ dir (contains fine_grained/ and basis_dynamics/trajectories.pkl).")
    ap.add_argument("--out-dir", default=None,
                    help="Where to write figures (default: <data-dir>/../figures/basis_dynamics).")
    args = ap.parse_args()
    data_dir = Path(args.data_dir).resolve()
    in_dir = data_dir / "fine_grained"
    traj_file = data_dir / "basis_dynamics" / "trajectories.pkl"
    out_dir = Path(args.out_dir).resolve() if args.out_dir else data_dir.parent / "figures" / "basis_dynamics"
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(traj_file, "rb") as f:
        trajectories = pickle.load(f)

    fig_overlay, ax = plt.subplots(figsize=(11, 5.5))
    fig_normed, ax_n = plt.subplots(figsize=(11, 5.5))
    commit_steps = {}
    cmap = plt.cm.viridis

    basis = ci = None
    fg_files = sorted(in_dir.glob("seed*_fine_grained.pt"))
    for i, fg_path in enumerate(fg_files):
        seed = int(re.match(r"seed(\d+)", fg_path.name).group(1))
        d = torch.load(fg_path, weights_only=False)
        W_E = d["W_E"]                      # (T, d_model, vocab)
        epochs = d["epochs"].numpy()
        if ci is None:
            p = W_E.shape[2] - 1
            basis, ci = char_index(p)
        char_E = char_energy_batch(W_E[:, :, :ci.p], basis, ci)   # (T, n_chars)

        final_ce = trajectories[seed]["char_energy"][-1]
        final_K = [k + 1 for k, e in enumerate(final_ce) if e >= 0.05 * final_ce.max()]
        K_mask = np.zeros(char_E.shape[1], dtype=bool)
        for k in final_K:
            K_mask[k - 1] = True

        K_mean = char_E[:, K_mask].mean(axis=1)
        nonK_mean = char_E[:, ~K_mask].mean(axis=1)
        ratio = K_mean / np.where(nonK_mean > 0, nonK_mean, 1.0)
        ratio_norm = ratio / ratio[0]

        step = bifurcation_step(char_E, K_mask, ratio=THRESHOLD)
        commit = int(epochs[step]) if step is not None else None
        commit_steps[seed] = commit

        color = cmap(i / max(1, len(fg_files) - 1))
        ax.plot(epochs, ratio, color=color, lw=1.0, alpha=0.85, label=f"seed {seed} ({commit})")
        ax_n.plot(epochs, ratio_norm, color=color, lw=1.0, alpha=0.85, label=f"seed {seed} ({commit})")
        if step is not None:
            ax_n.scatter([commit], [ratio_norm[step]], color=color, s=22, zorder=5)

    ax.set_xlabel("step"); ax.set_ylabel("mean(K energy) / mean(non-K energy)")
    ax.set_title("Bifurcation: K vs non-K energy ratio, all seeds (per-step)")
    ax.set_yscale("log"); ax.axhline(1.0, color="black", ls="--", lw=0.6, alpha=0.4)
    ax.grid(True, alpha=0.3); ax.legend(fontsize=7, ncol=2, loc="lower right")
    fig_overlay.tight_layout()
    out1 = out_dir / "bifurcation_overlay.png"
    fig_overlay.savefig(out1, dpi=130, bbox_inches="tight"); plt.close(fig_overlay)

    ax_n.set_xlabel("step"); ax_n.set_ylabel("ratio(t) / ratio(0)")
    ax_n.set_title(f"Bifurcation normed to init. Marker = commit step (ratio > {THRESHOLD}x init).")
    ax_n.set_yscale("log")
    ax_n.axhline(THRESHOLD, color="red", ls="--", lw=0.7, alpha=0.5, label=f"threshold = {THRESHOLD}x")
    ax_n.grid(True, alpha=0.3); ax_n.legend(fontsize=7, ncol=2, loc="lower right")
    fig_normed.tight_layout()
    out2 = out_dir / "bifurcation_overlay_normed.png"
    fig_normed.savefig(out2, dpi=130, bbox_inches="tight"); plt.close(fig_normed)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    finite = [c for c in commit_steps.values() if c is not None]
    if finite:
        ax.hist(finite, bins=15, color="#2ca02c", alpha=0.8)
    ax.set_xlabel(f"commit step (ratio exceeds {THRESHOLD}x init)"); ax.set_ylabel("seeds")
    ax.set_title(f"Distribution of bifurcation step across {len(finite)} / {len(commit_steps)} seeds. "
                 f"median = {int(np.median(finite)) if finite else 'n/a'}")
    ax.grid(True, alpha=0.3); fig.tight_layout()
    out3 = out_dir / "bifurcation_summary.png"
    fig.savefig(out3, dpi=130, bbox_inches="tight"); plt.close(fig)

    print(f"\nCommit steps (threshold = {THRESHOLD}x init ratio):")
    for s in sorted(commit_steps):
        print(f"  seed {s:>2}: {commit_steps[s]}")
    if finite:
        print(f"\n  median: {int(np.median(finite))}, mean: {int(np.mean(finite))}, "
              f"min: {min(finite)}, max: {max(finite)}")
    print(f"\nSaved {out1}\n      {out2}\n      {out3}")


if __name__ == "__main__":
    main()
