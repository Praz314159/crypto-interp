"""When does the gradient (resp. Adam m_t, v_t) first align with the final K?

For each seed, compute at every step t:
  rank_recall@K(X[t])  = |{top-|K| chars by X[t]} ∩ K| / |K|
where X is one of {W_E energy, grad energy, m_t energy, v_t energy}.

Plot the recall trajectories. Find the step at which each signal first reaches
some threshold (say, recall ≥ 0.8) and compare to the W_E bifurcation step.

Produces:
  figures/basis_dynamics/grad_alignment_<seed>.png
  figures/basis_dynamics/grad_alignment_summary.png
  figures/basis_dynamics/grad_alignment_summary.csv
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
IN_DIR = ROOT / "data" / "with_grad"
TRAJ_FILE = ROOT / "data" / "basis_dynamics" / "trajectories.pkl"
FIG_DIR = ROOT / "figures" / "basis_dynamics"


def topk_recall(X_t, K_set, k=None):
    """X_t: array length 56. Returns recall@k for default k = |K_set|."""
    if k is None:
        k = len(K_set)
    top = set((np.argsort(X_t)[-k:] + 1).tolist())
    return len(top & K_set) / max(1, len(K_set))


def first_step_threshold(recall_traj, threshold):
    above = np.where(recall_traj >= threshold)[0]
    return int(above[0]) if len(above) else None


def main():
    with open(TRAJ_FILE, "rb") as f:
        trajectories = pickle.load(f)

    seeds = []
    summary = []
    fg_files = sorted(IN_DIR.glob("seed*_with_grad.pt"))
    print(f"Found {len(fg_files)} files with grad+adam data")

    # Collect all-seed mean trajectory.
    all_seed_recalls = {"E": [], "G": [], "M": [], "V": []}

    for fg_path in fg_files:
        seed = int(re.match(r"seed(\d+)", fg_path.name).group(1))
        d = torch.load(fg_path, weights_only=False)
        E = d["char_E"].numpy()  # (T, 56)
        G = d["char_G"].numpy()
        M = d["char_M"].numpy()
        V = d["char_V"].numpy()
        T = E.shape[0]

        traj = trajectories.get(seed)
        if traj is None:
            print(f"  seed {seed}: no traj, skip")
            continue
        final_ce = traj["char_energy"][-1]
        Kset = set(int(k + 1) for k, e in enumerate(final_ce)
                   if e >= 0.05 * final_ce.max())
        nK = len(Kset)

        rec_E = np.array([topk_recall(E[t], Kset) for t in range(T)])
        rec_G = np.array([topk_recall(G[t], Kset) for t in range(T)])
        rec_M = np.array([topk_recall(M[t], Kset) for t in range(T)])
        rec_V = np.array([topk_recall(V[t], Kset) for t in range(T)])

        all_seed_recalls["E"].append(rec_E)
        all_seed_recalls["G"].append(rec_G)
        all_seed_recalls["M"].append(rec_M)
        all_seed_recalls["V"].append(rec_V)

        seeds.append(seed)
        summary.append(dict(
            seed=seed, K_size=nK,
            step_E50=first_step_threshold(rec_E, 0.5),
            step_G50=first_step_threshold(rec_G, 0.5),
            step_M50=first_step_threshold(rec_M, 0.5),
            step_V50=first_step_threshold(rec_V, 0.5),
            step_E80=first_step_threshold(rec_E, 0.8),
            step_G80=first_step_threshold(rec_G, 0.8),
            step_M80=first_step_threshold(rec_M, 0.8),
            step_V80=first_step_threshold(rec_V, 0.8),
            final_recall_E=float(rec_E[-1]),
            final_recall_G=float(rec_G[-1]),
            final_recall_M=float(rec_M[-1]),
            final_recall_V=float(rec_V[-1]),
        ))

        # Per-seed plot.
        fig, ax = plt.subplots(figsize=(10, 5))
        steps = np.arange(T)
        ax.plot(steps, rec_E, label="W_E energy", color="#1f77b4", lw=1.6)
        ax.plot(steps, rec_G, label="gradient", color="#d62728", lw=1.6)
        ax.plot(steps, rec_M, label="Adam m_t", color="#2ca02c", lw=1.6)
        ax.plot(steps, rec_V, label="Adam v_t", color="#ff7f0e", lw=1.6)
        ax.axhline(0.5, color="black", ls=":", lw=0.5, alpha=0.5)
        ax.axhline(0.8, color="black", ls=":", lw=0.5, alpha=0.5)
        ax.set_xlabel("step")
        ax.set_ylabel("top-|K| recall = |top-|K| ∩ K| / |K|")
        ax.set_title(f"seed {seed}: K-alignment of W_E vs gradient vs Adam m/v over [0, 500)")
        ax.legend(fontsize=9, loc="lower right")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.02)
        fig.tight_layout()
        out = FIG_DIR / f"grad_alignment_seed{seed:02d}.png"
        fig.savefig(out, dpi=130, bbox_inches="tight")
        plt.close(fig)

    # Mean trajectories.
    fig, ax = plt.subplots(figsize=(10, 5.5))
    if all_seed_recalls["E"]:
        T = len(all_seed_recalls["E"][0])
        steps = np.arange(T)
        for key, label, color in [
            ("E", "W_E energy", "#1f77b4"),
            ("G", "gradient", "#d62728"),
            ("M", "Adam m_t", "#2ca02c"),
            ("V", "Adam v_t", "#ff7f0e"),
        ]:
            stacked = np.stack(all_seed_recalls[key])  # (n_seeds, T)
            mean = stacked.mean(axis=0)
            lo = np.percentile(stacked, 25, axis=0)
            hi = np.percentile(stacked, 75, axis=0)
            ax.plot(steps, mean, color=color, lw=2.0, label=f"{label} (mean)")
            ax.fill_between(steps, lo, hi, color=color, alpha=0.18,
                            label=f"{label} (IQR)")
        ax.set_xlabel("step")
        ax.set_ylabel("top-|K| recall")
        ax.set_title(f"Mean K-alignment across {len(all_seed_recalls['E'])} seeds. "
                     f"Shaded = IQR.")
        ax.axhline(0.5, color="black", ls=":", lw=0.5, alpha=0.5)
        ax.axhline(0.8, color="black", ls=":", lw=0.5, alpha=0.5)
        ax.legend(fontsize=9, loc="lower right", ncol=2)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.02)
        fig.tight_layout()
        out = FIG_DIR / "grad_alignment_summary.png"
        fig.savefig(out, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out}")

    # Table.
    print()
    print(f"{'seed':>4} {'|K|':>3}  {'E@50':>5} {'G@50':>5} {'M@50':>5} {'V@50':>5}  "
          f"{'E@80':>5} {'G@80':>5} {'M@80':>5} {'V@80':>5}  "
          f"{'recE':>5} {'recG':>5} {'recM':>5} {'recV':>5}")
    def fmt(x): return f"{x:>5}" if x is not None else "    -"
    for r in summary:
        print(f"{r['seed']:>4} {r['K_size']:>3}  "
              f"{fmt(r['step_E50'])} {fmt(r['step_G50'])} {fmt(r['step_M50'])} {fmt(r['step_V50'])}  "
              f"{fmt(r['step_E80'])} {fmt(r['step_G80'])} {fmt(r['step_M80'])} {fmt(r['step_V80'])}  "
              f"{r['final_recall_E']:>5.2f} {r['final_recall_G']:>5.2f} "
              f"{r['final_recall_M']:>5.2f} {r['final_recall_V']:>5.2f}")

    # Aggregate step-to-threshold.
    print()
    for thresh in (50, 80):
        for key in ("E", "G", "M", "V"):
            col = f"step_{key}{thresh}"
            vals = [r[col] for r in summary if r[col] is not None]
            mn = int(np.median(vals)) if vals else None
            print(f"  median step to {key} recall >= {thresh/100}: "
                  f"{mn}  ({len(vals)}/{len(summary)} seeds)")

    # CSV
    import csv
    csv_path = FIG_DIR / "grad_alignment_summary.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        for r in summary:
            w.writerow(r)
    print(f"Saved {csv_path}")


if __name__ == "__main__":
    main()
