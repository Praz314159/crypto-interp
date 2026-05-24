"""Analyze the freeze-W_E ablation vs the control.

For each (freeze, control), plot:
  - log10 train + test loss vs step
  - W_U character energy: bifurcation visible? (analog of the W_E bifurcation
    in normal training)
  - K-alignment recall of W_U over time
  - For sanity, W_E energy (should be constant in freeze; bifurcates in control)
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
IN_DIR = ROOT / "data" / "freeze_we"
TRAJ_FILE = ROOT / "data" / "basis_dynamics" / "trajectories.pkl"
FIG_DIR = ROOT / "figures" / "basis_dynamics"


def order_of(k, n=112):
    return n // math.gcd(k, n)


ORDER_COLOR = {
    2: "#9467bd", 4: "#8c564b", 7: "#bcbd22", 8: "#17becf",
    14: "#e377c2", 16: "#1f77b4", 28: "#2ca02c", 56: "#ff7f0e", 112: "#d62728",
}


def topk_recall(X_t, K_set):
    k = len(K_set)
    top = set((np.argsort(X_t)[-k:] + 1).tolist())
    return len(top & K_set) / max(1, len(K_set))


def main():
    with open(TRAJ_FILE, "rb") as f:
        trajectories = pickle.load(f)

    files = sorted(IN_DIR.glob("seed*_*.pt"))
    print(f"Found {len(files)} freeze/control files")
    if not files:
        return

    # Group by seed.
    by_seed = {}
    for p in files:
        m = re.match(r"seed(\d+)_(freeze_we|control)\.pt", p.name)
        if not m:
            continue
        seed = int(m.group(1))
        tag = m.group(2)
        by_seed.setdefault(seed, {})[tag] = p

    for seed, paths in by_seed.items():
        traj = trajectories.get(seed)
        if traj is None:
            print(f"  seed {seed}: no trajectory, skip")
            continue
        final_ce = traj["char_energy"][-1]
        Kset = set(int(k + 1) for k, e in enumerate(final_ce)
                   if e >= 0.05 * final_ce.max())
        K_list = sorted(Kset)
        print(f"\n=== seed {seed}, final K (full-training) = {K_list} ===")

        # 1) Loss curves.
        fig, axes = plt.subplots(1, 2, figsize=(14, 4.5), sharey=True)
        for tag, color in [("freeze_we", "#d62728"), ("control", "#1f77b4")]:
            if tag not in paths:
                continue
            d = torch.load(paths[tag], weights_only=False)
            tl = d["train_losses"].numpy()
            xl = d["test_losses"].numpy()
            steps = np.arange(len(tl))
            axes[0].plot(steps, np.log10(tl), color=color, lw=1.0, label=tag)
            axes[1].plot(steps, np.log10(xl), color=color, lw=1.0, label=tag)
        axes[0].set_title("train loss (log10)"); axes[0].set_xlabel("step")
        axes[1].set_title("test loss (log10)"); axes[1].set_xlabel("step")
        for ax in axes:
            ax.grid(True, alpha=0.3); ax.legend(fontsize=10)
        fig.suptitle(f"seed {seed}: freeze-W_E vs control", fontsize=12)
        fig.tight_layout()
        out = FIG_DIR / f"freeze_we_loss_seed{seed:02d}.png"
        fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)

        # 2) W_U / W_E character trajectories per tag.
        for tag in paths:
            d = torch.load(paths[tag], weights_only=False)
            char_E_WU = d["char_E_WU"].numpy()
            char_E_WE = d["char_E_WE"].numpy()
            T = char_E_WU.shape[0]
            steps = np.arange(T)
            fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
            for ax, char_E, label in [
                (axes[0], char_E_WU, "W_U"),
                (axes[1], char_E_WE, "W_E"),
            ]:
                for k in range(1, 57):
                    o = order_of(k)
                    color = ORDER_COLOR.get(o, "#aaaaaa")
                    is_K = k in Kset
                    ax.plot(steps, char_E[:, k - 1],
                            color=color, alpha=0.95 if is_K else 0.15,
                            lw=1.4 if is_K else 0.6,
                            label=f"k={k} (o={o})" if is_K else None)
                ax.set_yscale("log")
                ax.set_ylabel(f"{label} energy in character k")
                ax.grid(True, alpha=0.3)
                ax.set_title(f"{label} per-character energy [{tag}]")
            axes[0].legend(fontsize=7, loc="lower right", ncol=2)
            axes[-1].set_xlabel("step")
            fig.suptitle(f"seed {seed}: per-character energy in W_U and W_E [{tag}]")
            fig.tight_layout()
            out = FIG_DIR / f"freeze_we_chars_seed{seed:02d}_{tag}.png"
            fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)

        # 3) Recall trajectories: alignment to full-training K.
        fig, ax = plt.subplots(figsize=(11, 5))
        for tag, color in [("freeze_we", "#d62728"), ("control", "#1f77b4")]:
            if tag not in paths:
                continue
            d = torch.load(paths[tag], weights_only=False)
            for which, ls in [("char_E_WU", "-"), ("char_E_WE", ":")]:
                arr = d[which].numpy()
                T = arr.shape[0]
                rec = np.array([topk_recall(arr[t], Kset) for t in range(T)])
                ax.plot(np.arange(T), rec, color=color, ls=ls, lw=1.4,
                        label=f"{tag}  ({which})")
        ax.axhline(0.5, color="black", ls=":", lw=0.5, alpha=0.5)
        ax.axhline(0.8, color="black", ls=":", lw=0.5, alpha=0.5)
        ax.set_xlabel("step")
        ax.set_ylabel("top-|K| recall of full-training K")
        ax.set_title(f"seed {seed}: K-alignment of W_U / W_E over training "
                     f"(K from full training)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc="lower right")
        ax.set_ylim(0, 1.02)
        fig.tight_layout()
        out = FIG_DIR / f"freeze_we_recall_seed{seed:02d}.png"
        fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)

        # 4) Final K identified per tag (top-|K| at end of training).
        for tag in paths:
            d = torch.load(paths[tag], weights_only=False)
            final_WU = d["char_E_WU"].numpy()[-1]
            final_WE = d["char_E_WE"].numpy()[-1]
            top_WU = sorted((np.argsort(final_WU)[-len(K_list):] + 1).tolist())
            top_WE = sorted((np.argsort(final_WE)[-len(K_list):] + 1).tolist())
            print(f"  [{tag}] final top-|K| in W_U = {top_WU}")
            print(f"  [{tag}] final top-|K| in W_E = {top_WE}")
            tl = d["train_losses"].numpy()[-1]
            xl = d["test_losses"].numpy()[-1]
            print(f"  [{tag}] log10 final train/test loss: {np.log10(tl):.3f} / {np.log10(xl):.3f}")

    print(f"\nFigures saved to {FIG_DIR}/freeze_we_*")


if __name__ == "__main__":
    main()
