"""Analyze the fine-grained (per-step) W_E trajectories in [0, 500).

For each seed:
  - Compute per-character W_E energy at every step.
  - Compute eigvec identity at every step.
  - Identify the "commit step": the earliest step after which the top-|K|
    characters by energy contain the final K (where final K is taken from the
    last checkpoint of the corresponding full run).

Produces:
  figures/basis_dynamics/fg_per_character_<seed>.png   (per-step trajectory)
  figures/basis_dynamics/fg_eig_identity_<seed>.png    (per-step eigvec id)
  figures/basis_dynamics/fg_summary.png                (commit-step distribution)
  figures/basis_dynamics/fg_summary.csv                (per-seed commit step)
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
FIG_DIR.mkdir(parents=True, exist_ok=True)


def order_of(k: int, n: int = 112) -> int:
    return n // math.gcd(k, n)


ORDER_COLOR = {
    2: "#9467bd", 4: "#8c564b", 7: "#bcbd22", 8: "#17becf",
    14: "#e377c2", 16: "#1f77b4", 28: "#2ca02c", 56: "#ff7f0e", 112: "#d62728",
}


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


def char_energies_batch(W_E_stack: torch.Tensor, basis: torch.Tensor,
                        char_index: dict[int, list[int]]) -> np.ndarray:
    """W_E_stack: (T, d_model, vocab). Returns (T, 56)."""
    W_v = W_E_stack[:, :, :113].double()
    coef = torch.einsum("kp,tdp->tkd", basis.double(), W_v)
    E = (coef ** 2).sum(dim=2)  # (T, n_basis)
    T = E.shape[0]
    out = np.zeros((T, 56))
    for k, rows in char_index.items():
        out[:, k - 1] = E[:, rows].sum(dim=1).cpu().numpy()
    return out


def commit_step(char_E: np.ndarray, final_K: list[int]) -> int | None:
    """Earliest step at which the top-|K| characters by energy are a superset
    of final_K. (If always true, returns 0; if never, returns None.)"""
    Kset = set(final_K)
    n = len(Kset)
    T = char_E.shape[0]
    last_bad = -1
    for t in range(T):
        top_n = set(np.argsort(char_E[t])[-n:] + 1)
        if not Kset.issubset(top_n):
            last_bad = t
    if last_bad == T - 1:
        return None
    return last_bad + 1


def commit_step_strict(char_E: np.ndarray, final_K: list[int]) -> int | None:
    """Earliest step after which the top-|K| set equals final_K exactly for all
    subsequent steps in the window."""
    Kset = set(final_K)
    n = len(Kset)
    T = char_E.shape[0]
    last_bad = -1
    for t in range(T):
        top_n = set(np.argsort(char_E[t])[-n:] + 1)
        if top_n != Kset:
            last_bad = t
    if last_bad == T - 1:
        return None
    return last_bad + 1


def main():
    basis, char_index = build_char_basis()
    with open(TRAJ_FILE, "rb") as f:
        trajectories = pickle.load(f)

    fg_files = sorted(IN_DIR.glob("seed*_fine_grained.pt"))
    print(f"Found {len(fg_files)} fine-grained files")
    summary_rows = []

    for fg_path in fg_files:
        m = re.match(r"seed(\d+)_fine_grained\.pt", fg_path.name)
        if not m:
            continue
        seed = int(m.group(1))
        d = torch.load(fg_path, weights_only=False)
        epochs = d["epochs"].numpy()
        W_E = d["W_E"]  # (T, d_model, vocab)
        T = W_E.shape[0]

        # Compute per-character energy at every step.
        char_E = char_energies_batch(W_E, basis, char_index)  # (T, 56)

        # Final K from the corresponding full-training trajectory.
        traj = trajectories.get(seed)
        if traj is None:
            print(f"  seed {seed}: no full trajectory, skipping")
            continue
        final_ce = traj["char_energy"][-1]
        thresh = 0.05 * final_ce.max()
        final_K = sorted([int(k + 1) for k, e in enumerate(final_ce) if e >= thresh])

        cs_loose = commit_step(char_E, final_K)
        cs_strict = commit_step_strict(char_E, final_K)
        train_loss = d["train_losses"].numpy()
        test_loss = d["test_losses"].numpy()

        print(f"  seed {seed:>2}: |K|={len(final_K)}, K={final_K}, "
              f"commit_loose={cs_loose}, commit_strict={cs_strict}")
        summary_rows.append(dict(
            seed=seed, K=final_K, len_K=len(final_K),
            commit_loose=cs_loose, commit_strict=cs_strict,
            test_loss_at_500=float(test_loss[-1]),
        ))

        # --- per-character plot ---
        fig, ax = plt.subplots(figsize=(11, 5))
        for k in range(1, 57):
            o = order_of(k)
            color = ORDER_COLOR.get(o, "#aaaaaa")
            is_K = k in final_K
            ax.plot(epochs, char_E[:, k - 1],
                    color=color, alpha=0.95 if is_K else 0.18,
                    lw=1.4 if is_K else 0.6,
                    label=(f"k={k} (o={o})" if is_K else None))
        if cs_loose is not None:
            ax.axvline(cs_loose, color="black", ls="--", lw=1, alpha=0.7,
                       label=f"commit-loose @ {cs_loose}")
        if cs_strict is not None:
            ax.axvline(cs_strict, color="red", ls=":", lw=1, alpha=0.7,
                       label=f"commit-strict @ {cs_strict}")
        ax.set_yscale("log")
        ax.set_xlabel("epoch (per-step)")
        ax.set_ylabel("energy in character k")
        ax.set_title(f"seed {seed}: per-step character energy, epochs [0, 500). "
                     f"K={final_K}")
        ax.legend(fontsize=7, loc="lower right", ncol=2)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        out = FIG_DIR / f"fg_per_character_seed{seed:02d}.png"
        fig.savefig(out, dpi=130, bbox_inches="tight")
        plt.close(fig)

    # --- summary: distribution of commit steps ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    losse = [r["commit_loose"] for r in summary_rows if r["commit_loose"] is not None]
    strict = [r["commit_strict"] for r in summary_rows if r["commit_strict"] is not None]
    if losse:
        axes[0].hist(losse, bins=20, color="#1f77b4", alpha=0.8)
        axes[0].set_title(f"commit-loose (top-|K| ⊇ final K) — {len(losse)}/{len(summary_rows)} seeds")
        axes[0].set_xlabel("step")
    if strict:
        axes[1].hist(strict, bins=20, color="#d62728", alpha=0.8)
        axes[1].set_title(f"commit-strict (top-|K| = final K) — {len(strict)}/{len(summary_rows)} seeds")
        axes[1].set_xlabel("step")
    for ax in axes:
        ax.grid(True, alpha=0.3)
    fig.suptitle("Fine-grained basin commitment: when does the K-identity emerge?", fontsize=11)
    fig.tight_layout()
    out = FIG_DIR / "fg_summary.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {out}")

    # CSV
    import csv
    csv_path = FIG_DIR / "fg_summary.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed", "len_K", "K", "commit_loose", "commit_strict", "test_loss_at_500"])
        for r in summary_rows:
            w.writerow([r["seed"], r["len_K"], "|".join(map(str, r["K"])),
                        r["commit_loose"], r["commit_strict"], f"{r['test_loss_at_500']:.4f}"])
    print(f"Saved {csv_path}")


if __name__ == "__main__":
    main()
