"""Bifurcation analysis across d_mlp budgets.

For each per-step fine-grained run (across d_mlp ∈ {32, 64, 128, 512} × seeds):
  - Identify final K from the corresponding full-training checkpoint.
  - Compute the K-vs-non-K energy ratio at every step.
  - Find the bifurcation step (first step where the ratio exceeds 1.5× init).

Plot the ratio trajectories overlaid by d_mlp, and summarize bifurcation-step
distributions per d_mlp.

Outputs:
  figures/basis_dynamics/bifurcation_by_dmlp_overlay.png
  figures/basis_dynamics/bifurcation_by_dmlp_summary.png
  data/basis_dynamics/bifurcation_by_dmlp.csv
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
FG_DIR = ROOT / "data" / "fine_grained"
FIG_DIR = ROOT / "figures" / "basis_dynamics"
OUT_DIR = ROOT / "data" / "basis_dynamics"


def build_char_basis():
    basis, names, _ = multiplicative_fourier_basis(113)
    char_idx = {}
    for i, nm in enumerate(names):
        m = re.match(r"mul (cos|sin) (\d+)", nm)
        if m:
            kk = int(m.group(2))
            char_idx.setdefault(kk, []).append(i)
    return basis.double(), char_idx


def char_energy_trajectory(W_E_stack, basis, char_idx):
    """W_E_stack: (T, d_model, vocab). Returns (T, 56) char energies."""
    W_v = W_E_stack[:, :, :113].double()
    coef = torch.einsum("kp,tdp->tkd", basis, W_v)
    E = (coef ** 2).sum(dim=2)
    T = E.shape[0]
    out = np.zeros((T, 56))
    for k, rows in char_idx.items():
        out[:, k - 1] = E[:, rows].sum(dim=1).cpu().numpy()
    return out


def find_final_K(run_dir):
    ck = sorted(run_dir.glob("checkpoint_*.pt"))
    if not ck:
        return None
    state = torch.load(ck[-1], weights_only=False, map_location="cpu")
    return state["model_state"]["embed.W_E"]


def main():
    basis, char_idx = build_char_basis()
    # Discover fine-grained files: (d_mlp, seed, path).
    runs = []
    # baseline d_mlp=512 — files seedNN_fine_grained.pt
    for p in sorted(FG_DIR.glob("seed??_fine_grained.pt")):
        m = re.match(r"seed(\d+)_fine_grained\.pt", p.name)
        if m:
            runs.append((512, int(m.group(1)), p))
    # d_mlp != 512 — files dmlpXXX_seedNN_fine_grained.pt
    for p in sorted(FG_DIR.glob("dmlp*_seed*_fine_grained.pt")):
        m = re.match(r"dmlp(\d+)_seed(\d+)_fine_grained\.pt", p.name)
        if m:
            runs.append((int(m.group(1)), int(m.group(2)), p))

    print(f"Found {len(runs)} fine-grained runs")
    summary = []
    trajectories_by_dmlp = {}

    for d_mlp, seed, fg_path in runs:
        # Locate the matching full-training run dir to read final K.
        if d_mlp == 512:
            run_dir = RUNS / f"dmodel_24_seed{seed}"
        else:
            run_dir = RUNS / f"dmodel_24_dmlp_{d_mlp}_seed{seed}"
        W_E_final = find_final_K(run_dir)
        if W_E_final is None:
            print(f"  d_mlp={d_mlp:>3} seed={seed:>2}: no final checkpoint, skip")
            continue
        # Final char-energy → K.
        coef = torch.einsum("kp,dp->kd", basis, W_E_final[:, :113].double())
        final_E = np.zeros(56)
        for k, rs in char_idx.items():
            final_E[k - 1] = float((coef[rs] ** 2).sum())
        K = set(int(k + 1) for k, e in enumerate(final_E) if e >= 0.05 * final_E.max())
        nonK = set(range(1, 57)) - K

        # Per-step character energy in fine-grained data.
        d = torch.load(fg_path, weights_only=False)
        epochs = d["epochs"].numpy()
        W_E_stack = d["W_E"]
        char_E = char_energy_trajectory(W_E_stack, basis, char_idx)

        K_mean = char_E[:, [k - 1 for k in K]].mean(axis=1)
        nonK_mean = char_E[:, [k - 1 for k in nonK]].mean(axis=1) if nonK else np.ones_like(K_mean)
        ratio = K_mean / np.where(nonK_mean > 0, nonK_mean, 1.0)
        ratio_norm = ratio / ratio[0]
        above = np.where(ratio_norm > 1.5)[0]
        bif = int(epochs[above[0]]) if len(above) else None

        summary.append(dict(d_mlp=d_mlp, seed=seed, K=sorted(K),
                            len_K=len(K), bif=bif))
        trajectories_by_dmlp.setdefault(d_mlp, []).append(
            (seed, epochs, ratio_norm)
        )
        print(f"  d_mlp={d_mlp:>3} seed={seed:>2}: |K|={len(K)}, "
              f"bif={'-' if bif is None else bif}")

    # CSV.
    csv_path = OUT_DIR / "bifurcation_by_dmlp.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["d_mlp", "seed", "K", "len_K", "bif"])
        for r in summary:
            w.writerow([r["d_mlp"], r["seed"], "|".join(map(str, r["K"])),
                        r["len_K"], r["bif"] if r["bif"] is not None else ""])
    print(f"\nSaved {csv_path}")

    # ---- Overlay plot: ratio trajectories by d_mlp ----
    cmap = {32: "#d62728", 64: "#ff7f0e", 128: "#9467bd", 512: "#1f77b4"}
    fig, ax = plt.subplots(figsize=(11, 6))
    for d_mlp in sorted(trajectories_by_dmlp):
        for seed, epochs, ratio_norm in trajectories_by_dmlp[d_mlp]:
            ax.plot(epochs, ratio_norm, color=cmap.get(d_mlp, "#7f7f7f"),
                    alpha=0.6, lw=1.2,
                    label=f"d_mlp={d_mlp} s{seed}")
    ax.axhline(1.5, color="red", ls="--", lw=0.7, alpha=0.5, label="bifurcation thresh")
    ax.set_yscale("log")
    ax.set_xlabel("step")
    ax.set_ylabel("K/non-K energy ratio, normalized to init")
    ax.set_title("Basin bifurcation across d_mlp budgets")
    ax.grid(True, alpha=0.3)
    # Single legend entry per d_mlp via deduplication.
    handles, labels = ax.get_legend_handles_labels()
    seen = {}
    for h, lab in zip(handles, labels):
        key = lab.split(" s")[0]
        if key not in seen:
            seen[key] = h
    ax.legend(seen.values(), seen.keys(), fontsize=9, loc="lower right")
    fig.tight_layout()
    out1 = FIG_DIR / "bifurcation_by_dmlp_overlay.png"
    fig.savefig(out1, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out1}")

    # ---- Summary: bifurcation step vs d_mlp ----
    fig, ax = plt.subplots(figsize=(8, 5))
    for r in summary:
        if r["bif"] is None:
            continue
        ax.scatter(r["d_mlp"], r["bif"],
                   color=cmap.get(r["d_mlp"], "#7f7f7f"),
                   s=80, alpha=0.85, edgecolor="black", linewidth=0.4)
        ax.annotate(f"s{r['seed']}", (r["d_mlp"], r["bif"]),
                    xytext=(4, 4), textcoords="offset points", fontsize=8)
    # Connect medians.
    by_d = {}
    for r in summary:
        if r["bif"] is not None:
            by_d.setdefault(r["d_mlp"], []).append(r["bif"])
    xs = sorted(by_d)
    ys = [np.median(by_d[x]) for x in xs]
    ax.plot(xs, ys, "k--", lw=0.8, alpha=0.6, label="median")
    ax.set_xscale("log", base=2)
    ax.set_xticks(xs)
    ax.set_xticklabels([str(x) for x in xs])
    ax.set_xlabel("d_mlp budget")
    ax.set_ylabel("bifurcation step (ratio > 1.5× init)")
    ax.set_title(f"Basin commitment time vs d_mlp ({len(summary)} runs)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out2 = FIG_DIR / "bifurcation_by_dmlp_summary.png"
    fig.savefig(out2, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out2}")


if __name__ == "__main__":
    main()
