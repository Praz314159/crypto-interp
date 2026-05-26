"""Basin-commitment visualization for a single run.

Shows how quickly the model commits to its final character basis, relative to
the grokking cliff. Three panels:
  (a) per-character energy trajectories, final K highlighted, with the
      bifurcation / commit / cliff steps marked and test loss overlaid.
  (b) K-vs-non-K mean energy over time.
  (c) K/non-K energy ratio (normalized to init) — the commitment signal.

All trajectory analysis is delegated to crypto_interp.interp (prime-parametric):
``char_index``/``char_energy_batch`` for the basis, and ``bifurcation_step`` /
``commit_step`` / ``cliff_step`` for the dynamics markers.

The per-character energy trajectory comes from either:
  --metrics METRICS.PT   (uses the logged ``freq_energies`` directly), or
  --fg-file FG.PT        (per-step ``W_E``; energy recomputed via char_energy_batch).

Usage:
    python experiments/003_dmodel_sweep_p113/scripts/viz_basin_commitment.py \
        --metrics experiments/003_dmodel_sweep_p113/runs/dmodel_24_dmlp_20_wd2_seed1/metrics.pt \
        --losses  experiments/003_dmodel_sweep_p113/runs/dmodel_24_dmlp_20_wd2_seed1/losses.pt \
        --tag dmlp20_wd2_seed1
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from crypto_interp.interp import (
    bifurcation_step,
    char_energy_batch,
    char_index,
    cliff_step,
    commit_step,
    order_of,
)

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures" / "basis_dynamics"

ORDER_COLOR = {
    2: "#9467bd", 4: "#8c564b", 7: "#bcbd22", 8: "#17becf",
    14: "#e377c2", 16: "#1f77b4", 28: "#2ca02c", 56: "#ff7f0e", 112: "#d62728",
}


def load_char_traj(args, basis, ci, p):
    """Return (epochs, char_E[T, n_chars]) from metrics.pt or a fine-grained file."""
    if args.metrics:
        M = torch.load(args.metrics, weights_only=False)
        return np.asarray(M["epochs"]), np.asarray(M["freq_energies"])
    d = torch.load(args.fg_file, weights_only=False)
    char_E = char_energy_batch(d["W_E"][:, :, :p], basis, ci)
    return d["epochs"].numpy(), char_E


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--metrics", help="metrics.pt with logged freq_energies + epochs.")
    src.add_argument("--fg-file", help="fine-grained .pt with per-step W_E + epochs.")
    ap.add_argument("--losses", default=None, help="losses.pt for the test-loss cliff overlay.")
    ap.add_argument("--k", default=None, help="Explicit K (comma-separated). Default: top chars >=5%% of final-row max.")
    ap.add_argument("--p", type=int, default=113)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    p = args.p
    basis, ci = char_index(p)
    n_chars = max(ci.freqs)
    epochs, char_E = load_char_traj(args, basis, ci, p)

    # Final K: explicit, else top characters >= 5% of the final-row max energy.
    final = char_E[-1]
    if args.k is not None:
        K = sorted(int(x) for x in args.k.split(","))
    else:
        K = sorted(int(k) for k in ci.freqs if final[k - 1] >= 0.05 * final.max())
    K_mask = np.zeros(n_chars, dtype=bool)
    K_mask[[k - 1 for k in K]] = True
    nonK = [k for k in range(1, n_chars + 1) if k not in K]
    print(f"K = {K} (orders {[order_of(k, p) for k in K]})")

    # Canonical dynamics markers (in epoch units via the epochs axis).
    bif_i = bifurcation_step(char_E, K_mask, ratio=1.5)
    com_i = commit_step(char_E, K, mode="subset")
    bif = int(epochs[bif_i]) if bif_i is not None else None
    com = int(epochs[com_i]) if com_i is not None else None
    cliff = None
    test_losses = None
    if args.losses:
        test_losses = np.asarray(torch.load(args.losses, weights_only=False)["test_losses"])
        ci_idx = cliff_step(test_losses, thresh=0.1)
        cliff = int(ci_idx) if ci_idx is not None else None
    print(f"bifurcation={bif}  commit={com}  cliff={cliff}")

    K_mean = char_E[:, K_mask].mean(axis=1)
    nonK_mean = char_E[:, ~K_mask].mean(axis=1)
    ratio_norm = (K_mean / np.where(nonK_mean > 0, nonK_mean, 1.0))
    ratio_norm = ratio_norm / ratio_norm[0]

    def marks(ax, with_test=False):
        for x, c, lab in [(bif, "black", "bifurcation"), (com, "purple", "commit"),
                          (cliff, "red", "cliff")]:
            if x is not None:
                ax.axvline(x, color=c, ls="--", lw=1.0, alpha=0.7, label=f"{lab} ({x})")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # (a) per-character trajectories, K highlighted, + test loss overlay.
    ax = axes[0]
    for k in range(1, n_chars + 1):
        is_K = k in K
        ax.plot(epochs, char_E[:, k - 1],
                color=ORDER_COLOR.get(order_of(k, p), "#aaaaaa"),
                alpha=0.95 if is_K else 0.10, lw=1.6 if is_K else 0.4,
                label=f"k={k} (o={order_of(k, p)})" if is_K else None)
    marks(ax)
    ax.set_yscale("log"); ax.set_xlabel("step"); ax.set_ylabel("W_E energy in character k")
    ax.set_title("(a) per-character energy (K highlighted)")
    ax.legend(fontsize=7, loc="lower right"); ax.grid(True, alpha=0.3)
    if test_losses is not None:
        axt = ax.twinx()
        axt.plot(np.arange(len(test_losses)), test_losses, color="black", lw=1.0, alpha=0.4)
        axt.set_yscale("log"); axt.set_ylabel("test loss (black)", color="gray")

    # (b) K mean vs non-K mean.
    ax = axes[1]
    ax.plot(epochs, K_mean, color="#d62728", lw=1.8, label=f"mean K ({len(K)} chars)")
    ax.plot(epochs, nonK_mean, color="#1f77b4", lw=1.8, label=f"mean non-K ({len(nonK)} chars)")
    marks(ax)
    ax.set_yscale("log"); ax.set_xlabel("step"); ax.set_ylabel("mean energy per character")
    ax.set_title("(b) K mean vs non-K mean"); ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)

    # (c) K/non-K ratio normalized to init — the commitment signal.
    ax = axes[2]
    ax.plot(epochs, ratio_norm, color="#2ca02c", lw=1.8, label="K/non-K ratio (norm to t=0)")
    ax.axhline(1.5, color="red", ls=":", lw=0.9, alpha=0.6, label="bifurcation threshold (1.5x)")
    marks(ax)
    ax.set_yscale("log"); ax.set_xlabel("step"); ax.set_ylabel("ratio(t) / ratio(0)")
    ax.set_title("(c) basin commitment signal"); ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"Basin commitment ({args.tag}) — K = {K}", fontsize=12)
    fig.tight_layout()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else FIG_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"basin_commitment_{args.tag}.png"
    fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
