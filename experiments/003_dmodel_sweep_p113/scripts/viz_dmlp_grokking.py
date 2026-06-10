"""Grokking trajectories for the d_mlp sweep runs.

Overlays test and train losses (log10) for all completed
``dmodel_24_dmlp_<M>_seed<S>`` runs, faceted by d_mlp.

Outputs:
  figures/basis_dynamics/dmlp_grokking_curves.png   — 2x2 dashboard
"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
FIG_DIR = ROOT / "figures" / "basis_dynamics"


def collect_runs():
    pat = re.compile(r"dmodel_24_dmlp_(\d+)_seed(\d+)$")
    out = {}
    for d in sorted(RUNS.iterdir()):
        m = pat.match(d.name)
        if not m or not d.is_dir():
            continue
        if not (d / "losses.pt").exists():
            continue
        d_mlp = int(m.group(1))
        seed = int(m.group(2))
        losses = torch.load(d / "losses.pt", weights_only=False)
        out.setdefault(d_mlp, []).append((seed, losses))
    for d in out:
        out[d].sort()
    return out


def find_cliff(test_losses, thresh=0.1):
    arr = np.asarray(test_losses)
    above = np.where(arr < thresh)[0]
    return int(above[0]) if len(above) else None


def main():
    runs = collect_runs()
    dmlps = sorted(runs)
    print(f"Found runs at d_mlp ∈ {dmlps}")
    for d in dmlps:
        for s, losses in runs[d]:
            tr = np.asarray(losses["train_losses"])
            te = np.asarray(losses["test_losses"])
            cliff = find_cliff(te)
            mem = find_cliff(tr)
            print(f"  d_mlp={d:>3} seed={s}: "
                  f"memorize @ {mem}, cliff @ {cliff}, "
                  f"final train={tr[-1]:.2e}, final test={te[-1]:.2e}")

    # Color scheme: one color per d_mlp, line style per seed.
    cmap = {64: "#d62728", 128: "#ff7f0e", 256: "#9467bd", 512: "#1f77b4"}
    linestyles = {1: "-", 2: "--", 3: ":"}

    # ---- combined train+test overlays ----
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), sharex=True, sharey=True)
    for ax, key, title in [
        (axes[0], "train_losses", "Train loss"),
        (axes[1], "test_losses", "Test loss"),
    ]:
        for d in dmlps:
            for seed, losses in runs[d]:
                vals = np.asarray(losses[key])
                ax.plot(np.arange(len(vals)), np.log10(vals + 1e-15),
                        color=cmap.get(d, "#7f7f7f"),
                        ls=linestyles.get(seed, "-"),
                        lw=1.2, alpha=0.85,
                        label=f"d_mlp={d} seed={seed}")
        ax.set_xlabel("epoch")
        ax.set_ylabel("log10 loss")
        ax.set_title(title)
        ax.set_xscale("log")
        ax.grid(True, alpha=0.3)
    axes[1].legend(fontsize=8, loc="upper right", ncol=2)
    fig.suptitle(f"d_mlp sweep — grokking trajectories ({sum(len(v) for v in runs.values())} runs)",
                 fontsize=12)
    fig.tight_layout()
    out1 = FIG_DIR / "dmlp_grokking_curves.png"
    fig.savefig(out1, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out1}")

    # ---- per-d_mlp facet ----
    fig, axes = plt.subplots(1, len(dmlps), figsize=(5 * len(dmlps), 5),
                              sharey=True)
    if len(dmlps) == 1:
        axes = [axes]
    for ax, d in zip(axes, dmlps):
        for seed, losses in runs[d]:
            tr = np.asarray(losses["train_losses"])
            te = np.asarray(losses["test_losses"])
            ax.plot(np.arange(len(tr)), np.log10(tr + 1e-15),
                    color="#1f77b4", ls=linestyles.get(seed, "-"),
                    lw=1.0, alpha=0.6, label=f"train seed={seed}" if seed == 1 else None)
            ax.plot(np.arange(len(te)), np.log10(te + 1e-15),
                    color="#d62728", ls=linestyles.get(seed, "-"),
                    lw=1.4, alpha=0.85,
                    label=f"test seed={seed}")
            cliff = find_cliff(te)
            if cliff is not None:
                ax.axvline(cliff, color="black", ls=":", lw=0.6, alpha=0.4)
                ax.annotate(f"{cliff}", (cliff, 0.5),
                            textcoords="offset points", xytext=(2, 0),
                            fontsize=7, rotation=90)
        ax.set_xscale("log")
        ax.set_xlabel("epoch")
        ax.set_title(f"d_mlp = {d}")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("log10 loss (red=test, blue=train)")
    fig.suptitle("Grokking trajectories by d_mlp (per-seed dashed/dotted)", fontsize=12)
    fig.tight_layout()
    out2 = FIG_DIR / "dmlp_grokking_facets.png"
    fig.savefig(out2, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out2}")


if __name__ == "__main__":
    main()
