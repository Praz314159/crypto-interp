"""For each Δ_k(Δlog) 1D reduction, compute the Fourier spectrum: find which
frequency ablating character k actually destroys. If ablating k=20 peaks at
frequency m=10, that is the harmonic-helper signature (k=20 = 2·10 boosts the
primary χ_10 via ReLU squaring).

Uses crypto_interp.interp.harmonic (prime-parametric).

Usage:
    python experiments/003_dmodel_sweep_p113/scripts/analyze_delta_spectrum.py \
        --run-dir experiments/003_dmodel_sweep_p113/runs/dmodel_24_dmlp_32_seed1 \
        --ks 3,10,20,51,6
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from crypto_interp.interp import (
    char_index,
    compute_logits_grid,
    delta_k_spectrum,
    load_run,
    order_of,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--ks", required=True)
    args = ap.parse_args()
    run_dir = Path(args.run_dir).resolve()
    ck = sorted(run_dir.glob("checkpoint_*.pt"))
    model, ds, _ = load_run(ck[-1])
    model.eval()
    p = ds.p
    basis, ci = char_index(p)
    logits_full = compute_logits_grid(model, ds)

    ks = [int(x) for x in args.ks.split(",")]
    fig, axes = plt.subplots(len(ks), 1, figsize=(13, 2.2 * len(ks)), sharex=True)
    if len(ks) == 1:
        axes = [axes]

    N = p - 1
    for i, k in enumerate(ks):
        _, energy, dominant = delta_k_spectrum(model, ds, ci, basis, k, logits_full=logits_full)
        top = sorted(range(len(energy)), key=lambda j: -energy[j])[:5]
        top_strs = [f"m={top[j]+1} E={energy[top[j]]:.2e}" for j in range(5)]
        print(f"ablating k={k:>3} (order {order_of(k, p)}):  dominant m={dominant}; "
              + "; ".join(top_strs))

        ax = axes[i]
        ax.bar(np.arange(1, N // 2 + 1), energy, color="#1f77b4", alpha=0.85)
        if k <= N // 2:
            ax.axvline(k, color="red", ls="--", lw=1.0, alpha=0.7, label=f"naive k={k}")
        if 2 * k <= N // 2:
            ax.axvline(2 * k, color="orange", ls="--", lw=1.0, alpha=0.7, label=f"2k={2*k}")
        if N - k <= N // 2:
            ax.axvline(N - k, color="green", ls="--", lw=1.0, alpha=0.7, label=f"-k={N-k}")
        ax.set_ylabel(f"ablate k={k}")
        ax.set_yscale("log")
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.3, axis="y")

    axes[-1].set_xlabel("Fourier frequency m of Δ_k(Δlog)")
    fig.suptitle("Spectrum of Δ_k(Δlog): what frequency does ablating each k destroy?",
                 fontsize=11)
    fig.tight_layout()
    out = run_dir / "delta_spectrum.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
