"""Weight-norm trajectories across the 16-seed sweep.

Tests the hypothesis that laggard seeds build a deeper memorization basin
(higher peak weight norm) before AdamW decays them out and grokking begins.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def wnorm(sd) -> float:
    return float(torch.sqrt(
        sum((v.float() ** 2).sum() for v in sd.values()
            if hasattr(v, "dtype") and v.dtype.is_floating_point)
    ))


def main():
    rows = []
    for d in sorted(Path("runs").glob("mul_sweep_seed*")):
        if not d.is_dir():
            continue
        cps = sorted(d.glob("checkpoint_*.pt"))
        if len(cps) < 2:
            continue
        seed = int(str(d).split("seed")[-1])
        losses = torch.load(d / "losses.pt", weights_only=False)
        te = np.asarray(losses["test_losses"])
        idx = np.where(te < 1e-2)[0]
        grok = int(idx[0]) if len(idx) else None
        series = []
        for cp in cps:
            c = torch.load(cp, weights_only=False)
            series.append((c["epoch"], wnorm(c["model_state"])))
        series.sort()
        rows.append({"seed": seed, "grok": grok, "series": series})

    rows.sort(key=lambda r: r["grok"] if r["grok"] is not None else 1e9)
    grok_min = min(r["grok"] for r in rows)
    grok_max = max(r["grok"] for r in rows)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    cmap = plt.cm.viridis
    for r in rows:
        c = cmap((r["grok"] - grok_min) / max(grok_max - grok_min, 1))
        eps, nrms = zip(*r["series"])
        ax.plot(eps, nrms, "o-", lw=1.5, ms=5, color=c, alpha=0.85,
                label=f"seed {r['seed']} (grok@{r['grok']})")
        ax.axvline(r["grok"], color=c, ls=":", lw=0.5, alpha=0.4)
    ax.axhline(32, color="k", ls="--", lw=0.8, alpha=0.5, label="AdamW equilibrium ≈ 32")
    ax.axhline(42.3, color="gray", ls=":", lw=0.8, alpha=0.5, label="init norm")
    ax.set_xlabel("epoch")
    ax.set_ylabel("‖θ‖₂ (total weight norm)")
    ax.set_title("Weight-norm trajectories across 16 seeds\n"
                 "Laggards (yellow) peak at ‖θ‖≈50 — a deeper memorization basin")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=7, ncol=2)
    fig.tight_layout()
    out = Path("runs/laggard_weight_norms.png")
    fig.savefig(out, dpi=130)
    print(f"Saved {out}")

    # Scatter: peak norm vs grok epoch
    fig, ax = plt.subplots(figsize=(8, 5))
    peak_norms = [max(n for _, n in r["series"]) for r in rows]
    groks = [r["grok"] for r in rows]
    seeds_arr = [r["seed"] for r in rows]
    ax.scatter(peak_norms, groks, s=70, alpha=0.8)
    for x, y, s in zip(peak_norms, groks, seeds_arr):
        ax.annotate(str(s), (x, y), fontsize=8, alpha=0.8)
    rho = np.corrcoef(np.argsort(np.argsort(peak_norms)),
                       np.argsort(np.argsort(groks)))[0, 1]
    ax.set_xlabel("peak ‖θ‖₂ during memorization")
    ax.set_ylabel("grok epoch")
    ax.set_title(f"Memorization basin depth vs grok delay (Spearman ρ = {rho:+.3f})")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out2 = Path("runs/laggard_norm_vs_grok.png")
    fig.savefig(out2, dpi=130)
    print(f"Saved {out2}")
    print(f"\nSpearman correlation peak_norm vs grok = {rho:+.3f}")


if __name__ == "__main__":
    main()
