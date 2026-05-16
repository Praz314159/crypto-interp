"""Mechanism analysis for laggard seeds.

Computes two progress measures across training and compares across seeds:

  1. Concentration C(t) := frac of W_E character energy on the *final* top-5
     freqs. Starts ~5/56 ≈ 0.09 (random) and rises monotonically to ~0.85.
     The slope of this curve is the rate of Fourier-circuit formation.

  2. Initial overlap O(0) := same fraction computed at epoch 0. Tells us
     whether laggards started with an unusually weak overlap on their
     eventual top-5 (a seed-init effect).
"""

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def grok_epoch(losses_path: Path, tau: float = 1e-2):
    l = torch.load(losses_path, weights_only=False)
    te = np.asarray(l["test_losses"])
    idx = np.where(te < tau)[0]
    return (int(idx[0]) if len(idx) else None), te


def main():
    sweep = []
    for d in sorted(Path("runs").glob("mul_sweep_seed*")):
        if not d.is_dir():
            continue
        m = torch.load(d / "metrics.pt", weights_only=False)
        seed = m["config"]["seed"]
        gk, te = grok_epoch(d / "losses.pt")
        fe = np.asarray(m["freq_energies"])
        ep = np.asarray(m["epochs"])
        final = fe[-1]
        top5 = np.argsort(-final)[:5]
        frac = fe[:, top5].sum(axis=1) / fe.sum(axis=1)
        sweep.append({"seed": seed, "grok": gk, "epochs": ep, "frac": frac,
                       "init_frac": float(frac[0]), "te": te})
    sweep.sort(key=lambda s: s["grok"] if s["grok"] is not None else 1e9)

    grok_min = min(s["grok"] for s in sweep)
    grok_max = max(s["grok"] for s in sweep)
    cmap = plt.cm.viridis

    # Plot 1: trajectories colored by grok epoch
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for s in sweep:
        c = cmap((s["grok"] - grok_min) / max(grok_max - grok_min, 1))
        ax.plot(s["epochs"], s["frac"], lw=1.5, alpha=0.85, color=c,
                label=f"seed {s['seed']} (grok@{s['grok']})")
    ax.set_xlabel("epoch")
    ax.set_ylabel("frac of W_E character energy on final top-5 freqs")
    ax.set_title("Fourier-circuit formation across 16 seeds\n(color = grok epoch; laggards in yellow)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=7, ncol=2)
    fig.tight_layout()
    p1 = Path("runs/laggard_concentration.png")
    fig.savefig(p1, dpi=130)
    print(f"Saved {p1}")

    # Plot 2: scatter of init frac vs grok epoch
    fig, ax = plt.subplots(figsize=(8, 5))
    xs = [s["init_frac"] for s in sweep]
    ys = [s["grok"] for s in sweep]
    seeds_arr = [s["seed"] for s in sweep]
    ax.scatter(xs, ys, s=70, alpha=0.8)
    for x, y, sd in zip(xs, ys, seeds_arr):
        ax.annotate(str(sd), (x, y), fontsize=8, alpha=0.8)
    # null baseline: 5/56 ≈ 0.0893
    ax.axvline(5/56, color="k", ls="--", lw=0.8, label="uniform 5/56 ≈ 0.089")
    ax.set_xlabel("initial frac of W_E energy on (eventual) top-5 freqs")
    ax.set_ylabel("grok epoch")
    ax.set_title("Did laggards start with a weaker init overlap on their eventual basis?")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    p2 = Path("runs/laggard_init_overlap.png")
    fig.savefig(p2, dpi=130)
    print(f"Saved {p2}")

    # Plot 3: time to reach concentration thresholds, vs grok epoch
    fig, ax = plt.subplots(figsize=(11, 5))
    thresholds = [0.2, 0.4, 0.6]
    for thresh, color in zip(thresholds, ["tab:green", "tab:orange", "tab:red"]):
        xs, ys = [], []
        for s in sweep:
            crossing = np.where(s["frac"] >= thresh)[0]
            if len(crossing):
                xs.append(s["grok"])
                ys.append(s["epochs"][crossing[0]])
        ax.scatter(xs, ys, s=60, alpha=0.7, color=color, label=f"first epoch where C(t) ≥ {thresh}")
    lim = max(grok_max + 500, max(max(ys, default=0) for _ in [None]) if True else 0)
    ax.plot([0, lim], [0, lim], "k:", lw=0.7, alpha=0.5, label="y = grok epoch")
    ax.set_xlabel("grok epoch")
    ax.set_ylabel("first epoch where concentration crosses threshold")
    ax.set_title("Concentration-threshold crossings vs grokking — does C(t) cross long before grok?")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    p3 = Path("runs/laggard_threshold_crossings.png")
    fig.savefig(p3, dpi=130)
    print(f"Saved {p3}")

    # Print summary
    print(f"\n{'seed':>4} {'grok':>6} {'init frac':>10} {'eps to .2':>10} {'eps to .4':>10} {'eps to .6':>10}")
    for s in sweep:
        line = f"{s['seed']:4d} {s['grok']:6d} {s['init_frac']:10.4f}"
        for t in thresholds:
            cr = np.where(s["frac"] >= t)[0]
            line += f"  {int(s['epochs'][cr[0]]) if len(cr) else -1:8d}"
        print(line)

    # Correlations
    print("\nSpearman correlations with grok epoch:")
    grokA = np.array([s["grok"] for s in sweep], dtype=float)
    inits = np.array([s["init_frac"] for s in sweep])
    xr = np.argsort(np.argsort(inits))
    yr = np.argsort(np.argsort(grokA))
    print(f"  init_frac     : ρ = {np.corrcoef(xr, yr)[0,1]:+.3f}")


if __name__ == "__main__":
    main()
