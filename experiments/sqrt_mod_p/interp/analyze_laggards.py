"""Why are some seeds slow to grok?

Loads the multi-seed sweep and produces:
  1. Per-seed grokking epoch (first epoch where test loss < tau).
  2. A 4x4 grid of energy(freq, epoch) trajectories — one panel per seed —
     showing the top-5 final key frequencies, so we can spot false starts.
  3. A comparison table: seed x {grok epoch, min key-freq order, #sig key
     freqs, max pre-grok energy of frequencies that were NOT in the final
     top-5 ("false-start energy")}.
"""

import argparse
import math
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def order_in_Zn(k: int, n: int) -> int:
    return n // math.gcd(k, n)


def freq_index_to_k(i: int, n: int) -> int:
    # freq i (0-indexed) corresponds to k = i+1 (1-indexed multiplicative chars)
    return i + 1


def load_sweep(glob: str):
    out = []
    for d in sorted(Path(".").glob(glob)):
        mp = d / "metrics.pt"
        lp = d / "losses.pt"
        if not mp.exists() or not lp.exists():
            continue
        m = torch.load(mp, weights_only=False)
        ls = torch.load(lp, weights_only=False)
        m["run_dir"] = d
        m["losses"] = ls  # {'train': [...], 'test': [...], 'epochs': [...]} or similar
        out.append(m)
    return out


def grokking_epoch(losses: dict, tau: float = 1e-2):
    """First epoch where test loss < tau."""
    test = np.asarray(losses["test_losses"], dtype=float)
    epochs = np.arange(len(test))
    below = np.where(test < tau)[0]
    if len(below) == 0:
        return None, test, epochs
    return int(epochs[below[0]]), test, epochs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="runs/mul_sweep_seed*")
    ap.add_argument("--tau", type=float, default=1e-2, help="test-loss threshold for grok")
    ap.add_argument("--out-dir", default="runs")
    ap.add_argument("--top-K", type=int, default=5)
    args = ap.parse_args()

    sweep = load_sweep(args.glob)
    sweep.sort(key=lambda m: m["config"]["seed"])
    p = sweep[0]["config"]["p"]
    n = p - 1
    n_seeds = len(sweep)
    print(f"Loaded {n_seeds} seeds, p={p}, n={n}, tau={args.tau}")

    rows = []
    per_seed = []
    for m in sweep:
        seed = m["config"]["seed"]
        losses = m["losses"]
        grok, test_curve, loss_epochs = grokking_epoch(losses, tau=args.tau)

        epochs = list(m["epochs"])
        energies = np.asarray(m["freq_energies"])  # (T, F)
        final = energies[-1]
        ranked = np.argsort(-final)
        topk = ranked[: args.top_K].tolist()
        topk_freqs = [freq_index_to_k(i, n) for i in topk]
        topk_orders = [order_in_Zn(k, n) for k in topk_freqs]
        min_order = min(topk_orders)
        max_e = float(final.max())
        n_sig = int((final[topk] > 0.1 * max_e).sum())

        # False-start energy: max energy across training of any freq NOT in topk,
        # measured pre-grok.
        if grok is not None:
            pre_grok_mask = np.asarray(epochs) <= grok
        else:
            pre_grok_mask = np.ones(len(epochs), dtype=bool)
        non_top_idx = [i for i in range(energies.shape[1]) if i not in topk]
        if len(non_top_idx) > 0 and pre_grok_mask.any():
            sub = energies[np.ix_(pre_grok_mask, non_top_idx)]
            false_start = float(sub.max())
            # which freq carried that energy?
            local_argmax = int(np.unravel_index(np.argmax(sub), sub.shape)[1])
            false_start_k = freq_index_to_k(non_top_idx[local_argmax], n)
            false_start_ratio = false_start / max_e
        else:
            false_start = float("nan")
            false_start_k = -1
            false_start_ratio = float("nan")

        per_seed.append({
            "seed": seed, "grok": grok, "test": test_curve, "loss_epochs": loss_epochs,
            "epochs": epochs, "energies": energies, "topk": topk, "topk_freqs": topk_freqs,
            "topk_orders": topk_orders,
        })
        rows.append((seed, grok, min_order, n_sig, false_start_ratio, false_start_k, topk_freqs, topk_orders))

    # ---- Table ----
    rows.sort(key=lambda r: (r[1] is None, r[1] or 0))
    print(f"\n{'seed':>4} {'grok@':>6} {'min ord':>7} {'#sig':>4} {'false/max':>9} {'fs_k':>5}  top-5 freqs (orders)")
    for r in rows:
        seed, grok, min_o, ns, fsr, fsk, kfs, kos = r
        kfo = ", ".join(f"{k}({o})" for k, o in zip(kfs, kos))
        gstr = f"{grok}" if grok is not None else "  --"
        print(f"{seed:4d} {gstr:>6} {min_o:7d} {ns:4d} {fsr:9.3f} {fsk:5d}  {kfo}")

    # ---- Plot 1: 4x4 grid of energy trajectories ----
    n_cols = 4
    n_rows = int(np.ceil(n_seeds / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), sharex=False)
    axes = axes.flatten()
    grok_by_seed = {r[0]: r[1] for r in rows}
    for ax, s in zip(axes, per_seed):
        ep = np.asarray(s["epochs"])
        for idx, kf, ko in zip(s["topk"], s["topk_freqs"], s["topk_orders"]):
            color = "tab:red" if ko == 2 else ("tab:orange" if ko <= 8 else "tab:blue")
            ax.plot(ep, s["energies"][:, idx], lw=1.4, alpha=0.85, color=color,
                    label=f"k={kf} (ord {ko})")
        gk = grok_by_seed[s["seed"]]
        if gk is not None:
            ax.axvline(gk, color="k", ls="--", lw=0.8, alpha=0.6)
        ax.set_title(f"seed {s['seed']}" + (f"  grok@{gk}" if gk else "  (no grok)"), fontsize=10)
        ax.set_yscale("log")
        ax.grid(True, alpha=0.25)
        ax.tick_params(labelsize=8)
        ax.legend(fontsize=6, loc="lower right")
    for ax in axes[n_seeds:]:
        ax.axis("off")
    fig.suptitle("Per-seed energy trajectories of final top-5 frequencies\n"
                 "red=ord 2, orange=ord ≤8, blue=other; dashed=grok onset", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = Path(args.out_dir) / "laggard_energy_grid.png"
    fig.savefig(out, dpi=130)
    print(f"\nSaved {out}")

    # ---- Plot 2: test-loss curves overlaid, laggards highlighted ----
    fig, ax = plt.subplots(figsize=(11, 5))
    grok_epochs = [r[1] for r in rows if r[1] is not None]
    median_grok = float(np.median(grok_epochs)) if grok_epochs else float("inf")
    for s in per_seed:
        gk = grok_by_seed[s["seed"]]
        is_lag = (gk is None) or (gk > 2 * median_grok)
        ax.plot(s["loss_epochs"], s["test"], lw=1.2,
                color=("tab:red" if is_lag else "tab:gray"),
                alpha=(0.95 if is_lag else 0.45),
                label=f"seed {s['seed']}" if is_lag else None)
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("test loss")
    ax.set_title(f"Test loss across {n_seeds} seeds (laggards = grok > 2×median = {2*median_grok:.0f}; red)")
    ax.axhline(args.tau, color="k", ls=":", lw=0.8, label=f"τ={args.tau}")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out2 = Path(args.out_dir) / "laggard_loss_overlay.png"
    fig.savefig(out2, dpi=130)
    print(f"Saved {out2}")

    # ---- Plot 3: scatter of grok epoch vs (min key order, #sig, false-start ratio) ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    grokA = np.array([r[1] if r[1] is not None else np.nan for r in rows], dtype=float)
    min_o = np.array([r[2] for r in rows])
    n_sig = np.array([r[3] for r in rows])
    fs_r = np.array([r[4] for r in rows])
    seeds_arr = [r[0] for r in rows]

    axes[0].scatter(min_o, grokA, s=50, alpha=0.8)
    for x, y, s in zip(min_o, grokA, seeds_arr):
        if not np.isnan(y):
            axes[0].annotate(str(s), (x, y), fontsize=7, alpha=0.7)
    axes[0].set_xlabel("min order among top-5 key freqs")
    axes[0].set_ylabel("grokking epoch")
    axes[0].set_xscale("log")
    axes[0].set_title("Grok epoch vs min character order")
    axes[0].grid(True, alpha=0.3)

    axes[1].scatter(n_sig, grokA, s=50, alpha=0.8)
    for x, y, s in zip(n_sig, grokA, seeds_arr):
        if not np.isnan(y):
            axes[1].annotate(str(s), (x, y), fontsize=7, alpha=0.7)
    axes[1].set_xlabel("# significant key freqs (≥0.1×max)")
    axes[1].set_ylabel("grokking epoch")
    axes[1].set_title("Grok epoch vs # significant freqs")
    axes[1].grid(True, alpha=0.3)

    axes[2].scatter(fs_r, grokA, s=50, alpha=0.8)
    for x, y, s in zip(fs_r, grokA, seeds_arr):
        if not np.isnan(y):
            axes[2].annotate(str(s), (x, y), fontsize=7, alpha=0.7)
    axes[2].set_xlabel("false-start energy ratio (pre-grok max-non-key / final max)")
    axes[2].set_ylabel("grokking epoch")
    axes[2].set_title("Grok epoch vs false-start strength")
    axes[2].grid(True, alpha=0.3)
    fig.tight_layout()
    out3 = Path(args.out_dir) / "laggard_correlations.png"
    fig.savefig(out3, dpi=130)
    print(f"Saved {out3}")

    # Correlations
    mask = ~np.isnan(grokA)
    if mask.sum() >= 3:
        print("\nSpearman-ish correlations with grok epoch:")
        for name, x in [("min_order", min_o), ("n_sig", n_sig), ("false_start_ratio", fs_r)]:
            xm = x[mask].astype(float)
            ym = grokA[mask]
            xr = np.argsort(np.argsort(xm))
            yr = np.argsort(np.argsort(ym))
            if np.std(xr) > 0 and np.std(yr) > 0:
                rho = float(np.corrcoef(xr, yr)[0, 1])
                print(f"  {name:>20s} : ρ = {rho:+.3f}")


if __name__ == "__main__":
    main()
