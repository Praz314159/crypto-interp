"""Cliff time vs commitment metrics, across all seeds.

Definitions:
  cliff_time         = first epoch at which test_loss < 0.1 (well below the
                       ~10 memorization plateau).
  bifurcation_step   = step at which K/non-K energy ratio first exceeds 1.5×
                       its init value (from fine-grained per-step data; see
                       analyze_bifurcation.py).
  ratio_at_1500      = mean(K energy) / mean(non-K energy) at epoch 1500
                       (from full-training trajectories.pkl).

Produces:
  figures/basis_dynamics/cliff_vs_bifurcation.png        scatter
  figures/basis_dynamics/cliff_prediction_from_ratio.png regression
  figures/basis_dynamics/test_loss_trajectories.png      all-seed loss curves
  figures/basis_dynamics/grokking_dashboard.png          combined 4-panel
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
RUNS = ROOT / "runs"
TRAJ_FILE = ROOT / "data" / "basis_dynamics" / "trajectories.pkl"
FG_DIR = ROOT / "data" / "fine_grained"
FIG_DIR = ROOT / "figures" / "basis_dynamics"


def order_of(k, n=112):
    return n // math.gcd(k, n)


def find_cliff(test_losses, threshold=0.1):
    arr = np.asarray(test_losses)
    above = np.where(arr < threshold)[0]
    return int(above[0]) if len(above) else None


def k_class(K):
    """Classify K by its dominant character order: mfe / primitive / pure_Z7 / mixed."""
    orders = [order_of(k) for k in K]
    if any(o == 112 for o in orders):
        return "primitive"
    if all(o == 7 or o == 14 or o == 56 for o in orders):
        return "mfe"
    if all(o in (2, 4, 8, 16) for o in orders):
        return "pure_Z16"
    if all(o == 7 for o in orders):
        return "pure_Z7"
    return "mixed"


def main():
    with open(TRAJ_FILE, "rb") as f:
        trajectories = pickle.load(f)

    # --- Per-seed data ---
    rows = []
    for seed in sorted(trajectories):
        traj = trajectories[seed]
        run_dir = RUNS / f"dmodel_24_seed{seed}"
        if not (run_dir / "losses.pt").exists():
            continue
        losses = torch.load(run_dir / "losses.pt", weights_only=False)
        test_losses = np.asarray(losses["test_losses"])
        train_losses = np.asarray(losses["train_losses"])
        cliff = find_cliff(test_losses)

        ce = traj["char_energy"]
        epochs = traj["epochs"]
        K = sorted([int(k + 1) for k, e in enumerate(ce[-1]) if e >= 0.05 * ce[-1].max()])
        nonK = [k for k in range(1, 57) if k not in K]

        # Ratio at epoch 1500 (use closest checkpoint).
        if 1500 in epochs:
            i1500 = int(np.where(epochs == 1500)[0][0])
        else:
            i1500 = int(np.argmin(np.abs(epochs - 1500)))
        if i1500 < len(ce):
            K_mean = ce[i1500, [k - 1 for k in K]].mean()
            nonK_mean = ce[i1500, [k - 1 for k in nonK]].mean() if nonK else 1
            ratio_1500 = K_mean / max(nonK_mean, 1e-12)
        else:
            ratio_1500 = None

        # Bifurcation step (from fine-grained, recompute on the fly).
        fg = FG_DIR / f"seed{seed:02d}_fine_grained.pt"
        bif = None
        if fg.exists():
            d = torch.load(fg, weights_only=False)
            # We need K/nonK energy per step. Recompute from W_E using the basis.
            from crypto_interp.interp.bases import multiplicative_fourier_basis
            basis, names, _ = multiplicative_fourier_basis(113)
            char_idx = {}
            for i, nm in enumerate(names):
                m = re.match(r"mul (cos|sin) (\d+)", nm)
                if m:
                    kk = int(m.group(2))
                    char_idx.setdefault(kk, []).append(i)
            W_E = d["W_E"]  # (T, d_model, vocab)
            W_v = W_E[:, :, :113].double()
            coef = torch.einsum("kp,tdp->tkd", basis.double(), W_v)
            E = (coef ** 2).sum(dim=2).cpu().numpy()
            char_E = np.zeros((W_v.shape[0], 56))
            for k_char, rrs in char_idx.items():
                char_E[:, k_char - 1] = E[:, rrs].sum(axis=1)
            K_mean = char_E[:, [k - 1 for k in K]].mean(axis=1)
            nonK_mean = char_E[:, [k - 1 for k in nonK]].mean(axis=1) if nonK else np.ones_like(K_mean)
            ratio_t = K_mean / np.where(nonK_mean > 0, nonK_mean, 1.0)
            ratio_norm = ratio_t / ratio_t[0]
            above = np.where(ratio_norm > 1.5)[0]
            bif = int(above[0]) if len(above) else None

        rows.append(dict(
            seed=seed, K=K, len_K=len(K), klass=k_class(K),
            cliff=cliff, ratio_1500=ratio_1500, bifurcation=bif,
            test_losses=test_losses, train_losses=train_losses,
            # Memorize-time (train_loss reaches < 0.1 for first time)
            memorize=find_cliff(train_losses, 0.1),
        ))

    print(f"Loaded {len(rows)} seeds")
    print(f"{'seed':>4} {'|K|':>3} {'class':>12} {'mem':>7} {'cliff':>7} "
          f"{'bif':>5} {'ratio@1500':>11}  K")
    for r in rows:
        print(f"{r['seed']:>4} {r['len_K']:>3} {r['klass']:>12} "
              f"{r['memorize']:>7} {r['cliff']:>7} "
              f"{str(r['bifurcation']):>5} "
              f"{r['ratio_1500']:>11.3f}  {r['K']}")

    # --- Test-loss trajectory plot ---
    klass_colors = {"primitive": "#d62728", "mfe": "#1f77b4",
                    "pure_Z7": "#2ca02c", "pure_Z16": "#9467bd",
                    "mixed": "#7f7f7f"}
    fig, ax = plt.subplots(figsize=(11, 5.5))
    seen = set()
    for r in rows:
        if r["cliff"] is None:
            continue
        col = klass_colors.get(r["klass"], "#7f7f7f")
        eps = np.arange(len(r["test_losses"]))
        label = r["klass"] if r["klass"] not in seen else None
        seen.add(r["klass"])
        ax.plot(eps, np.log10(r["test_losses"]),
                color=col, lw=0.9, alpha=0.7, label=label)
    ax.set_xscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("log10 test loss")
    ax.set_title(f"Test loss trajectory, all {len(rows)} seeds (color = K class)")
    ax.legend(fontsize=9, loc="lower left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = FIG_DIR / "test_loss_trajectories.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")

    # --- Cliff vs bifurcation step ---
    valid = [r for r in rows if r["cliff"] is not None and r["bifurcation"] is not None]
    fig, ax = plt.subplots(figsize=(7.5, 6))
    for r in valid:
        col = klass_colors.get(r["klass"], "#7f7f7f")
        ax.scatter(r["bifurcation"], r["cliff"], s=70, color=col, alpha=0.85,
                   edgecolor="black", linewidth=0.4)
        ax.annotate(str(r["seed"]), (r["bifurcation"], r["cliff"]),
                    xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.set_xlabel("bifurcation step (K/non-K ratio > 1.5× init)")
    ax.set_ylabel("cliff time (test loss < 0.1)")
    ax.set_yscale("log")
    ax.set_title(f"Cliff time vs bifurcation step ({len(valid)} seeds)")
    ax.grid(True, alpha=0.3)
    legend_handles = [plt.Line2D([0], [0], marker="o", color="w",
                                  markerfacecolor=c, markersize=8,
                                  markeredgecolor="black", label=lbl)
                       for lbl, c in klass_colors.items()]
    ax.legend(handles=legend_handles, fontsize=9, loc="upper right")
    fig.tight_layout()
    out = FIG_DIR / "cliff_vs_bifurcation.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")

    # --- Cliff prediction from ratio_at_1500 (regression) ---
    valid_r = [r for r in rows if r["cliff"] is not None and r["ratio_1500"] is not None
               and r["ratio_1500"] > 0]
    x = np.log(np.array([r["ratio_1500"] for r in valid_r]))
    y = np.log(np.array([r["cliff"] for r in valid_r]))
    slope, intercept = np.polyfit(x, y, 1)
    A = np.exp(intercept)
    # y = slope*x + intercept   =>   cliff = A * ratio^slope
    fig, ax = plt.subplots(figsize=(7.5, 6))
    for r in valid_r:
        col = klass_colors.get(r["klass"], "#7f7f7f")
        ax.scatter(r["ratio_1500"], r["cliff"], s=70, color=col, alpha=0.85,
                   edgecolor="black", linewidth=0.4)
        ax.annotate(str(r["seed"]), (r["ratio_1500"], r["cliff"]),
                    xytext=(4, 4), textcoords="offset points", fontsize=8)
    xs = np.linspace(x.min(), x.max(), 100)
    ax.plot(np.exp(xs), np.exp(intercept + slope * xs),
            "k--", lw=1.4,
            label=f"cliff ≈ {A:.0f} × ratio^{slope:.2f}")
    # Compute Spearman rho.
    from scipy.stats import spearmanr
    rho, _ = spearmanr([r["ratio_1500"] for r in valid_r],
                       [r["cliff"] for r in valid_r])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("K/non-K ratio at epoch 1500")
    ax.set_ylabel("cliff time (test loss < 0.1)")
    ax.set_title(f"Cliff prediction. n={len(valid_r)} seeds. "
                 f"Spearman ρ = {rho:.2f}")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = FIG_DIR / "cliff_prediction_from_ratio.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")
    print(f"Regression: cliff = {A:.1f} × ratio^{slope:.3f}   ρ={rho:.3f}")

    # --- Combined dashboard ---
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    # 1) test loss trajectories
    ax = axes[0, 0]
    seen = set()
    for r in rows:
        if r["cliff"] is None:
            continue
        col = klass_colors.get(r["klass"], "#7f7f7f")
        label = r["klass"] if r["klass"] not in seen else None
        seen.add(r["klass"])
        ax.plot(np.arange(len(r["test_losses"])),
                np.log10(r["test_losses"]),
                color=col, lw=0.9, alpha=0.7, label=label)
    ax.set_xscale("log"); ax.set_xlabel("epoch"); ax.set_ylabel("log10 test loss")
    ax.set_title("(a) test-loss trajectory by K class")
    ax.legend(fontsize=8, loc="lower left"); ax.grid(True, alpha=0.3)

    # 2) cliff vs bifurcation
    ax = axes[0, 1]
    for r in valid:
        col = klass_colors.get(r["klass"], "#7f7f7f")
        ax.scatter(r["bifurcation"], r["cliff"], s=70, color=col, alpha=0.85,
                   edgecolor="black", linewidth=0.4)
        ax.annotate(str(r["seed"]), (r["bifurcation"], r["cliff"]),
                    xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.set_yscale("log")
    ax.set_xlabel("bifurcation step (per-step W_E commitment)")
    ax.set_ylabel("cliff time")
    ax.set_title(f"(b) cliff vs bifurcation step ({len(valid)} seeds)")
    ax.grid(True, alpha=0.3)

    # 3) cliff vs ratio@1500
    ax = axes[1, 0]
    for r in valid_r:
        col = klass_colors.get(r["klass"], "#7f7f7f")
        ax.scatter(r["ratio_1500"], r["cliff"], s=70, color=col, alpha=0.85,
                   edgecolor="black", linewidth=0.4)
        ax.annotate(str(r["seed"]), (r["ratio_1500"], r["cliff"]),
                    xytext=(4, 4), textcoords="offset points", fontsize=8)
    xs = np.linspace(x.min(), x.max(), 100)
    ax.plot(np.exp(xs), np.exp(intercept + slope * xs), "k--", lw=1.4,
            label=f"cliff ≈ {A:.0f} × ratio^{slope:.2f}")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("K/non-K ratio at epoch 1500")
    ax.set_ylabel("cliff time")
    ax.set_title(f"(c) cliff vs K-ratio @ 1500. ρ={rho:.2f}")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # 4) memorize → cliff scatter
    ax = axes[1, 1]
    for r in rows:
        if r["memorize"] is None or r["cliff"] is None:
            continue
        col = klass_colors.get(r["klass"], "#7f7f7f")
        ax.scatter(r["memorize"], r["cliff"], s=70, color=col, alpha=0.85,
                   edgecolor="black", linewidth=0.4)
        ax.annotate(str(r["seed"]), (r["memorize"], r["cliff"]),
                    xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("memorize epoch (train_loss < 0.1)")
    ax.set_ylabel("cliff time")
    ax.set_title("(d) cliff time vs memorize epoch")
    ax.grid(True, alpha=0.3)
    # y=x reference
    lo = min([r["memorize"] for r in rows if r["memorize"]])
    hi = max([r["cliff"] for r in rows if r["cliff"]])
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.6, alpha=0.5)

    fig.suptitle(f"Grokking dashboard — {len(rows)} seeds at d_model=24, p=113", fontsize=13)
    fig.tight_layout()
    out = FIG_DIR / "grokking_dashboard.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
