"""Cliff time vs commitment metrics, across all seeds.

  cliff_time       = first epoch with test_loss < 0.1.
  bifurcation_step = step where K/non-K energy ratio first exceeds 1.5x init.
  ratio_at_1500    = mean(K)/mean(non-K) energy at epoch 1500.

Uses crypto_interp.interp (prime inferred from the trajectory width).

Produces:
  figures/basis_dynamics/{cliff_vs_bifurcation,cliff_prediction_from_ratio,
                          test_loss_trajectories,grokking_dashboard}.png
"""
from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from crypto_interp.interp import bifurcation_step, char_energy_batch, char_index, find_cliff, order_of

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
TRAJ_FILE = ROOT / "data" / "basis_dynamics" / "trajectories.pkl"
FG_DIR = ROOT / "data" / "fine_grained"
FIG_DIR = ROOT / "figures" / "basis_dynamics"

KLASS_COLORS = {"primitive": "#d62728", "odd": "#2ca02c",
                "two_power": "#9467bd", "mixed": "#7f7f7f"}


def k_class(K, p):
    """Coarse, prime-parametric label of K by its character orders."""
    n = p - 1
    orders = [order_of(k, p) for k in K]
    if any(o == n for o in orders):
        return "primitive"
    if all(o % 2 == 1 for o in orders):
        return "odd"
    if all((o & (o - 1)) == 0 for o in orders):
        return "two_power"
    return "mixed"


def main():
    with open(TRAJ_FILE, "rb") as f:
        trajectories = pickle.load(f)

    L = len(next(iter(trajectories.values()))["char_energy"][-1])
    p = 2 * L + 1                       # n = p-1 even => n_chars = n//2 => p = 2L+1
    basis, ci = char_index(p)

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
        K = sorted(int(k + 1) for k, e in enumerate(ce[-1]) if e >= 0.05 * ce[-1].max())
        nonK = [k for k in range(1, L + 1) if k not in K]

        i1500 = int(np.where(epochs == 1500)[0][0]) if 1500 in epochs else int(np.argmin(np.abs(epochs - 1500)))
        ratio_1500 = None
        if i1500 < len(ce):
            K_mean = ce[i1500, [k - 1 for k in K]].mean()
            nonK_mean = ce[i1500, [k - 1 for k in nonK]].mean() if nonK else 1.0
            ratio_1500 = K_mean / max(nonK_mean, 1e-12)

        # Bifurcation step from fine-grained per-step W_E.
        fg = FG_DIR / f"seed{seed:02d}_fine_grained.pt"
        bif = None
        if fg.exists():
            d = torch.load(fg, weights_only=False)
            char_E = char_energy_batch(d["W_E"][:, :, :p], basis, ci)
            K_mask = np.zeros(char_E.shape[1], dtype=bool)
            for k in K:
                K_mask[k - 1] = True
            bif = bifurcation_step(char_E, K_mask, ratio=1.5)

        rows.append(dict(seed=seed, K=K, len_K=len(K), klass=k_class(K, p),
                         cliff=cliff, ratio_1500=ratio_1500, bifurcation=bif,
                         test_losses=test_losses, train_losses=train_losses,
                         memorize=find_cliff(train_losses, 0.1)))

    print(f"Loaded {len(rows)} seeds")
    print(f"{'seed':>4} {'|K|':>3} {'class':>10} {'mem':>7} {'cliff':>7} {'bif':>5} {'ratio@1500':>11}  K")
    for r in rows:
        print(f"{r['seed']:>4} {r['len_K']:>3} {r['klass']:>10} {str(r['memorize']):>7} "
              f"{str(r['cliff']):>7} {str(r['bifurcation']):>5} "
              f"{(r['ratio_1500'] if r['ratio_1500'] is not None else float('nan')):>11.3f}  {r['K']}")

    # --- test-loss trajectories ---
    fig, ax = plt.subplots(figsize=(11, 5.5))
    seen = set()
    for r in rows:
        if r["cliff"] is None:
            continue
        col = KLASS_COLORS.get(r["klass"], "#7f7f7f")
        label = r["klass"] if r["klass"] not in seen else None
        seen.add(r["klass"])
        ax.plot(np.arange(len(r["test_losses"])), np.log10(r["test_losses"]),
                color=col, lw=0.9, alpha=0.7, label=label)
    ax.set_xscale("log"); ax.set_xlabel("epoch"); ax.set_ylabel("log10 test loss")
    ax.set_title(f"Test loss trajectory, all {len(rows)} seeds (color = K class)")
    ax.legend(fontsize=9, loc="lower left"); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "test_loss_trajectories.png", dpi=130, bbox_inches="tight"); plt.close(fig)

    # --- cliff vs bifurcation ---
    valid = [r for r in rows if r["cliff"] is not None and r["bifurcation"] is not None]
    fig, ax = plt.subplots(figsize=(7.5, 6))
    for r in valid:
        ax.scatter(r["bifurcation"], r["cliff"], s=70, color=KLASS_COLORS.get(r["klass"], "#7f7f7f"),
                   alpha=0.85, edgecolor="black", linewidth=0.4)
        ax.annotate(str(r["seed"]), (r["bifurcation"], r["cliff"]), xytext=(4, 4),
                    textcoords="offset points", fontsize=8)
    ax.set_xlabel("bifurcation step (K/non-K ratio > 1.5x init)"); ax.set_ylabel("cliff time")
    ax.set_yscale("log"); ax.set_title(f"Cliff time vs bifurcation step ({len(valid)} seeds)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "cliff_vs_bifurcation.png", dpi=130, bbox_inches="tight"); plt.close(fig)

    # --- cliff prediction from ratio@1500 ---
    valid_r = [r for r in rows if r["cliff"] and r["ratio_1500"] and r["ratio_1500"] > 0]
    x = np.log(np.array([r["ratio_1500"] for r in valid_r]))
    y = np.log(np.array([r["cliff"] for r in valid_r]))
    slope, intercept = np.polyfit(x, y, 1)
    A = np.exp(intercept)
    fig, ax = plt.subplots(figsize=(7.5, 6))
    for r in valid_r:
        ax.scatter(r["ratio_1500"], r["cliff"], s=70, color=KLASS_COLORS.get(r["klass"], "#7f7f7f"),
                   alpha=0.85, edgecolor="black", linewidth=0.4)
    xs = np.linspace(x.min(), x.max(), 100)
    ax.plot(np.exp(xs), np.exp(intercept + slope * xs), "k--", lw=1.4,
            label=f"cliff ~ {A:.0f} x ratio^{slope:.2f}")
    # Spearman via rank correlation (avoid scipy dependency).
    rr = np.array([r["ratio_1500"] for r in valid_r]); cc = np.array([r["cliff"] for r in valid_r])
    rho = float(np.corrcoef(np.argsort(np.argsort(rr)), np.argsort(np.argsort(cc)))[0, 1])
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("K/non-K ratio at epoch 1500"); ax.set_ylabel("cliff time")
    ax.set_title(f"Cliff prediction. n={len(valid_r)} seeds. Spearman rho = {rho:.2f}")
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "cliff_prediction_from_ratio.png", dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"\nRegression: cliff = {A:.1f} x ratio^{slope:.3f}   rho={rho:.3f}")


if __name__ == "__main__":
    main()
