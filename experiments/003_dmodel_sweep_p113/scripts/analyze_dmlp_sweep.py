"""Analyze the (d_mlp × seed) sweep runs.

For each run matching ``dmodel_24_dmlp_<M>_seed<S>``, identify K from the final
checkpoint's W_E (using the multiplicative-Fourier basis), report d_mlp_cost
and primitive count, and compare cliff time.

Outputs:
  data/basis_dynamics/dmlp_sweep_summary.csv
  figures/basis_dynamics/dmlp_K_size_vs_budget.png
  figures/basis_dynamics/dmlp_cost_vs_budget.png
  figures/basis_dynamics/dmlp_primitive_use.png
"""
from __future__ import annotations

import csv
import math
import pickle
import re
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from crypto_interp.interp.bases import multiplicative_fourier_basis

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
FIG_DIR = ROOT / "figures" / "basis_dynamics"
OUT_DIR = ROOT / "data" / "basis_dynamics"


def order_of(k, n=112):
    return n // math.gcd(k, n)


def build_char_basis():
    basis, names, _ = multiplicative_fourier_basis(113)
    char_idx = {}
    for i, nm in enumerate(names):
        m = re.match(r"mul (cos|sin) (\d+)", nm)
        if m:
            kk = int(m.group(2))
            char_idx.setdefault(kk, []).append(i)
    return basis.double(), char_idx


def char_energy_WE(W_E_dp, basis, char_idx, n_chars=56):
    coef = torch.einsum("kp,dp->kd", basis, W_E_dp.double())
    E = (coef ** 2).sum(dim=1).cpu().numpy()
    out = np.zeros(n_chars)
    for k, rows in char_idx.items():
        out[k - 1] = float(E[rows].sum())
    return out


def find_cliff(test_losses, thresh=0.1):
    arr = np.asarray(test_losses)
    above = np.where(arr < thresh)[0]
    return int(above[0]) if len(above) else None


def main():
    basis, char_idx = build_char_basis()
    # Discover all (d_mlp, seed) runs.
    pat = re.compile(r"dmodel_24_dmlp_(\d+)_seed(\d+)$")
    runs = []
    for d in sorted(RUNS.iterdir()):
        m = pat.match(d.name)
        if not m or not d.is_dir():
            continue
        d_mlp = int(m.group(1))
        seed = int(m.group(2))
        ck = sorted(d.glob("checkpoint_*.pt"))
        if not ck:
            continue
        runs.append((d_mlp, seed, d, ck[-1]))

    # Also include the baseline d_mlp=512 seeds 1, 2, 3 for comparison.
    for s in (1, 2, 3):
        d = RUNS / f"dmodel_24_seed{s}"
        ck = sorted(d.glob("checkpoint_*.pt"))
        if ck:
            runs.append((512, s, d, ck[-1]))

    runs.sort()
    rows = []
    for d_mlp, seed, run_dir, last_ck in runs:
        state = torch.load(last_ck, weights_only=False, map_location="cpu")
        W_E = state["model_state"]["embed.W_E"].double()
        char_E = char_energy_WE(W_E[:, :113], basis, char_idx)
        K = sorted([int(k + 1) for k, e in enumerate(char_E)
                    if e >= 0.05 * char_E.max()])
        orders = sorted([order_of(k) for k in K], reverse=True)
        n_prim = sum(1 for o in orders if o == 112)
        d_mlp_cost = sum(orders)
        has_leg = 2 in orders

        losses_path = run_dir / "losses.pt"
        cliff = None
        if losses_path.exists():
            losses = torch.load(losses_path, weights_only=False)
            cliff = find_cliff(losses["test_losses"])

        rows.append(dict(
            d_mlp=d_mlp, seed=seed, K=K, K_size=len(K),
            orders=orders, n_primitive=n_prim,
            d_mlp_cost=d_mlp_cost, has_legendre=has_leg,
            cliff=cliff,
        ))

    print(f"{'d_mlp':>5} {'seed':>4}  {'cliff':>6} {'|K|':>3} {'cost':>5} "
          f"{'#prim':>5} {'Leg':>3}   orders                       K")
    for r in rows:
        cliff_s = "-" if r["cliff"] is None else str(r["cliff"])
        leg = "Y" if r["has_legendre"] else "."
        print(f"{r['d_mlp']:>5} {r['seed']:>4}  {cliff_s:>6} {r['K_size']:>3} "
              f"{r['d_mlp_cost']:>5} {r['n_primitive']:>5} {leg:>3}   "
              f"{str(r['orders']):<28}  {r['K']}")

    # Write CSV.
    csv_path = OUT_DIR / "dmlp_sweep_summary.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["d_mlp", "seed", "K", "K_size", "orders",
                    "n_primitive", "d_mlp_cost", "has_legendre", "cliff"])
        for r in rows:
            w.writerow([r["d_mlp"], r["seed"],
                        "|".join(map(str, r["K"])),
                        r["K_size"],
                        "|".join(map(str, r["orders"])),
                        r["n_primitive"], r["d_mlp_cost"],
                        "Y" if r["has_legendre"] else "N",
                        r["cliff"] if r["cliff"] is not None else ""])
    print(f"\nSaved {csv_path}")

    # Plots: |K|, d_mlp_cost, n_primitive as a function of d_mlp budget.
    dmlps = sorted(set(r["d_mlp"] for r in rows))
    by_d = {d: [r for r in rows if r["d_mlp"] == d] for d in dmlps}

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, key, title, ylab in [
        (axes[0], "K_size", "|K| vs d_mlp budget", "|K|"),
        (axes[1], "d_mlp_cost", "actual d_mlp_cost vs budget", "Σ order(k)"),
        (axes[2], "n_primitive", "# primitive chars vs budget", "# order-112 in K"),
    ]:
        xs, ys = [], []
        for d in dmlps:
            for r in by_d[d]:
                xs.append(d)
                ys.append(r[key])
        ax.scatter(xs, ys, s=80, alpha=0.75, edgecolor="black", linewidth=0.4)
        ax.set_xscale("log", base=2)
        ax.set_xticks(dmlps); ax.set_xticklabels([str(d) for d in dmlps])
        ax.set_xlabel("d_mlp budget")
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        if key == "d_mlp_cost":
            # Draw budget line.
            ax.plot(dmlps, dmlps, "k--", lw=0.7, alpha=0.5, label="cost = budget")
            ax.legend(fontsize=8)
    fig.suptitle(f"d_mlp sweep summary ({len(rows)} runs)", fontsize=11)
    fig.tight_layout()
    out = FIG_DIR / "dmlp_K_summary.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
