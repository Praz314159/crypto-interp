"""Empirical distribution of K-order multisets across all 17 seeds.

For each seed's final K, compute:
  - the order multiset (sorted descending) of its characters
  - the d_model_cost = 2|K| - (1 if Legendre present else 0)
  - the d_mlp_cost  = sum of orders
  - the realized worst-case logit gap

Compare against the full basis_cache (8668 multisets) to see whether the
observed seeds sit on the Pareto frontier or near it.

Outputs:
  figures/basis_space_with_all_seeds.png   (overview with all observed seeds)
  figures/basis_dynamics/order_multiset_freq.png  (frequency by class)
  figures/basis_dynamics/order_multiset_table.csv
"""
from __future__ import annotations

import math
import pickle
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TRAJ_FILE = ROOT / "data" / "basis_dynamics" / "trajectories.pkl"
CACHE = ROOT / "data" / "basis_cache.pkl"
FIG_DIR = ROOT / "figures"
FIG_DIR_BASIS = ROOT / "figures" / "basis_dynamics"


def order_of(k, n=112):
    return n // math.gcd(k, n)


COS_TABLE = np.cos(2 * np.pi * np.outer(np.arange(1, 113), np.arange(1, 113)) / 112)


def gap_for_basis(K):
    L = COS_TABLE[np.array(K) - 1].sum(axis=0)
    return float(len(K) - L.max())


def main():
    with open(TRAJ_FILE, "rb") as f:
        traj = pickle.load(f)
    with open(CACHE, "rb") as f:
        import pandas as pd
        df = pickle.load(f) if False else pd.read_pickle(CACHE)

    rows = []
    for seed in sorted(traj):
        ce = traj[seed]["char_energy"][-1]
        K = sorted([int(k + 1) for k, e in enumerate(ce) if e >= 0.05 * ce.max()])
        orders = tuple(sorted([order_of(k) for k in K], reverse=True))
        has_leg = 2 in orders
        d_model_cost = 2 * len(K) - (1 if has_leg else 0)
        d_mlp_cost = sum(orders)
        gap = gap_for_basis(K)
        # Find this multiset in the cache.
        row_cache = df[df["multiset"] == orders]
        best_gap = float(row_cache["best_gap"].iloc[0]) if len(row_cache) else None
        rows.append(dict(
            seed=seed, K=K, orders=orders, size=len(K),
            has_legendre=has_leg, d_model_cost=d_model_cost,
            d_mlp_cost=d_mlp_cost, gap=gap, best_gap=best_gap,
            n_primitive=sum(1 for o in orders if o == 112),
            n_order56=sum(1 for o in orders if o == 56),
            n_order28=sum(1 for o in orders if o == 28),
            n_order16=sum(1 for o in orders if o == 16),
        ))

    print(f"{'seed':>4} {'|K|':>3} {'d_m':>3} {'d_mlp':>6} {'gap':>5} "
          f"{'best':>5} {'#112':>4} {'#56':>3} {'#28':>3} {'#16':>3} "
          f"{'Leg':>3}   orders")
    for r in rows:
        print(f"{r['seed']:>4} {r['size']:>3} {r['d_model_cost']:>3} "
              f"{r['d_mlp_cost']:>6} {r['gap']:>5.2f} "
              f"{str(r['best_gap'])[:5]:>5} "
              f"{r['n_primitive']:>4} {r['n_order56']:>3} {r['n_order28']:>3} "
              f"{r['n_order16']:>3} {'Y' if r['has_legendre'] else '.':>3}   "
              f"{list(r['orders'])}")

    # Frequency of multisets.
    multi_count = Counter(r["orders"] for r in rows)
    print(f"\nMultiset frequency ({len(multi_count)} distinct of {len(rows)}):")
    for ms, c in sorted(multi_count.items(), key=lambda x: -x[1]):
        print(f"  ×{c}  {list(ms)}")

    # K-size distribution.
    sizes = Counter(r["size"] for r in rows)
    print(f"\nK-size distribution:")
    for sz in sorted(sizes):
        print(f"  |K|={sz}: {sizes[sz]} seeds")

    # Class summary.
    print(f"\nLegendre present: {sum(r['has_legendre'] for r in rows)}/{len(rows)}")
    print(f"Has order-112:    {sum(r['n_primitive'] > 0 for r in rows)}/{len(rows)}")
    print(f"All-primitive K:  {sum(r['n_primitive'] == r['size'] for r in rows)}/{len(rows)}")
    print(f"No primitive K:   {sum(r['n_primitive'] == 0 for r in rows)}/{len(rows)}")

    # --- Pareto plot with all 17 seeds annotated ---
    df_filt = df[df["best_gap"] > 0.5].copy()
    fig, ax = plt.subplots(figsize=(13, 7))
    # Background scatter.
    sc = ax.scatter(df_filt["d_mlp_cost"], df_filt["best_gap"],
                    c=df_filt["n_primitive"], cmap="plasma",
                    vmin=0, vmax=6, s=8 + 4 * df_filt["size"],
                    alpha=0.25, edgecolors="none")
    cb = fig.colorbar(sc, ax=ax, shrink=0.7, pad=0.02)
    cb.set_label("# of primitive (order-112) characters")

    # Pareto frontier.
    pareto_df = df_filt.sort_values("d_mlp_cost").reset_index(drop=True)
    dml = pareto_df["d_mlp_cost"].values
    gp = pareto_df["best_gap"].values
    pareto_dml, pareto_gp = [], []
    best = -np.inf
    for v, g in zip(dml, gp):
        if g > best:
            best = g
            pareto_dml.append(v)
            pareto_gp.append(g)
    ax.step(pareto_dml, pareto_gp, where="post", color="black", lw=2.0,
            zorder=5, label=f"Pareto frontier ({len(pareto_dml)} steps)")

    # Mark each seed.
    cmap = plt.cm.tab20
    used_orders = {}
    for r in rows:
        if r["best_gap"] is None:
            continue
        key = r["orders"]
        if key not in used_orders:
            used_orders[key] = len(used_orders)
        color = cmap(used_orders[key] / max(1, len(used_orders) - 1))
        ax.scatter([r["d_mlp_cost"]], [r["best_gap"]], s=260, marker="*",
                   c=[color], edgecolors="white", linewidths=1.5, zorder=10)
        ax.annotate(f"s{r['seed']}", (r["d_mlp_cost"], r["best_gap"]),
                    xytext=(7, 7), textcoords="offset points", fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white",
                              ec=color, alpha=0.9), zorder=11)
    ax.set_xscale("log")
    ax.set_xlabel("d_mlp cost (Σ order(k))")
    ax.set_ylabel(r"max worst-case logit gap $\Delta_{\min}$")
    ax.set_title(f"Basis space at p=113 with all 17 observed seeds.\n"
                 f"Background: {len(df_filt)} multisets with gap > 0.5; "
                 f"size ∝ |K|.")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=10)
    fig.tight_layout()
    out = FIG_DIR / "basis_space_with_all_seeds.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {out}")

    # --- Size+class bar chart ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    sz_keys = sorted(sizes)
    ax.bar([str(s) for s in sz_keys], [sizes[s] for s in sz_keys], color="#1f77b4")
    ax.set_xlabel("|K|"); ax.set_ylabel("# seeds")
    ax.set_title("K-size distribution")

    ax = axes[1]
    classes = ["all primitive\n(K⊂o=112)", "primitive in K\n(≥1 o=112)",
               "no primitive\n(0 o=112)", "Legendre (o=2)\nin K"]
    cnts = [
        sum(r["n_primitive"] == r["size"] for r in rows),
        sum(r["n_primitive"] > 0 for r in rows),
        sum(r["n_primitive"] == 0 for r in rows),
        sum(r["has_legendre"] for r in rows),
    ]
    ax.bar(classes, cnts, color=["#d62728", "#ff7f0e", "#2ca02c", "#9467bd"])
    ax.set_ylabel("# seeds")
    ax.set_title("K class structure (17 seeds)")
    for ax in axes:
        ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    out2 = FIG_DIR_BASIS / "order_multiset_freq.png"
    fig.savefig(out2, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out2}")

    # CSV
    import csv
    csv_path = FIG_DIR_BASIS / "order_multiset_table.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed", "size", "K", "orders", "d_model_cost", "d_mlp_cost",
                    "gap", "best_gap_in_cache", "n_primitive", "n_order56",
                    "n_order28", "n_order16", "has_legendre"])
        for r in rows:
            w.writerow([r["seed"], r["size"],
                        "|".join(map(str, r["K"])),
                        "|".join(map(str, r["orders"])),
                        r["d_model_cost"], r["d_mlp_cost"],
                        f"{r['gap']:.3f}",
                        f"{r['best_gap']:.3f}" if r["best_gap"] is not None else "",
                        r["n_primitive"], r["n_order56"],
                        r["n_order28"], r["n_order16"],
                        "Y" if r["has_legendre"] else "N"])
    print(f"Saved {csv_path}")


if __name__ == "__main__":
    main()
