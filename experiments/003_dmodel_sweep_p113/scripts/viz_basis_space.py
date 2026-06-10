"""Visualize basis-space structure from the cached multiset data.

Loads ``data/basis_cache.pkl`` and produces two plots:
  (1) Overview — all multisets in cost-quality space with Pareto frontier overlaid.
  (2) Per-size subplots — separate panel for each |K|, each with its own Pareto curve.

Usage:
    python experiments/003_dmodel_sweep_p113/scripts/viz_basis_space.py
"""
from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "basis_cache.pkl"
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

SEED_OBS = {
    "seed 0": ((112, 112, 112, 56, 2), "#1f77b4"),
    "seed 1": ((112, 56, 28, 28, 14, 2), "#d62728"),
    "seed 2": ((112, 56, 14, 2), "#2ca02c"),
}


# ---------- utilities ----------

def load_cache():
    with open(CACHE, "rb") as f:
        return pickle.load(f)


def pareto_curve(df_sub):
    """Return arrays (d_mlp, gap) tracing the (lower-d_mlp-is-better,
    higher-gap-is-better) Pareto frontier of df_sub."""
    s = df_sub.sort_values("d_mlp_cost").reset_index(drop=True)
    dml, gap = s["d_mlp_cost"].values, s["best_gap"].values
    pareto_dml, pareto_gap = [], []
    best = -np.inf
    for v, g in zip(dml, gap):
        if g > best:
            best = g
            pareto_dml.append(v)
            pareto_gap.append(g)
    return np.array(pareto_dml), np.array(pareto_gap)


def find_seed(df, ms_tup):
    key = tuple(sorted(ms_tup, reverse=True))
    rows = df[df["multiset"] == key]
    return rows.iloc[0] if len(rows) else None


def scatter_multisets(ax, sub, *, vmin=0, vmax=6, alpha_no=0.4, alpha_leg=0.6,
                     base_size=20, size_scale=8):
    """Scatter helper: colored by # primitives, shaped by Legendre."""
    leg_mask = sub["has_legendre"].astype(bool).values
    sc = None
    for has_l, marker, alpha in [(False, "o", alpha_no), (True, "s", alpha_leg)]:
        ss = sub[leg_mask == has_l]
        if not len(ss):
            continue
        sc = ax.scatter(
            ss["d_mlp_cost"], ss["best_gap"],
            c=ss["n_primitive"], cmap="plasma", vmin=vmin, vmax=vmax,
            marker=marker, s=base_size + size_scale * ss["size"],
            alpha=alpha, edgecolors="none",
        )
    return sc


def annotate_seeds(ax, df, *, sizes=None):
    """Star and label the observed seeds present in df (optionally filtered to sizes)."""
    for name, (ms, color) in SEED_OBS.items():
        if sizes is not None and len(ms) not in sizes:
            continue
        row = find_seed(df, ms)
        if row is None:
            continue
        ax.scatter([row["d_mlp_cost"]], [row["best_gap"]], s=380, marker="*",
                   c=color, edgecolors="white", linewidths=1.8, zorder=10)
        ax.annotate(name, (row["d_mlp_cost"], row["best_gap"]),
                    xytext=(10, 10), textcoords="offset points",
                    fontsize=10, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white",
                              ec=color, alpha=0.95), zorder=11)


# ---------- plots ----------

def plot_overview(df, gap_min=0.5):
    """Combined plot: all sizes, Pareto frontier overlaid."""
    f = df[df["best_gap"] > gap_min].copy()
    fig, ax = plt.subplots(figsize=(13, 7))
    sc = scatter_multisets(ax, f)
    # Pareto frontier
    dml, gap = pareto_curve(f)
    ax.step(dml, gap, where="post", color="black", lw=2.0, zorder=5,
            label=f"Pareto frontier ({len(dml)} points)")
    cbar = fig.colorbar(sc, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label("# of primitive (order-112) characters")
    annotate_seeds(ax, df)
    legend = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="gray", markersize=9,
               label="no Legendre", markeredgecolor="black"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="gray", markersize=9,
               label="contains Legendre", markeredgecolor="black"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="red", markersize=14,
               label="observed seed", markeredgecolor="white"),
        Line2D([0], [0], color="black", lw=2.0, label="Pareto frontier"),
    ]
    ax.legend(handles=legend, loc="lower right", fontsize=10, framealpha=0.95)
    ax.set_xscale("log")
    ax.set_xlabel("d_mlp cost (Σ order(k))")
    ax.set_ylabel(r"worst-case logit gap $\Delta_{\min}$")
    ax.set_title(
        f"Basis space at p=113.  {len(f)} multisets of size 3–7 with gap > {gap_min}.\n"
        f"Color = # primitive characters; shape = Legendre; size ∝ |K|.",
        fontsize=11,
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = FIG_DIR / "basis_space_overview.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"Saved {out}")


def plot_per_size(df, gap_min=0.3):
    """One panel per |K|, with that size's own Pareto frontier."""
    sizes = sorted(df["size"].unique())
    nrows, ncols = 2, 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(17, 9.5), sharey=False)
    axes = axes.ravel()
    last_sc = None
    for i, sz in enumerate(sizes):
        ax = axes[i]
        sub = df[(df["size"] == sz) & (df["best_gap"] > gap_min)]
        if not len(sub):
            ax.set_visible(False)
            continue
        last_sc = scatter_multisets(ax, sub, base_size=12, size_scale=10)
        dml, gap = pareto_curve(sub)
        ax.step(dml, gap, where="post", color="black", lw=2.0, zorder=5)
        annotate_seeds(ax, df, sizes=[sz])
        ax.set_xscale("log")
        ax.set_xlim(10, 800)
        ax.set_title(f"|K| = {sz}   ({len(sub)} multisets, frontier {len(dml)} steps)",
                     fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("d_mlp cost")
        ax.set_ylabel(r"worst-case gap $\Delta_{\min}$")
    # Hide any leftover axes (we use 6 panels for 5 sizes).
    for j in range(len(sizes), nrows * ncols):
        axes[j].set_visible(False)
    if last_sc is not None:
        cbar = fig.colorbar(last_sc, ax=axes.tolist(), shrink=0.55, pad=0.02,
                            location="right")
        cbar.set_label("# of primitive (order-112) characters")
    fig.suptitle(
        f"Basis space per-size, p=113.  Panels: |K| = 3..7.  Black step = Pareto frontier.\n"
        f"Color = # primitives; shape = Legendre (square / circle).",
        fontsize=12,
    )
    out = FIG_DIR / "basis_space_per_size.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"Saved {out}")


def main():
    df = load_cache()
    print(f"Loaded {len(df)} multisets, sizes {sorted(df['size'].unique())}")
    plot_overview(df)
    plot_per_size(df)


if __name__ == "__main__":
    main()
