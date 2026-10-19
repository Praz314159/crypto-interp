"""Figures for :mod:`crypto_interp.measures.noise`.

- :func:`classes`: one run's residual power per Fourier mode class.
- :func:`population`: over runs, the class composition of the noise; noise at
  the target against the CE gap L_emp − L_sym; total noise power against L_emp.
  The table should carry a ``regime`` column (merge with kernel_noise).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from .kernel import REGIME_COLOR

CLASS_COLOR = {
    "off_resonance": "#1f77b4",
    "single_c": "#ff7f0e",
    "generic": "#bcbd22",
    "pair_ac": "#2ca02c",
    "pair_bc": "#98df8a",
    "single_a": "#d62728",
    "single_b": "#ff9896",
    "pair_ab": "#9467bd",
}


def classes(n, title: str = ""):
    """``n`` is a :class:`~crypto_interp.measures.NoiseModes`."""
    names = list(CLASS_COLOR)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(names, [getattr(n, c) for c in names], color=[CLASS_COLOR[c] for c in names])
    ax.set(
        ylabel="residual power",
        yscale="log",
        title=f"{title}\nL_emp={n.L_emp:.3g}, L_sym={n.L_sym:.3g}".strip(),
    )
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    ax.grid(True, axis="y", alpha=0.3, which="both")
    fig.tight_layout()
    return fig


def population(df, label: str = "seed"):
    """One bar / point per row of a noise_modes table (with a ``regime`` column)."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))
    labels = [str(v) for v in df[label]]
    colors = [REGIME_COLOR.get(r, "gray") for r in df.get("regime", ["?"] * len(df))]

    ax = axes[0]
    bottom = np.zeros(len(df))
    for cls, col in CLASS_COLOR.items():
        frac = (df[cls] / df["total_power"]).to_numpy()
        ax.bar(labels, 100 * frac, bottom=100 * bottom, label=cls, color=col)
        bottom += frac
    ax.set(
        ylabel="% of residual power",
        xlabel=label,
        ylim=(0, 105),
        title="Noise composition by mode class",
    )
    ax.legend(fontsize=8, loc="upper right", bbox_to_anchor=(1.02, 1.0))

    for ax, x, y, xl, yl, title in [
        (
            axes[1],
            df["noise_at_target_var"],
            df["L_emp"] - df["L_sym"],
            "Var ε(a, b, ab) (noise at target)",
            "L_emp − L_sym",
            "Target noise vs the CE gap",
        ),
        (
            axes[2],
            df["total_power"],
            df["L_emp"],
            "total residual power",
            "L_emp",
            "Total noise power vs loss",
        ),
    ]:
        ax.scatter(x, y, c=colors, s=80, edgecolor="black", linewidth=0.6)
        for xi, yi, s in zip(x, y, labels, strict=True):
            ax.annotate(f" {s}", (xi, yi), fontsize=9)
        ax.set(xscale="log", yscale="log", xlabel=xl, ylabel=yl, title=title)
        ax.grid(True, alpha=0.3, which="both")
    xx = np.geomspace(
        max(float(df["noise_at_target_var"].min()), 1e-12),
        float(df["noise_at_target_var"].max()),
        100,
    )
    axes[1].plot(xx, xx / 2, "--", color="gray", alpha=0.5, label="y = x / 2 (Gaussian)")
    axes[1].legend(fontsize=9)
    fig.tight_layout()
    return fig
