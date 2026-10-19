"""Figures for :mod:`crypto_interp.measures.errors`.

- :func:`profile`: one run, four panels: margin histogram, target vs best
  alternative logit, where the errors are on the (a, b) grid, and per-token
  error rates in the A and B positions.
- :func:`population`: margin distributions, worst margins, and accuracy against
  error count, over several runs colored by regime.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from .kernel import REGIME_COLOR


def profile(e, title: str = ""):
    """``e`` is an :class:`~crypto_interp.measures.ErrorProfile`."""
    p = len(e.err_by_target)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9.6))

    ax = axes[0][0]
    ax.hist(e.margin.ravel(), bins=80, color="steelblue", edgecolor="black", linewidth=0.4)
    ax.axvline(0, color="red", linestyle="--", label="0 (error boundary)")
    ax.axvline(
        e.kernel_margin,
        color="green",
        linestyle=":",
        label=f"kernel margin = {e.kernel_margin:.2g}",
    )
    ax.set(
        xlabel="margin = target − max alternative",
        ylabel="# of (a, b) pairs",
        title=f"{title}\naccuracy {e.accuracy:.4%}, {e.n_wrong} wrong".strip(),
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[0][1]
    right = ~e.wrong.ravel()
    tl, ma = e.target_logit.ravel(), e.max_alt_logit.ravel()
    if (~right).any():
        ax.scatter(
            tl[~right], ma[~right], s=4, color="red", alpha=0.7, label=f"wrong (n={e.n_wrong})"
        )
    ax.scatter(tl[right], ma[right], s=2, color="C0", alpha=0.3, label="correct")
    lo, hi = min(tl.min(), ma.min()), max(tl.max(), ma.max())
    ax.plot([lo, hi], [lo, hi], "k--", alpha=0.3, label="margin = 0")
    ax.set(
        xlabel="target logit L(a, b, ab)",
        ylabel="best alternative logit",
        title="target vs best alternative",
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1][0]
    ax.imshow(e.wrong, cmap="Reds", aspect="auto", extent=[1, p, p, 1], interpolation="nearest")
    ax.set(xlabel="b", ylabel="a", title=f"where errors happen ({e.n_wrong} / {(p - 1) ** 2})")

    ax = axes[1][1]
    x = np.arange(1, p)
    ax.bar(x - 0.2, e.err_by_a, width=0.4, color="steelblue", label="as a")
    ax.bar(x + 0.2, e.err_by_b, width=0.4, color="orange", label="as b")
    ax.set(xlabel="token", ylabel="error rate", title="per-token error rate", xlim=(0, p))
    ax.legend()
    fig.tight_layout()
    return fig


def population(profiles: dict, regimes: dict):
    """``profiles`` maps a label (e.g. seed) to an ErrorProfile; ``regimes`` maps the
    same labels to a regime name."""
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    for label, e in profiles.items():
        col = REGIME_COLOR.get(regimes.get(label), "gray")
        axes[0].hist(
            e.margin.ravel(),
            bins=60,
            alpha=0.4,
            color=col,
            density=True,
            label=f"{label} ({regimes.get(label, '?')})",
        )
        axes[1].plot(np.sort(e.margin.ravel())[:500], color=col, alpha=0.7, label=str(label))
        axes[2].scatter([e.accuracy], [e.n_wrong], c=col, s=120, edgecolor="black", linewidth=0.6)
        axes[2].annotate(f" {label}", (e.accuracy, e.n_wrong), fontsize=10)
    for ax in axes[:2]:
        ax.axvline(0, color="black", ls="--", alpha=0.5) if ax is axes[0] else ax.axhline(
            0, color="black", ls="--", alpha=0.5
        )
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    axes[0].set(
        xlabel="margin", ylabel="density", yscale="log", title="Per-example margin distributions"
    )
    axes[1].set(xlabel="rank (worst 500 examples)", ylabel="margin", title="Worst 500 margins")
    axes[2].set(
        xlabel="accuracy",
        ylabel="# wrong (a, b)",
        yscale="symlog",
        title="Accuracy and error count",
    )
    axes[2].grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    return fig
