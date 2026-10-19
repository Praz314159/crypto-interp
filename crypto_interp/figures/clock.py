"""Figures for :mod:`crypto_interp.measures.clock`.

- :func:`residual_vs_error`: over the tokens of many runs (a ``token_table``),
  clock deviation against token error rate, and its distribution, for tokens
  of grokked runs, and for clean and defect tokens of noisy runs.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from ..measures.errors import DEFECT_ERROR_RATE


def residual_vs_error(tokens):
    err = np.maximum(tokens["err_a"], tokens["err_b"])
    noisy = tokens["regime"] == "algorithmic-but-noisy"
    groups = [
        (tokens["regime"] == "grokked", "#9ca3af", "tokens of grokked runs", 0.25, 8),
        (noisy & (err <= DEFECT_ERROR_RATE), "#2563eb", "noisy runs: clean tokens", 0.35, 10),
        (noisy & (err > DEFECT_ERROR_RATE), "#dc2626", "noisy runs: defect tokens", 0.9, 26),
    ]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    bins = np.linspace(0, float(tokens["in_dev"].max()), 60)
    for sel, col, label, alpha, size in groups:
        if not sel.any():
            continue
        ax.scatter(
            tokens.loc[sel, "in_dev"],
            err[sel],
            c=col,
            alpha=alpha,
            s=size,
            label=label,
            edgecolors="none",
        )
        ax2.hist(
            tokens.loc[sel, "in_dev"],
            bins=bins,
            density=True,
            histtype="step",
            lw=1.8,
            color=col,
            label=label,
        )
    ax.set(
        xlabel="deviation from the fitted clock",
        ylabel="token error rate (max of A, B)",
        title="Is the defect a bent clock hand?",
    )
    ax.set_yscale("symlog", linthresh=1e-3)
    ax2.set(
        xlabel="deviation from the fitted clock", ylabel="density", title="Deviation distributions"
    )
    for a in (ax, ax2):
        a.grid(alpha=0.3)
        a.legend(fontsize=9)
    fig.tight_layout()
    return fig
