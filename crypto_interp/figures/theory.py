"""Figures for :mod:`crypto_interp.theory`."""

from __future__ import annotations

import matplotlib.pyplot as plt


def helper_fvu(fvu: dict, k: int):
    """FVU against MLP width from :func:`crypto_interp.theory.relu_harmonics.product_fit_fvu`."""
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.plot(
        fvu["widths"],
        fvu["primary"],
        "o-",
        color="#d62728",
        lw=1.8,
        label=rf"primary only $\{{\chi_{{{k}}}\}}$",
    )
    ax.plot(
        fvu["widths"],
        fvu["helper"],
        "s-",
        color="#1f77b4",
        lw=1.8,
        label=rf"primary + helper $\{{\chi_{{{k}}}, \chi_{{{2 * k}}}\}}$",
    )
    ax.set(
        yscale="log",
        xlabel="MLP width (# ReLU neurons)",
        ylabel="fraction of variance unexplained",
        title=rf"Approximating $\cos({k}(\varphi_a+\varphi_b))$ with a ReLU MLP",
    )
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    return fig
