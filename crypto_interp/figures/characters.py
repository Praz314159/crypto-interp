"""Figures for :mod:`crypto_interp.measures.characters`.

- :func:`ablation`: Δlog10 test loss from ablating each character, against its
  energy and ranked, colored by character order, with K starred.
- :func:`delta_spectrum`: for each k in K, the frequencies its ablation
  destroys (a helper's peak sits off k).
- :func:`delta`: one character's ablation contribution Δ_k over (ab, c), in
  natural and dlog order, and reduced to the dlog offset.
- :func:`cluster`: one character's neuron-cluster output beside the reference
  cos(θ_k(a) + θ_k(b)), in natural and dlog order.
- :func:`fourier2d`: MLP-output energy over character pairs, and its diagonal.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from ..measures.characters import dlog_order


def _order_colors(orders):
    palette = plt.get_cmap("tab10")
    distinct = sorted(set(int(o) for o in orders))
    return {o: palette(i % 10) for i, o in enumerate(distinct)}


def ablation(abl, sup, title: str = ""):
    """``abl`` is a CharAblation, ``sup`` the run's EnergySupport."""
    ks = np.arange(1, len(abl.dlog10) + 1)
    inK = np.isin(ks, sup.K)
    colors = _order_colors(sup.char_order)
    c = [colors[int(o)] for o in sup.char_order]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(15, 5.5))
    for mask, marker, size in [(~inK, "o", 49), (inK, "*", 196)]:
        ax.scatter(
            sup.energy[mask],
            abl.dlog10[mask],
            c=np.array(c, dtype=object)[mask].tolist(),
            marker=marker,
            s=size,
            alpha=0.85,
            edgecolor="black",
            linewidth=0.5,
        )
    for k in ks[inK | (abl.dlog10 > 1.5)]:
        ax.annotate(
            f"k={k}",
            (sup.energy[k - 1], abl.dlog10[k - 1]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7,
        )
    ax.set(
        xscale="log",
        xlabel="W_E energy in χ_k",
        ylabel="Δlog10 test loss after ablating χ_k",
        title=f"{title} ablation impact (★ = K)".strip(),
    )
    ax.grid(True, alpha=0.3)
    ax.legend(
        handles=[
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=col,
                markeredgecolor="black",
                markersize=8,
                label=f"order {o}",
            )
            for o, col in colors.items()
        ],
        fontsize=7,
        loc="lower right",
        ncol=2,
    )

    order = np.argsort(-abl.dlog10)
    ax2.bar(
        np.arange(len(order)),
        abl.dlog10[order],
        color=[c[i] for i in order],
        edgecolor="black",
        linewidth=0.4,
    )
    for i, j in enumerate(order):
        if inK[j]:
            ax2.text(
                i, abl.dlog10[j] + 0.1, f"k={j + 1}", fontsize=7, ha="center", fontweight="bold"
            )
    ax2.set(xlabel="characters, by ablation impact", ylabel="Δlog10 test loss", title="ranked")
    ax2.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    return fig


def delta_spectrum(h, title: str = ""):
    """``h`` is a Helpers result."""
    n = h.delta_diff.shape[1]
    fig, axes = plt.subplots(len(h.K), 1, figsize=(13, 2.2 * len(h.K)), sharex=True, squeeze=False)
    for ax, k, spec, m in zip(axes[:, 0], h.K, h.spectrum, h.dominant, strict=True):
        ax.bar(np.arange(1, len(spec) + 1), spec, color="#1f77b4", alpha=0.85)
        for f, col, lab in [
            (k, "red", f"k={k}"),
            (2 * k, "orange", f"2k={2 * k}"),
            (n - k, "green", f"−k={n - k}"),
        ]:
            if f <= n // 2:
                ax.axvline(f, color=col, ls="--", lw=1.0, alpha=0.7, label=lab)
        ax.set(ylabel=f"ablate k={k}\n(peak {m})", yscale="log")
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.3, axis="y")
    axes[-1, 0].set_xlabel("Fourier frequency m of Δ_k(Δlog)")
    fig.suptitle(f"{title} which frequency does ablating each k destroy?".strip(), fontsize=11)
    fig.tight_layout()
    return fig


def delta(h, k: int, p: int):
    """Δ_k(ab, c) for one k in ``h.K``."""
    j = h.K.index(k)
    f_ab, f_diff = h.delta_ab[j], h.delta_diff[j]
    idx = dlog_order(p)
    d = np.arange(p - 1)
    ref = np.cos(2 * np.pi * k * d / (p - 1))
    vmax = float(np.abs(f_ab).max())
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, data, lab in [(axes[0], f_ab, "natural"), (axes[1], f_ab[idx][:, idx], "dlog order")]:
        ax.imshow(data, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="equal", origin="upper")
        ax.set(xlabel=f"c ({lab})", ylabel=f"ab ({lab})", title=f"Δ_{k}(ab, c), {lab}")
    scale = (f_diff @ ref) / (ref @ ref)
    axes[2].plot(d, f_diff, "b-", lw=1.6, label=f"Δ_{k}(Δlog)")
    axes[2].plot(d, scale * ref, "r--", lw=1.4, label=f"{scale:.2e}·cos(2π{k}Δlog/(p−1))")
    axes[2].set(
        xlabel="Δlog = log c − log ab",
        ylabel=f"mean Δ_{k}",
        title=f"reduced to the dlog offset; corr {h.delta_corr[j]:+.3f}",
    )
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def cluster(nc, k: int, p: int):
    """Neuron cluster for one k in ``nc.K`` against its reference."""
    j = nc.K.index(k)
    idx = dlog_order(p)
    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    for row, (data, name) in enumerate(
        [(nc.signal[j], "cluster output"), (nc.reference[j], "cos(θ_k(a) + θ_k(b))")]
    ):
        vmax = float(np.abs(data).max()) or 1.0
        for col, (d, lab) in enumerate([(data, "natural"), (data[idx][:, idx], "dlog order")]):
            ax = axes[row, col]
            ax.imshow(d, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="equal", origin="upper")
            ax.set(title=f"{name}, {lab}", xlabel=f"b ({lab})", ylabel=f"a ({lab})")
    fig.suptitle(
        f"χ_{k}: {nc.sizes[j]} neurons; correlation with reference {nc.corr[j]:+.3f}", fontsize=11
    )
    fig.tight_layout()
    return fig


def fourier2d(f2d, title: str = ""):
    E = f2d.energy
    nch = E.shape[0]
    diag = np.diag(E)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    im = ax.imshow(
        np.log10(E + 1e-12),
        cmap="viridis",
        aspect="equal",
        origin="lower",
        extent=[1, nch + 1, 1, nch + 1],
    )
    ax.plot([1, nch], [1, nch], "w--", lw=0.5, alpha=0.5)
    ax.set(xlabel="k_b", ylabel="k_a", title="log10 E[k_a, k_b]")
    plt.colorbar(im, ax=ax, shrink=0.85)
    ax2.bar(
        np.arange(1, nch + 1), diag, color="#d62728", alpha=0.85, edgecolor="black", linewidth=0.3
    )
    for k in np.flatnonzero(diag > 0.05 * diag.max()) + 1:
        ax2.text(k, diag[k - 1] * 1.02, str(k), ha="center", fontsize=8, fontweight="bold")
    ax2.set(xlabel="k", ylabel="E[k, k]", yscale="log", title="diagonal")
    ax2.grid(True, alpha=0.3, axis="y")
    fig.suptitle(
        f"{title} MLP output by character pair: {100 * f2d.diagonal_fraction:.1f}% "
        "on the diagonal".strip(),
        fontsize=11,
    )
    fig.tight_layout()
    return fig
