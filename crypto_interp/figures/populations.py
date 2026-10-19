"""Population figures over joined measure tables.

- :func:`lattice`: per prime, the grokking rate, the size of K over grokked runs,
  and the helper multipliers (needs ``p``, ``regime``, ``K`` from kernel_noise
  and ``helper_mult`` from helpers).
- :func:`by_d_mlp`: |K| and the grokking cliff against MLP width (needs
  ``d_mlp``, ``K``, ``cliff_epoch``, ``status``).
- :func:`cliff_vs_bifurcation`: when K starts to outgrow the other characters,
  and the K / non-K energy ratio at ``at_epoch`` (pass the value the table was
  computed with), against when the run grokks (a dynamics_markers table).
- :func:`bifurcation_overlay`: the K / non-K energy ratio over training, one
  line per run (DynamicsMarkers results).
"""

from __future__ import annotations

from collections import Counter

import matplotlib.pyplot as plt
import numpy as np


def _factor_str(n: int) -> str:
    parts, d = [], 2
    while d * d <= n:
        e = 0
        while n % d == 0:
            n //= d
            e += 1
        if e:
            parts.append(f"{d}^{e}" if e > 1 else str(d))
        d += 1
    if n > 1:
        parts.append(str(n))
    return "·".join(parts)


def lattice(df):
    primes = sorted(df["p"].unique())
    grok = df[df["regime"] == "grokked"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    ax = axes[0]
    rates = [(df[df.p == p].regime == "grokked").mean() for p in primes]
    ax.bar(range(len(primes)), rates, color="steelblue", alpha=0.85)
    for i, p in enumerate(primes):
        ax.text(
            i,
            rates[i] + 0.02,
            f"{(grok.p == p).sum()}/{(df.p == p).sum()}",
            ha="center",
            fontsize=9,
        )
    ax.set_xticks(range(len(primes)), [f"p={p}\n{_factor_str(p - 1)}" for p in primes])
    ax.set(ylabel="fraction grokked", ylim=(0, 1.05), title="Grokking rate by prime")

    width = 0.8 / len(primes)
    sizes = sorted({len(k) for k in grok["K"]}) or [0]
    for i, p in enumerate(primes):
        c = Counter(len(k) for k in grok[grok.p == p]["K"])
        axes[1].bar(
            np.array(sizes) + (i - (len(primes) - 1) / 2) * width,
            [c.get(s, 0) for s in sizes],
            width=width,
            label=f"p={p}",
        )
    axes[1].set(xticks=sizes, xlabel="|K|", ylabel="# grokked runs", title="Size of K")
    axes[1].legend(fontsize=9)

    mults = sorted({m for ms in grok["helper_mult"] for m in ms if m}) or [2]
    for i, p in enumerate(primes):
        c = Counter(m for ms in grok[grok.p == p]["helper_mult"] for m in ms if m)
        axes[2].bar(
            np.array(mults) + (i - (len(primes) - 1) / 2) * width,
            [c.get(m, 0) for m in mults],
            width=width,
            label=f"p={p}",
        )
    axes[2].set(
        xticks=mults,
        xlabel="helper multiplier (χ_helper = χ_primary^j)",
        ylabel="# helpers",
        title="Helper multipliers",
    )
    axes[2].legend(fontsize=9)
    for ax in axes:
        ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    return fig


def by_d_mlp(df):
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    rng = np.random.default_rng(0)
    for status, col in [
        ("grokked", "#15803d"),
        ("memorized-only", "#dc2626"),
        ("no-fit", "#6b7280"),
    ]:
        sel = df[df["status"] == status]
        if sel.empty:
            continue
        x = sel["d_mlp"] * np.exp(rng.uniform(-0.05, 0.05, len(sel)))
        ax.scatter(x, [len(k) for k in sel["K"]], c=col, s=40, alpha=0.8, label=status)
        ax2.scatter(x, sel["cliff_epoch"], c=col, s=40, alpha=0.8, label=status)
    ax.set(xscale="log", xlabel="d_mlp", ylabel="|K|", title="Characters used vs MLP width")
    ax2.set(xscale="log", xlabel="d_mlp", ylabel="cliff epoch", title="Grokking time vs MLP width")
    for a in (ax, ax2):
        a.grid(True, alpha=0.3)
        a.legend(fontsize=8)
    fig.tight_layout()
    return fig


def cliff_vs_bifurcation(df, at_epoch: int = 1500):
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))
    for klass, sel in df.groupby("K_class"):
        ax.scatter(sel["bifurcation_epoch"], sel["cliff_epoch"], s=50, alpha=0.8, label=klass)
        ax2.scatter(sel["ratio_at"], sel["cliff_epoch"], s=50, alpha=0.8, label=klass)
    ax.set(
        xlabel="bifurcation epoch (K outgrows the rest)",
        ylabel="cliff epoch",
        title="Bifurcation vs grokking cliff",
    )
    ax2.set(
        xscale="log",
        xlabel=f"K / non-K energy ratio at epoch {at_epoch}",
        ylabel="cliff epoch",
        title="Energy ratio vs grokking cliff",
    )
    for a in (ax, ax2):
        a.grid(True, alpha=0.3)
        a.legend(fontsize=8, title="K class")
    fig.tight_layout()
    return fig


def bifurcation_overlay(results: dict):
    """``results`` maps a label (e.g. seed) to a DynamicsMarkers result."""
    fig, ax = plt.subplots(figsize=(11, 5))
    cmap = plt.get_cmap("viridis")
    for i, (label, d) in enumerate(results.items()):
        if not len(d.epochs):
            continue
        col = cmap(i / max(1, len(results) - 1))
        ax.semilogy(
            d.epochs,
            d.ratio,
            color=col,
            lw=1.0,
            alpha=0.85,
            label=f"{label} (bif {d.bifurcation_epoch})",
        )
        if d.bifurcation_epoch is not None:
            ax.axvline(d.bifurcation_epoch, color=col, ls=":", lw=0.8, alpha=0.6)
    ax.set(
        xlabel="epoch",
        ylabel="K / non-K mean energy",
        title="When the final characters start to outgrow the rest",
    )
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    return fig
