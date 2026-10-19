"""Figures for the kernel/noise decomposition.

Every function takes a table from :data:`crypto_interp.measures.kernel_noise`
``.over(...)`` and returns a matplotlib Figure; none of them computes anything
from a model. Callers save the figure.

- :func:`regimes`: one point per run. L_emp against L_sym, plus the CRT floor
  against L_K for CRT failures.
- :func:`trajectory`: one run over training (``checkpoints="all"``).
- :func:`trajectory_grid`: a panel per run of a population's trajectories.
- :func:`trajectory_overlay`: population medians and IQR bands per prime, plus
  one exemplar seed per regime.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

REGIME_ORDER = ("grokked", "algorithmic-but-noisy", "CRT-fail", "intermediate")
REGIME_COLOR = {
    "grokked": "#15803d",
    "algorithmic-but-noisy": "#d97706",
    "CRT-fail": "#dc2626",
    "intermediate": "#6b7280",
}
REGIME_MARKER = {"grokked": "o", "algorithmic-but-noisy": "s", "CRT-fail": "X", "intermediate": "^"}
EPS = 1e-10
LOSS_LIM = (EPS, 20.0)


def _final(traj):
    """The last-epoch row of each run in a trajectory table."""
    return traj.sort_values("epoch").groupby("run", sort=False).tail(1).set_index("run")


def regimes(df):
    """Two panels over runs (latest checkpoints): L_emp vs L_sym, colored by prime and
    marked by regime; and, for CRT failures, the algebraic floor log gcd against L_K,
    the loss of the kernel restricted to K, which the floor bounds from below."""
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))
    primes = sorted(df["p"].unique())
    color = {p: f"C{i}" for i, p in enumerate(primes)}

    for p in primes:
        for reg in REGIME_ORDER:
            sel = df[(df["p"] == p) & (df["regime"] == reg)]
            if sel.empty:
                continue
            ax.scatter(
                np.clip(sel["L_sym"], EPS, None),
                np.clip(sel["L_emp"], EPS, None),
                c=color[p],
                marker=REGIME_MARKER[reg],
                s=60,
                alpha=0.75,
                edgecolor="black",
                linewidth=0.4,
                label=f"p={p} {reg}",
            )
    ax.plot(LOSS_LIM, LOSS_LIM, "k--", alpha=0.3, label="y = x")
    ax.set(xscale="log", yscale="log", xlim=LOSS_LIM, ylim=LOSS_LIM)
    ax.set_xlabel("L_symmetric (algorithm)")
    ax.set_ylabel("L_empirical (model)")
    ax.set_title("Regimes across primes")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=7, loc="lower right", ncol=2)

    fails = df[df["regime"] == "CRT-fail"]
    for p in primes:
        sel = fails[fails["p"] == p]
        if sel.empty:
            continue
        ax2.scatter(
            sel["crt_floor"],
            sel["L_K"],
            c=color[p],
            s=80,
            alpha=0.85,
            edgecolor="black",
            linewidth=0.4,
            label=f"p={p}",
        )
        for _, r in sel.iterrows():
            ax2.annotate(f" s{r['seed']}", (r["crt_floor"], r["L_K"]), fontsize=8)
    top = max(1.0, float(fails["L_K"].max()) if not fails.empty else 1.0)
    ax2.plot([0, top], [0, top], "k--", alpha=0.3, label="y = x (floor)")
    ax2.set_xlabel("algebraic floor log gcd(K ∪ {p−1})")
    ax2.set_ylabel("observed L_K (kernel restricted to K)")
    ax2.set_title("CRT failures sit above the algebraic floor")
    ax2.grid(True, alpha=0.3)
    if not fails.empty:
        ax2.legend(fontsize=9)
    fig.tight_layout()
    return fig


def _loss_curves(ax, t, *, markersize=4):
    ax.semilogy(
        t["epoch"],
        np.clip(t["L_emp"], EPS, None),
        "o-",
        color="C0",
        markersize=markersize,
        label="$L_{\\rm emp}$",
    )
    ax.semilogy(
        t["epoch"],
        np.clip(t["L_sym"], EPS, None),
        "s-",
        color="C1",
        markersize=markersize,
        label="$L_{\\rm sym}$",
    )
    ax.semilogy(
        t["epoch"],
        np.clip(t["L_K"], EPS, None),
        "^-",
        color="C2",
        markersize=markersize - 1,
        label="$L_{K}$",
    )
    ax.set_ylim(*LOSS_LIM)
    ax.grid(True, alpha=0.3, which="both")


def trajectory(traj):
    """One run over training: the three losses, and |K|."""
    t = traj.sort_values("epoch")
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 3.6))
    _loss_curves(ax, t)
    ax.set(xlabel="epoch", ylabel="cross-entropy (log)", title=str(t["run"].iloc[0]))
    ax.legend(fontsize=8, loc="lower left")
    ax2.plot(t["epoch"], [len(k) for k in t["K"]], "o-", color="C3", markersize=4)
    ax2.set(xlabel="epoch", ylabel="|K|", title="|K| over training")
    ax2.set_ylim(bottom=0)
    ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def trajectory_grid(traj, title: str = ""):
    """A panel per run, titled by seed and final regime."""
    finals = _final(traj).sort_values(["p", "seed"])
    runs = list(finals.index)
    cols = min(4, len(runs))
    rows = (len(runs) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.0 * cols, 3.0 * rows), squeeze=False)
    for i, run in enumerate(runs):
        ax = axes[i // cols][i % cols]
        _loss_curves(ax, traj[traj["run"] == run].sort_values("epoch"), markersize=3)
        reg = finals.loc[run, "regime"]
        ax.set_title(
            f"seed {finals.loc[run, 'seed']} — {reg}",
            color=REGIME_COLOR[reg],
            fontsize=10,
            fontweight="bold",
        )
        ax.set(xlabel="epoch", ylabel="CE (log)")
        if i == 0:
            ax.legend(fontsize=7, loc="lower left")
    for j in range(len(runs), rows * cols):
        axes[j // cols][j % cols].set_visible(False)
    if title:
        fig.suptitle(title, fontsize=13, y=1.0)
    fig.tight_layout()
    return fig


def _on_grid(runs, key, grid):
    """Each run's ``key`` interpolated onto a common epoch grid. np.interp clamps,
    so early-stopped runs carry their final value forward."""
    return np.stack([np.interp(grid, t["epoch"], np.clip(t[key], EPS, None)) for t in runs])


def trajectory_overlay(traj):
    """Per prime: median L_emp and L_sym with IQR bands over the runs whose kernel
    is eventually solved (final L_sym < 1e-2), and one median exemplar per regime."""
    primes = sorted(traj["p"].unique())
    fig, axes = plt.subplots(2, len(primes), figsize=(4.9 * len(primes), 7.6), squeeze=False)
    finals = _final(traj)
    for j, p in enumerate(primes):
        runs_p = [r for r in finals.index if finals.loc[r, "p"] == p]
        series = {r: traj[traj["run"] == r].sort_values("epoch") for r in runs_p}
        grid = np.linspace(0, max(float(s["epoch"].max()) for s in series.values()), 240)

        ax = axes[0][j]
        finders = [series[r] for r in runs_p if finals.loc[r, "L_sym"] < 1e-2]
        if finders:
            for key, col, ls, label in [
                ("L_emp", "#1d4ed8", "-", "$L_{\\rm emp}$"),
                ("L_sym", "#ea580c", "--", "$L_{\\rm sym}$"),
            ]:
                arr = _on_grid(finders, key, grid)
                ax.semilogy(grid, np.median(arr, axis=0), color=col, ls=ls, lw=2.0, label=label)
                ax.fill_between(
                    grid,
                    np.quantile(arr, 0.25, axis=0),
                    np.quantile(arr, 0.75, axis=0),
                    color=col,
                    alpha=0.14,
                    lw=0,
                )
            ax.legend(fontsize=9, loc="upper right", framealpha=0.9)
        ax.set_title(
            f"$p = {p}$ — {len(finders)}/{len(runs_p)} runs solve the kernel", fontsize=10.5
        )
        ax.set_ylim(*LOSS_LIM)
        ax.grid(True, alpha=0.25, which="both")

        ax2 = axes[1][j]
        tally = []
        for reg in REGIME_ORDER:
            group = [r for r in runs_p if finals.loc[r, "regime"] == reg]
            if not group:
                continue
            tally.append(f"{len(group)} {reg}")
            group.sort(key=lambda r: finals.loc[r, "L_emp"])
            ex = group[len(group) // 2]
            s, col = series[ex], REGIME_COLOR[reg]
            ax2.semilogy(
                s["epoch"],
                np.clip(s["L_emp"], EPS, None),
                color=col,
                lw=1.8,
                label=f"seed {finals.loc[ex, 'seed']} — {reg}",
            )
            ax2.semilogy(
                s["epoch"], np.clip(s["L_sym"], EPS, None), color=col, lw=1.2, ls="--", alpha=0.8
            )
        ax2.set_title(f"exemplars ({', '.join(tally)})", fontsize=9.5)
        ax2.set_ylim(*LOSS_LIM)
        ax2.set_xlabel("epoch")
        ax2.grid(True, alpha=0.25, which="both")
        ax2.legend(fontsize=8, loc="upper right", framealpha=0.9)
        if j == 0:
            ax.set_ylabel("cross-entropy (log)")
            ax2.set_ylabel("cross-entropy (log)")
    fig.suptitle(
        "Kernel/noise decomposition across primes. Solid: model loss "
        "($L_{\\rm emp}$); dashed: loss of the symmetric kernel ($L_{\\rm sym}$)",
        fontsize=11.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig
