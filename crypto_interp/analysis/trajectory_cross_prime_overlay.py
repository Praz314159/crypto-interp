"""Cross-prime trajectory summary: population medians + one exemplar per regime.

Two rows, one column per prime:

  Row 1 — population median of L_empirical (solid) and L_symmetric (dashed)
      with IQR bands, on a common step grid. The headline claim is the gap:
      the dashed line dives early (the algorithm forms), the solid line
      follows late (the noise decays).
  Row 2 — one representative seed per outcome regime (the seed whose final
      L_emp is the regime median), colored by regime. Shows the per-seed
      behavior the medians average over.

Usage:
    python -m crypto_interp.analysis.trajectory_cross_prime_overlay \\
        --in-dirs outputs/theory_trajectory/003_dmodel_sweep_p113 \\
                  outputs/theory_trajectory/004_p127 \\
                  outputs/theory_trajectory/005_p181 \\
        --primes 113 127 181 \\
        --out-file outputs/theory_trajectory/cross_prime_overlay.png
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REGIME_ORDER = ["grokked", "algorithmic-but-noisy", "CRT-fail", "intermediate"]
REGIME_COLOR = {"grokked": "#15803d", "algorithmic-but-noisy": "#d97706",
                "CRT-fail": "#dc2626", "intermediate": "#6b7280"}
EPS = 1e-10


def classify(L_emp_final: float, L_sym_final: float) -> str:
    if L_sym_final > 0.3 and L_emp_final > 0.3 and L_emp_final - L_sym_final < 0.3:
        return "CRT-fail"
    if L_emp_final < 1e-2:
        return "grokked"
    if L_sym_final < 1e-2:
        return "algorithmic-but-noisy"
    return "intermediate"


def load(in_dir: Path) -> list[dict]:
    out = []
    for f in sorted(in_dir.glob("*.npz")):
        d = dict(np.load(f))
        d["seed"] = int(re.search(r"seed(\d+)", f.stem).group(1))
        d["regime"] = classify(float(d["L_emp"][-1]), float(d["L_sym"][-1]))
        out.append(d)
    return out


def _on_grid(trajs: list[dict], key: str, grid: np.ndarray) -> np.ndarray:
    """Stack one quantity from every trajectory onto a common step grid.

    np.interp clamps at the edges, so early-stopped seeds carry their final
    (converged) value forward.
    """
    return np.stack([
        np.interp(grid, t["step"], np.clip(t[key], EPS, None)) for t in trajs
    ])


def plot(in_dirs: list[Path], primes: list[int], out_file: Path) -> None:
    ncol = len(in_dirs)
    fig, axes = plt.subplots(2, ncol, figsize=(4.9 * ncol, 7.6), squeeze=False)

    for j, (in_dir, p) in enumerate(zip(in_dirs, primes)):
        trajs = load(in_dir)
        n_seeds = len(trajs)
        grid = np.linspace(0, max(t["step"].max() for t in trajs), 240)

        # ---- Row 1: medians + IQR bands ---------------------------------
        # Quantiles only over the seeds that eventually find the algorithm
        # (final L_sym < 1e-2): mixing in CRT-fail/intermediate seeds smears
        # the bands across 8 decades. Those regimes are row 2's story.
        ax = axes[0][j]
        finders = [t for t in trajs if float(t["L_sym"][-1]) < 1e-2]
        for key, color, ls, label in [("L_emp", "#1d4ed8", "-", "$L_{\\rm empirical}$"),
                                      ("L_sym", "#ea580c", "--", "$L_{\\rm symmetric}$")]:
            arr = _on_grid(finders, key, grid)
            med = np.median(arr, axis=0)
            q1, q3 = np.quantile(arr, 0.25, axis=0), np.quantile(arr, 0.75, axis=0)
            ax.semilogy(grid, med, color=color, ls=ls, lw=2.0, label=label)
            ax.fill_between(grid, q1, q3, color=color, alpha=0.14, lw=0)
        ax.set_title(f"$p = {p}$ — {len(finders)}/{n_seeds} algorithm-finding seeds",
                     fontsize=10.5)
        ax.set_ylim(EPS, 20)
        ax.grid(True, alpha=0.25, which="both")
        ax.legend(fontsize=9, loc="upper right", framealpha=0.9)
        if j == 0:
            ax.set_ylabel("cross-entropy (log)")

        # ---- Row 2: one exemplar per regime -----------------------------
        ax2 = axes[1][j]
        by_regime: dict[str, list[dict]] = {}
        for t in trajs:
            by_regime.setdefault(t["regime"], []).append(t)
        chosen = []
        for reg in REGIME_ORDER:
            group = by_regime.get(reg)
            if not group or len(chosen) >= 3:
                continue
            finals = np.array([float(t["L_emp"][-1]) for t in group])
            chosen.append(group[int(np.argsort(finals)[len(finals) // 2])])
        for t in chosen:
            col = REGIME_COLOR[t["regime"]]
            ax2.semilogy(t["step"], np.clip(t["L_emp"], EPS, None),
                         color=col, lw=1.8,
                         label=f"seed {t['seed']} — {t['regime']}")
            ax2.semilogy(t["step"], np.clip(t["L_sym"], EPS, None),
                         color=col, lw=1.2, ls="--", alpha=0.8)
        tally = ", ".join(f"{len(by_regime[r])} {r}" for r in REGIME_ORDER
                          if r in by_regime)
        ax2.set_title(f"exemplars ({tally})", fontsize=9.5)
        ax2.set_ylim(EPS, 20)
        ax2.grid(True, alpha=0.25, which="both")
        ax2.set_xlabel("training step")
        ax2.legend(fontsize=8, loc="upper right", framealpha=0.9)
        if j == 0:
            ax2.set_ylabel("cross-entropy (log)")

    fig.suptitle("Kernel/noise decomposition across primes — median with IQR band; "
                 "solid: actual loss ($L_{\\rm emp}$); dashed: loss of the "
                 "symmetric kernel alone ($L_{\\rm sym}$)", fontsize=11.5)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_file}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dirs", nargs="+", required=True)
    ap.add_argument("--primes", nargs="+", type=int, required=True)
    ap.add_argument("--out-file", required=True)
    args = ap.parse_args()
    plot([Path(p) for p in args.in_dirs], args.primes, Path(args.out_file))


if __name__ == "__main__":
    main()
