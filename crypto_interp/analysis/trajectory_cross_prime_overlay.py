"""Overlay grokking trajectories across primes on a single plot.

Three-panel: one panel per prime, all that prime's trajectories overlaid.
Shows L_emp, L_sym, L_K as three light curves per seed, colored by regime.

This is the "are the trajectories shaped similarly across primes" view —
complement to the per-prime grid plots from trajectory_population_plot.py.

Usage:
    python -m crypto_interp.analysis.trajectory_cross_prime_overlay \\
        --in-dirs experiments/theory_trajectory \\
                  experiments/theory_trajectory_p127 \\
                  experiments/theory_trajectory_p181 \\
        --primes 113 127 181 \\
        --out-file experiments/theory_trajectory/cross_prime_overlay.png
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def classify(L_emp_final: float, L_sym_final: float) -> tuple[str, str]:
    if L_sym_final > 0.3 and L_emp_final > 0.3 and L_emp_final - L_sym_final < 0.3:
        return ("red", "CRT-fail")
    if L_emp_final < 1e-2:
        return ("green", "grokked")
    if L_sym_final < 1e-2:
        return ("darkorange", "algorithmic-but-noisy")
    return ("gray", "intermediate")


def load(in_dir: Path) -> list[dict]:
    out = []
    for f in sorted(in_dir.glob("*.npz")):
        d = dict(np.load(f))
        d["seed"] = int(re.search(r"seed(\d+)", f.stem).group(1))
        d["color"], d["regime"] = classify(float(d["L_emp"][-1]), float(d["L_sym"][-1]))
        out.append(d)
    return out


def plot(in_dirs: list[Path], primes: list[int], out_file: Path) -> None:
    fig, axes = plt.subplots(1, len(in_dirs), figsize=(5.5 * len(in_dirs), 4.5),
                             squeeze=False)
    for ax, in_dir, p in zip(axes[0], in_dirs, primes):
        trajs = load(in_dir)
        for t in trajs:
            ax.semilogy(t["step"], np.clip(t["L_emp"], 1e-10, None),
                        color=t["color"], alpha=0.55, linewidth=1.2)
            ax.semilogy(t["step"], np.clip(t["L_sym"], 1e-10, None),
                        color=t["color"], alpha=0.55, linewidth=1.2,
                        linestyle="--")
        # Regime tally
        regs = {}
        for t in trajs:
            regs[t["regime"]] = regs.get(t["regime"], 0) + 1
        leg = ", ".join(f"{n}× {r}" for r, n in regs.items())
        ax.set_title(f"$p = {p}$  ({len(trajs)} seeds: {leg})", fontsize=10)
        ax.set_xlabel("training step")
        ax.set_ylabel("CE (log)")
        ax.set_ylim(1e-9, 20)
        ax.grid(True, alpha=0.3, which="both")
        # Custom legend
        handles = [
            plt.Line2D([], [], color="green", label="grokked"),
            plt.Line2D([], [], color="darkorange", label="algorithmic-but-noisy"),
            plt.Line2D([], [], color="red", label="CRT-fail"),
            plt.Line2D([], [], color="gray", label="intermediate"),
            plt.Line2D([], [], color="black", linestyle="-", label="$L_{\\rm emp}$"),
            plt.Line2D([], [], color="black", linestyle="--", label="$L_{\\rm sym}$"),
        ]
        ax.legend(handles=handles, fontsize=7, loc="lower left")
    fig.tight_layout()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, dpi=130, bbox_inches="tight")
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
