"""Make a clean per-prime population trajectory plot.

Reads the .npz files written by theory_trajectory.py (each has step, L_emp,
L_sym, L_K, K_size for one seed) and produces a grid plot with regime
labels, color-coded by outcome.

Usage:
    # One prime per output PNG, large panels:
    python -m crypto_interp.analysis.trajectory_population_plot \\
        --in-dir experiments/theory_trajectory \\
        --out-file experiments/theory_trajectory/population_p113.png \\
        --title "p = 113, d_mlp = 20, wd = 2"

    # Combined view across primes:
    python -m crypto_interp.analysis.trajectory_population_plot \\
        --in-dir experiments/theory_trajectory experiments/theory_trajectory_p127 experiments/theory_trajectory_p181 \\
        --prime-labels 113 127 181 \\
        --out-file experiments/theory_trajectory/all_primes.png \\
        --title "Trajectories across primes"
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


CKPT_RE = re.compile(r"seed(\d+)")


def classify_trajectory(L_emp_final: float, L_sym_final: float) -> tuple[str, str]:
    """(color, regime_label) based on final losses."""
    GROK_T = 1e-2
    if L_sym_final > 0.3 and L_emp_final > 0.3 and L_emp_final - L_sym_final < 0.3:
        return ("red", "CRT-fail")
    if L_emp_final < GROK_T:
        return ("green", "grokked")
    if L_sym_final < GROK_T:
        return ("darkorange", "algorithmic-but-noisy")
    return ("gray", "intermediate")


def load_trajs(in_dirs: list[Path]) -> list[dict]:
    out = []
    for d in in_dirs:
        for f in sorted(d.glob("*.npz")):
            m = CKPT_RE.search(f.stem)
            if not m:
                continue
            data = dict(np.load(f))
            data["seed"] = int(m.group(1))
            data["name"] = f.stem
            data["color"], data["regime"] = classify_trajectory(
                float(data["L_emp"][-1]), float(data["L_sym"][-1]))
            out.append(data)
    return out


def grid_panels(trajs: list[dict], title: str, out_file: Path,
                ymin: float = 1e-9, ymax: float = 20.0) -> None:
    n = len(trajs)
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.0 * cols, 3.0 * rows),
                             squeeze=False)
    for i, t in enumerate(trajs):
        ax = axes[i // cols][i % cols]
        ax.semilogy(t["step"], np.clip(t["L_emp"], ymin, None), "o-",
                    color="C0", markersize=3, label="$L_{\\rm emp}$")
        ax.semilogy(t["step"], np.clip(t["L_sym"], ymin, None), "s-",
                    color="C1", markersize=3, label="$L_{\\rm sym}$")
        ax.semilogy(t["step"], np.clip(t["L_K"], ymin, None), "^-",
                    color="C2", markersize=2.5, label="$L_{\\theta,K}$")
        ax.set_title(f"seed {t['seed']} — {t['regime']}",
                     color=t["color"], fontsize=10, fontweight="bold")
        ax.set_xlabel("step")
        ax.set_ylabel("CE (log)")
        ax.set_ylim(ymin, ymax)
        ax.grid(True, alpha=0.3, which="both")
        if i == 0:
            ax.legend(fontsize=7, loc="lower left")
    # Hide unused
    for j in range(n, rows * cols):
        axes[j // cols][j % cols].set_visible(False)
    fig.suptitle(title, fontsize=13, y=1.00)
    fig.tight_layout()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_file}  ({n} seeds)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", nargs="+", required=True,
                    help="Directory or directories of theory_trajectory .npz files.")
    ap.add_argument("--out-file", required=True)
    ap.add_argument("--title", default="Trajectories")
    args = ap.parse_args()

    in_dirs = [Path(p) for p in args.in_dir]
    trajs = load_trajs(in_dirs)
    if not trajs:
        raise SystemExit("no trajectories found")
    trajs.sort(key=lambda t: (t["regime"], t["seed"]))
    grid_panels(trajs, args.title, Path(args.out_file))


if __name__ == "__main__":
    main()
