"""Trajectory of theory-baseline quantities over training.

Tests the prediction: the symmetric kernel (the algorithmic identity) forms
*early* in training; "grokking" is the decay of the asymmetric noise term
on top of it, not the emergence of the algorithm itself.

Three quantities tracked at each checkpoint:
    L_empirical  — model's CE on the full nonzero (a,b) grid
    L_symmetric  — CE of the symmetric kernel (logits averaged at fixed offset)
    L_theory_K   — CE of K-truncated kernel using the K essential at that step

Three predictions:
    (1) L_symmetric drops before the L_empirical cliff (algorithm forms early)
    (2) L_theory_K tracks L_symmetric once K stabilizes (truncation is lossless)
    (3) L_empirical - L_symmetric = "asymmetric-noise budget"; decays through
        the grokking transition

Usage:
    python -m crypto_interp.analysis.theory_trajectory \\
        --run-dir experiments/003_dmodel_sweep_p113/runs/dmodel_24_dmlp_20_wd2_seed7

Or for a small population (figures saved to --out-dir):
    python -m crypto_interp.analysis.theory_trajectory \\
        --seeds 7 8 10 11 12 13 14 15 \\
        --runs-root experiments/003_dmodel_sweep_p113/runs \\
        --tag dmodel_24_dmlp_20_wd2 \\
        --out-dir experiments/theory_trajectory
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from crypto_interp.interp import Session
from crypto_interp.interp.load import load_run
from crypto_interp.interp.theory import theory_baseline


CKPT_RE = re.compile(r"checkpoint_(\d+)\.pt$")


def list_checkpoints(run_dir: Path) -> list[tuple[int, Path]]:
    """Return [(step, path), ...] sorted by step."""
    out = []
    for p in run_dir.glob("checkpoint_*.pt"):
        m = CKPT_RE.search(p.name)
        if m:
            out.append((int(m.group(1)), p))
    return sorted(out)


def analyze_trajectory(run_dir: Path) -> dict:
    """Compute theory-baseline numbers at every checkpoint of one run.

    Returns dict with arrays: step, L_emp, L_sym, L_K, K_size.
    """
    ckpts = list_checkpoints(run_dir)
    if not ckpts:
        raise FileNotFoundError(f"no checkpoints under {run_dir}")

    steps, L_emp, L_sym, L_K, K_sizes, K_sets = [], [], [], [], [], []
    for step, ckpt_path in ckpts:
        try:
            model, ds, _ = load_run(ckpt_path, device="cpu")
        except Exception as e:
            print(f"  step {step:>6}: load failed ({e})")
            continue
        S = Session(model, ds)
        try:
            r = theory_baseline(S)
        except Exception as e:
            print(f"  step {step:>6}: theory_baseline failed ({e})")
            continue
        steps.append(step)
        L_emp.append(r["L_empirical"])
        L_sym.append(r["L_symmetric"])
        L_K.append(r["L_theory_K"])
        K_sizes.append(len(r["K"]))
        K_sets.append(tuple(sorted(int(k) for k in r["K"])))
        print(f"  step {step:>6}: |K|={len(r['K']):>2} K={list(r['K'])} "
              f" L_emp={r['L_empirical']:.3g}  L_sym={r['L_symmetric']:.3g}  "
              f"L_K={r['L_theory_K']:.3g}")
    return {
        "step": np.array(steps),
        "L_emp": np.array(L_emp),
        "L_sym": np.array(L_sym),
        "L_K": np.array(L_K),
        "K_size": np.array(K_sizes),
        "K_sets": K_sets,
        "run_dir": str(run_dir),
    }


def plot_one(traj: dict, ax, *, title: str | None = None) -> None:
    ax.semilogy(traj["step"], np.clip(traj["L_emp"], 1e-9, None),
                "o-", label="L_empirical", color="C0", markersize=4)
    ax.semilogy(traj["step"], np.clip(traj["L_sym"], 1e-9, None),
                "s-", label="L_symmetric (algorithm)", color="C1", markersize=4)
    ax.semilogy(traj["step"], np.clip(traj["L_K"], 1e-9, None),
                "^-", label="L_theory_K (truncated)", color="C2", markersize=3)
    ax.set_xlabel("training step")
    ax.set_ylabel("CE (log scale)")
    ax.set_title(title or Path(traj["run_dir"]).name)
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=8, loc="lower left")


def plot_K_size(traj: dict, ax, *, title: str | None = None) -> None:
    ax.plot(traj["step"], traj["K_size"], "o-", color="C3", markersize=4)
    ax.set_xlabel("training step")
    ax.set_ylabel("|K|")
    ax.set_title("|K| over training")
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=None,
                    help="Single run directory (alternative to --seeds / --tag).")
    ap.add_argument("--runs-root", default="experiments/003_dmodel_sweep_p113/runs",
                    help="Used with --seeds and --tag.")
    ap.add_argument("--tag", default="dmodel_24_dmlp_20_wd2",
                    help="Run-name prefix; full name = '{tag}_seed{N}'.")
    ap.add_argument("--seeds", nargs="+", type=int, default=None,
                    help="Seed numbers to plot.")
    ap.add_argument("--out-dir", default="experiments/theory_trajectory",
                    help="Where to save figures and CSVs.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.run_dir:
        runs = [Path(args.run_dir)]
    elif args.seeds:
        runs = [Path(args.runs_root) / f"{args.tag}_seed{s}" for s in args.seeds]
    else:
        raise SystemExit("must provide --run-dir or --seeds")

    trajs = []
    for r in runs:
        if not r.exists():
            print(f"skip missing {r}")
            continue
        print(f"\n=== {r.name} ===")
        traj = analyze_trajectory(r)
        trajs.append(traj)
        # Save numeric arrays as .npz; save K_sets separately as a small text log
        np.savez(out_dir / f"{r.name}.npz", **{
            k: v for k, v in traj.items() if k not in ("run_dir", "K_sets")
        })
        with open(out_dir / f"{r.name}.K_trajectory.txt", "w") as f:
            for step, ks in zip(traj["step"], traj["K_sets"]):
                f.write(f"{int(step):>6}  K={list(ks)}\n")

    # Per-run combined figure (loss + K)
    for t in trajs:
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
        plot_one(t, axes[0], title=f"{Path(t['run_dir']).name}")
        plot_K_size(t, axes[1])
        fig.tight_layout()
        fig.savefig(out_dir / f"{Path(t['run_dir']).name}.png", dpi=120)
        plt.close(fig)

    # Population grid
    if len(trajs) > 1:
        cols = min(4, len(trajs))
        rows = (len(trajs) + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.2 * rows),
                                 squeeze=False)
        for i, t in enumerate(trajs):
            ax = axes[i // cols][i % cols]
            plot_one(t, ax, title=Path(t["run_dir"]).name)
        # Hide unused
        for j in range(len(trajs), rows * cols):
            axes[j // cols][j % cols].set_visible(False)
        fig.tight_layout()
        fig.savefig(out_dir / "population.png", dpi=120)
        plt.close(fig)

    print(f"\nWrote {len(trajs)} trajectory figure(s) + population grid to {out_dir}/")


if __name__ == "__main__":
    main()
