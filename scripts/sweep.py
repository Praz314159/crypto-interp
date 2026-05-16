"""Multi-seed parallel sweep for an experiment.

Usage:
    python scripts/sweep.py --experiment 001_mul_p113 --seeds 0..15 --concurrency 4

Each child seed becomes its own subprocess (``python scripts/train.py
--experiment <id> --seed-override <seed> --tag sweep_seed<seed>``).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from crypto_interp.training import parse_seeds, run_sweep
from crypto_interp.training.sweep import SweepArgs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True,
                        help="Experiment id under experiments/ (e.g. 001_mul_p113).")
    parser.add_argument("--seeds", default="0..15",
                        help="Seeds as range '0..15' or list '0,1,2,5'.")
    parser.add_argument("--epochs", type=int, default=8000,
                        help="Num epochs per run.")
    parser.add_argument("--metrics-every", type=int, default=50,
                        help="Record per-frequency embedding energy every N epochs.")
    parser.add_argument("--save-every", type=int, default=8000,
                        help="Save checkpoint every N epochs (default = end-of-run only).")
    parser.add_argument("--concurrency", type=int, default=4,
                        help="Max parallel subprocesses.")
    parser.add_argument("--threads-per-job", type=int, default=2,
                        help="OMP_NUM_THREADS per child; 0 = unset.")
    args = parser.parse_args()

    runs_dir = REPO_ROOT / "experiments" / args.experiment / "runs"
    seeds = parse_seeds(args.seeds)

    sweep_args = SweepArgs(
        experiment=args.experiment,
        seeds=seeds,
        epochs=args.epochs,
        metrics_every=args.metrics_every,
        save_every=args.save_every,
        concurrency=args.concurrency,
        threads_per_job=args.threads_per_job,
        train_script=str(REPO_ROOT / "scripts" / "train.py"),
    )
    run_sweep(sweep_args, runs_dir)


if __name__ == "__main__":
    main()
