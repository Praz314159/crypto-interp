"""Seed sweep at fixed d_model for experiment 003.

Trains the d_model_sweep config at one (or more) chosen ``d_model`` values
across a list of seeds, so we can ask: does the model reliably pick the same
key-frequency set, or does it depend on initialization?

Motivating question: at d_model=24 the seed-0 run picked
K = {1, 9, 23, 30, 56}, including the order-2 Legendre character. Is that a
robust attractor (capacity pressure → Legendre) or a lucky basin?

Usage:
    python experiments/003_dmodel_sweep_p113/scripts/run_seed_sweep.py
    python experiments/003_dmodel_sweep_p113/scripts/run_seed_sweep.py --seeds 1,2,3
    python experiments/003_dmodel_sweep_p113/scripts/run_seed_sweep.py --d-model 32 --seeds 1,2
    python experiments/003_dmodel_sweep_p113/scripts/run_seed_sweep.py --epochs 4000   # quick pilot

Each child writes a log to runs/dmodel_<N>_seed<S>.log and checkpoints/losses
under runs/dmodel_<N>_seed<S>/.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_ID = "003_dmodel_sweep_p113"
# Seeds 1..7 by default — seed 0 already has the long-running dmodel_24 run.
DEFAULT_SEEDS = [1, 2, 3, 4, 5, 6, 7]
DEFAULT_D_MODEL = 24
# d=24 at seed 0 groked at epoch ~21k; budget 40k to leave headroom for slow
# seeds. Cheap enough that overbudgeting doesn't matter much.
DEFAULT_EPOCHS = 40_000


def parse_list(spec: str) -> list[int]:
    return [int(s) for s in spec.split(",") if s.strip()]


def run_one(d_model: int, seed: int, epochs: int, runs_dir: Path,
            extra_args: list[str]) -> int:
    tag = f"dmodel_{d_model}_seed{seed}"
    cmd = [
        sys.executable, "-u",
        str(REPO_ROOT / "scripts" / "train.py"),
        "--experiment", EXPERIMENT_ID,
        "--override", f"d_model={d_model}",
        "--seed-override", str(seed),
        "--num-epochs", str(epochs),
        "--tag", tag,
    ]
    cmd += extra_args

    log_path = runs_dir / f"{tag}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n=== d_model={d_model} seed={seed} → {tag} (log: {log_path}) ===")
    print("  cmd:", " ".join(cmd))
    start = time.time()
    with open(log_path, "w") as logf:
        proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT,
                                cwd=REPO_ROOT)
        rc = proc.wait()
    elapsed = time.time() - start
    status = "ok" if rc == 0 else f"FAILED (rc={rc})"
    print(f"  done in {elapsed:.0f}s — {status}")
    return rc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--d-model", type=int, default=DEFAULT_D_MODEL,
                        help=f"d_model for every run in the sweep. "
                             f"Default: {DEFAULT_D_MODEL}.")
    parser.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)),
                        help=f"Comma-separated seeds. Default: "
                             f"{','.join(map(str, DEFAULT_SEEDS))}")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS,
                        help=f"num_epochs per run. Default: {DEFAULT_EPOCHS}.")
    args, extra = parser.parse_known_args()

    seeds = parse_list(args.seeds)
    runs_dir = REPO_ROOT / "experiments" / EXPERIMENT_ID / "runs"

    print(f"Sequential seed sweep at fixed d_model")
    print(f"  experiment: {EXPERIMENT_ID}")
    print(f"  d_model:    {args.d_model}")
    print(f"  seeds:      {seeds}")
    print(f"  epochs:     {args.epochs}")
    if extra:
        print(f"  extra args (passed through): {extra}")

    overall_start = time.time()
    results: list[tuple[int, int]] = []  # (seed, rc)
    for s in seeds:
        rc = run_one(args.d_model, s, args.epochs, runs_dir, extra)
        results.append((s, rc))

    total = time.time() - overall_start
    print(f"\nSweep complete in {total:.0f}s ({total/60:.1f} min).")
    print("Summary:")
    for s, rc in results:
        status = "ok" if rc == 0 else f"FAILED (rc={rc})"
        print(f"  seed={s}: {status}")

    failed = [s for s, rc in results if rc != 0]
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
