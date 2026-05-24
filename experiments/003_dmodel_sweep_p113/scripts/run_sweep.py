"""Sequential d_model sweep driver for experiment 003.

Iterates over a list of d_model values, calling ``scripts/train.py`` once
per value with ``--override d_model=<N> --tag dmodel_<N>``. Runs are strictly
sequential (one after another) — this is for CPU-bound work where
parallelism would just thrash and where we want clean per-run logs.

Usage:
    python experiments/003_dmodel_sweep_p113/scripts/run_sweep.py
    python experiments/003_dmodel_sweep_p113/scripts/run_sweep.py --d-models 16,24,32
    python experiments/003_dmodel_sweep_p113/scripts/run_sweep.py --epochs 4000   # quick pilot

Each child writes a log to runs/dmodel_<N>.log and checkpoints/losses
under runs/dmodel_<N>/.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_ID = "003_dmodel_sweep_p113"
DEFAULT_D_MODELS = [16, 24, 32, 64]


def parse_list(spec: str) -> list[int]:
    """Parse '16,24,32' into [16, 24, 32]."""
    return [int(s) for s in spec.split(",") if s.strip()]


def run_one(d_model: int, epochs: int | None, runs_dir: Path, extra_args: list[str]) -> int:
    """Run a single child training. Returns the return code."""
    tag = f"dmodel_{d_model}"
    cmd = [
        sys.executable, "-u",
        str(REPO_ROOT / "scripts" / "train.py"),
        "--experiment", EXPERIMENT_ID,
        "--override", f"d_model={d_model}",
        "--tag", tag,
    ]
    if epochs is not None:
        cmd += ["--num-epochs", str(epochs)]
    cmd += extra_args

    log_path = runs_dir / f"{tag}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n=== d_model={d_model} → {tag} (log: {log_path}) ===")
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
    parser.add_argument("--d-models", default=",".join(map(str, DEFAULT_D_MODELS)),
                        help=f"Comma-separated d_model values. Default: "
                             f"{','.join(map(str, DEFAULT_D_MODELS))}")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override num_epochs for every child (handy for pilots). "
                             "Default: use CONFIG.num_epochs (20000).")
    args, extra = parser.parse_known_args()

    d_models = parse_list(args.d_models)
    runs_dir = REPO_ROOT / "experiments" / EXPERIMENT_ID / "runs"

    print(f"Sequential d_model sweep")
    print(f"  experiment: {EXPERIMENT_ID}")
    print(f"  d_models:   {d_models}")
    print(f"  epochs:     {args.epochs if args.epochs else 'CONFIG default (20000)'}")
    if extra:
        print(f"  extra args (passed through): {extra}")

    overall_start = time.time()
    results: list[tuple[int, int]] = []  # (d_model, rc)
    for d in d_models:
        rc = run_one(d, args.epochs, runs_dir, extra)
        results.append((d, rc))

    total = time.time() - overall_start
    print(f"\nSweep complete in {total:.0f}s ({total/60:.1f} min).")
    print("Summary:")
    for d, rc in results:
        status = "ok" if rc == 0 else f"FAILED (rc={rc})"
        print(f"  d_model={d}: {status}")

    failed = [d for d, rc in results if rc != 0]
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
