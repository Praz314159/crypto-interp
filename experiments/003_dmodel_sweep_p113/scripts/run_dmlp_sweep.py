"""d_mlp sweep at fixed d_model for experiment 003.

Tests the prediction from the Legendre-channel note (§3): under tight ``d_mlp``,
the model should shift K toward low-order characters, because cost-per-tile
scales with character order (~100 tiles per primitive, ~4 per order-4, 2 per
Legendre).

Holding ``d_model = 24`` (the smallest groking width in the d_model sweep) and
shrinking ``d_mlp`` from the default 512:
  - At d_mlp = 256 we predict the same K minus the non-load-bearing cluster
    (e.g. k=1 at seed 0), since 4 essential clusters easily fit in 256.
  - As d_mlp drops further, primitive clusters should be evicted first in
    favor of low-order characters (order-4, order-8, Legendre).
  - Below some threshold, grokking should fail.

Usage:
    python experiments/003_dmodel_sweep_p113/scripts/run_dmlp_sweep.py
    python experiments/003_dmodel_sweep_p113/scripts/run_dmlp_sweep.py --d-mlps 256,128
    python experiments/003_dmodel_sweep_p113/scripts/run_dmlp_sweep.py --d-model 32

Each child writes a log to runs/dmodel_<N>_dmlp_<M>.log and checkpoints under
runs/dmodel_<N>_dmlp_<M>/.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_ID = "003_dmodel_sweep_p113"
DEFAULT_D_MODEL = 24
# Coarse log-ish sweep from 256 down to 32. 512 is the d=24 baseline already.
DEFAULT_D_MLPS = [256, 128, 96, 64, 48, 32]
# Larger budget than the seed sweep — small d_mlp may grok slowly or not at all.
DEFAULT_EPOCHS = 60_000


def parse_list(spec: str) -> list[int]:
    return [int(s) for s in spec.split(",") if s.strip()]


def run_one(d_model: int, d_mlp: int, epochs: int, runs_dir: Path,
            extra_args: list[str]) -> int:
    tag = f"dmodel_{d_model}_dmlp_{d_mlp}"
    cmd = [
        sys.executable, "-u",
        str(REPO_ROOT / "scripts" / "train.py"),
        "--experiment", EXPERIMENT_ID,
        "--override", f"d_model={d_model}",
        "--override", f"d_mlp={d_mlp}",
        "--num-epochs", str(epochs),
        "--tag", tag,
    ]
    cmd += extra_args

    log_path = runs_dir / f"{tag}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n=== d_model={d_model} d_mlp={d_mlp} → {tag} (log: {log_path}) ===")
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
                        help=f"d_model held fixed. Default: {DEFAULT_D_MODEL}.")
    parser.add_argument("--d-mlps", default=",".join(map(str, DEFAULT_D_MLPS)),
                        help=f"Comma-separated d_mlp values. Default: "
                             f"{','.join(map(str, DEFAULT_D_MLPS))}")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS,
                        help=f"num_epochs per run. Default: {DEFAULT_EPOCHS}.")
    args, extra = parser.parse_known_args()

    d_mlps = parse_list(args.d_mlps)
    runs_dir = REPO_ROOT / "experiments" / EXPERIMENT_ID / "runs"

    print(f"Sequential d_mlp sweep at fixed d_model")
    print(f"  experiment: {EXPERIMENT_ID}")
    print(f"  d_model:    {args.d_model}")
    print(f"  d_mlps:     {d_mlps}")
    print(f"  epochs:     {args.epochs}")
    if extra:
        print(f"  extra args (passed through): {extra}")

    overall_start = time.time()
    results: list[tuple[int, int]] = []  # (d_mlp, rc)
    for m in d_mlps:
        rc = run_one(args.d_model, m, args.epochs, runs_dir, extra)
        results.append((m, rc))

    total = time.time() - overall_start
    print(f"\nSweep complete in {total:.0f}s ({total/60:.1f} min).")
    print("Summary:")
    for m, rc in results:
        status = "ok" if rc == 0 else f"FAILED (rc={rc})"
        print(f"  d_mlp={m}: {status}")

    failed = [m for m, rc in results if rc != 0]
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
