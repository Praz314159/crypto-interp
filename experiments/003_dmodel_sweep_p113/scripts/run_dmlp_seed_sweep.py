"""Cross-product sweep over (d_mlp, seed) at fixed d_model=24.

d_mlp ∈ {64, 128, 256} × seed ∈ {1, 2, 3}  = 9 new runs at 40K epochs each.
d_mlp=512 is omitted — seeds 1-17 already cover that condition.

Each run writes to runs/dmodel_24_dmlp_<M>_seed<S>/.

Usage:
    python experiments/003_dmodel_sweep_p113/scripts/run_dmlp_seed_sweep.py
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_ID = "003_dmodel_sweep_p113"
D_MODEL = 24
DEFAULT_D_MLPS = [64, 128, 256]
DEFAULT_SEEDS = [1, 2, 3]
DEFAULT_EPOCHS = 40_000


def run_one(d_mlp: int, seed: int, epochs: int, runs_dir: Path,
            experiment_id: str = EXPERIMENT_ID, p: int | None = None) -> int:
    # Prime-prefix the tag so multi-prime runs do not collide with p=113 runs.
    prefix = f"p{p}_" if p is not None else ""
    tag = f"{prefix}dmodel_{D_MODEL}_dmlp_{d_mlp}_seed{seed}"
    cmd = [
        sys.executable, "-u",
        str(REPO_ROOT / "scripts" / "train.py"),
        "--experiment", experiment_id,
        "--override", f"d_model={D_MODEL}",
        "--override", f"d_mlp={d_mlp}",
        "--seed-override", str(seed),
        "--num-epochs", str(epochs),
        "--tag", tag,
    ]
    if p is not None:
        cmd[cmd.index("--override"):cmd.index("--override")] = ["--override", f"p={p}"]
    log_path = runs_dir / f"{tag}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n=== d_mlp={d_mlp} seed={seed}{f' p={p}' if p else ''} → {tag} ===")
    start = time.time()
    with open(log_path, "w") as logf:
        proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT,
                                cwd=REPO_ROOT)
        rc = proc.wait()
    elapsed = time.time() - start
    print(f"  done in {elapsed:.0f}s — rc={rc}")
    return rc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d-mlps", type=str,
                    default=",".join(map(str, DEFAULT_D_MLPS)))
    ap.add_argument("--seeds", type=str,
                    default=",".join(map(str, DEFAULT_SEEDS)))
    ap.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    ap.add_argument("--p", type=int, default=None,
                    help="Override the prime (multi-prime sweeps). Omit for the config default (p=113).")
    ap.add_argument("--experiment", type=str, default=EXPERIMENT_ID,
                    help="Experiment id whose config.py to use.")
    args = ap.parse_args()
    d_mlps = [int(s) for s in args.d_mlps.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]

    runs_dir = REPO_ROOT / "experiments" / args.experiment / "runs"
    overall = time.time()
    results = []
    for m in d_mlps:
        for s in seeds:
            rc = run_one(m, s, args.epochs, runs_dir, experiment_id=args.experiment, p=args.p)
            results.append((m, s, rc))
    print(f"\nSweep done in {(time.time() - overall)/60:.1f} min.")
    for m, s, rc in results:
        print(f"  d_mlp={m} seed={s}: rc={rc}")


if __name__ == "__main__":
    main()
