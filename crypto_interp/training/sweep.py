"""Multi-seed parallel sweep driver.

Launches multiple training subprocesses (limited concurrency) and waits for
all to finish. Each child process invokes ``scripts/train.py`` with a per-seed
``--tag`` and ``--seed-override``. Lives in the library so the CLI is thin.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


def parse_seeds(spec: str) -> list[int]:
    """Parse '0..16' (inclusive range) or '0,1,2,5' into a list of seeds."""
    spec = spec.strip()
    if ".." in spec:
        lo, hi = spec.split("..")
        return list(range(int(lo), int(hi) + 1))
    return [int(s) for s in spec.split(",") if s.strip()]


@dataclass
class SweepArgs:
    experiment: str            # experiment id, e.g. "001_mul_p113"
    seeds: list[int]
    epochs: int
    metrics_every: int
    concurrency: int
    threads_per_job: int       # OMP_NUM_THREADS per child; 0 = unset
    save_every: int = 0        # 0 means use cfg.save_every; explicit lets sweep save only at end
    train_script: str = "scripts/train.py"


def _launch(seed: int, args: SweepArgs, runs_dir: Path) -> subprocess.Popen:
    tag = f"sweep_seed{seed}"
    cmd = [
        sys.executable, "-u", args.train_script,
        "--experiment", args.experiment,
        "--seed-override", str(seed),
        "--tag", tag,
        "--num-epochs", str(args.epochs),
        "--metrics-every", str(args.metrics_every),
        "--log-every", "500",
    ]
    if args.save_every > 0:
        cmd.extend(["--save-every", str(args.save_every)])
    log_path = runs_dir / f"{tag}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_fh = open(log_path, "w")
    env = os.environ.copy()
    if args.threads_per_job > 0:
        env["OMP_NUM_THREADS"] = str(args.threads_per_job)
        env["MKL_NUM_THREADS"] = str(args.threads_per_job)
    proc = subprocess.Popen(cmd, stdout=log_fh, stderr=subprocess.STDOUT, env=env)
    proc.log_fh = log_fh        # stash for cleanup
    proc.seed = seed
    proc.tag = tag
    print(f"  launched seed {seed} (pid {proc.pid}) → {runs_dir / tag}/")
    return proc


def run_sweep(args: SweepArgs, runs_dir: Path) -> tuple[list[int], list[int]]:
    """Run the sweep. Returns (completed_seeds, failed_seeds)."""
    runs_dir = Path(runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)
    print(f"Sweep: experiment={args.experiment}, seeds={args.seeds}, "
          f"epochs={args.epochs}, concurrency={args.concurrency}, "
          f"threads_per_job={args.threads_per_job}")

    pending = list(args.seeds)
    running: list[subprocess.Popen] = []
    completed: list[int] = []
    failed: list[int] = []
    start = time.time()

    while pending or running:
        while pending and len(running) < args.concurrency:
            running.append(_launch(pending.pop(0), args, runs_dir))
        time.sleep(2.0)
        still = []
        for p in running:
            rc = p.poll()
            if rc is None:
                still.append(p)
                continue
            p.log_fh.close()
            elapsed = time.time() - start
            if rc == 0:
                completed.append(p.seed)
                print(f"  [+{elapsed:6.0f}s] seed {p.seed} done. "
                      f"{len(completed)}/{len(args.seeds)} done.")
            else:
                failed.append(p.seed)
                print(f"  [+{elapsed:6.0f}s] seed {p.seed} FAILED (rc={rc}). "
                      f"See {runs_dir / (p.tag + '.log')}")
        running = still

    print(f"\nSweep complete in {time.time() - start:.0f}s.")
    print(f"  Completed: {sorted(completed)}")
    if failed:
        print(f"  Failed:    {sorted(failed)}")
    return completed, failed
