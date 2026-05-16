"""Multi-seed sweep driver for trajectory analysis.

Launches multiple training runs in parallel (limited concurrency), waits for all
to finish. Each run saves loss curves + per-frequency embedding energy snapshots
for trajectory analysis.

Usage:
    python3 sweep.py --task mul --seeds 0..16 --concurrency 4 --epochs 8000
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


def parse_seeds(spec: str) -> list[int]:
    """Parse '0..16' or '0,1,2,5' into a list of seeds."""
    if ".." in spec:
        lo, hi = spec.split("..")
        return list(range(int(lo), int(hi) + 1))
    return [int(s) for s in spec.split(",") if s.strip()]


def launch(seed: int, args) -> subprocess.Popen:
    tag = f"{args.task}_sweep_seed{seed}"
    cmd = [
        sys.executable, "-u", "train.py",
        "--task", args.task,
        "--p", str(args.p),
        "--frac-train", str(args.frac_train),
        "--seed", str(seed),
        "--num-epochs", str(args.epochs),
        "--tag", tag,
        "--log-every", "500",
        "--save-every", str(args.epochs),  # only save 1 checkpoint at end (less disk)
        "--metrics-every", str(args.metrics_every),
    ]
    log_path = Path("runs") / f"{tag}.log"
    log_path.parent.mkdir(exist_ok=True)
    log_fh = open(log_path, "w")
    env = os.environ.copy()
    if args.threads_per_job > 0:
        env["OMP_NUM_THREADS"] = str(args.threads_per_job)
        env["MKL_NUM_THREADS"] = str(args.threads_per_job)
    proc = subprocess.Popen(cmd, stdout=log_fh, stderr=subprocess.STDOUT, env=env)
    proc.log_fh = log_fh
    proc.seed = seed
    proc.tag = tag
    print(f"  launched seed {seed} (pid {proc.pid}) → runs/{tag}/")
    return proc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="mul")
    parser.add_argument("--p", type=int, default=113)
    parser.add_argument("--frac-train", type=float, default=0.3)
    parser.add_argument("--seeds", type=str, default="0..15",
                        help="seeds: e.g. '0..15' (inclusive range) or '0,1,2,5'")
    parser.add_argument("--epochs", type=int, default=8000,
                        help="num epochs per run (8000 is well past grokking for mul/p=113)")
    parser.add_argument("--metrics-every", type=int, default=50)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--threads-per-job", type=int, default=2,
                        help="OMP_NUM_THREADS per job; 0 = unset")
    args = parser.parse_args()

    seeds = parse_seeds(args.seeds)
    print(f"Sweep: task={args.task}, p={args.p}, seeds={seeds}, "
          f"epochs={args.epochs}, concurrency={args.concurrency}, "
          f"threads_per_job={args.threads_per_job}")

    pending = list(seeds)
    running: list[subprocess.Popen] = []
    completed = []
    failed = []
    start = time.time()

    while pending or running:
        # Launch as many as concurrency allows
        while pending and len(running) < args.concurrency:
            seed = pending.pop(0)
            running.append(launch(seed, args))
        # Poll
        time.sleep(2.0)
        still = []
        for p in running:
            rc = p.poll()
            if rc is None:
                still.append(p)
            else:
                p.log_fh.close()
                elapsed = time.time() - start
                if rc == 0:
                    completed.append(p.seed)
                    print(f"  [+{elapsed:6.0f}s] seed {p.seed} done. {len(completed)}/{len(seeds)} done.")
                else:
                    failed.append(p.seed)
                    print(f"  [+{elapsed:6.0f}s] seed {p.seed} FAILED (rc={rc}). See runs/{p.tag}.log")
        running = still

    print(f"\nSweep complete in {time.time() - start:.0f}s.")
    print(f"  Completed: {sorted(completed)}")
    if failed:
        print(f"  Failed:    {sorted(failed)}")


if __name__ == "__main__":
    main()
