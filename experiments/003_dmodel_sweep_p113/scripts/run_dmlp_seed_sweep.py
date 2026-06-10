"""Cross-product sweep over (d_mlp, seed) at fixed d_model=24.

Launches ``scripts/train.py`` once per (d_mlp, seed) cell. Optional weight-decay
override (folded into the run tag as ``_wd<value>``) and parallel launching in
batches of ``--batch-size`` concurrent processes (default 1 = sequential).

Each run writes to runs/dmodel_24_dmlp_<M>[_wd<W>]_seed<S>/.

Examples:
    # 9-cell d_mlp x seed sweep, sequential (original behavior)
    python experiments/003_dmodel_sweep_p113/scripts/run_dmlp_seed_sweep.py

    # 15 wd=2 d_mlp=20 runs (seeds 7-21), 4 at a time (Colab/CUDA)
    python experiments/003_dmodel_sweep_p113/scripts/run_dmlp_seed_sweep.py \
        --d-mlps 20 --seeds 7-21 --weight-decay 2.0 \
        --epochs 20000 --batch-size 4
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


def parse_int_list(spec: str) -> list[int]:
    """Parse '20' or '7,8,9' or '7-21' (inclusive range) into a list of ints."""
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-")
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    return out


def make_tag(d_mlp: int, seed: int, wd: float | None, p: int | None) -> str:
    prefix = f"p{p}_" if p is not None else ""          # avoid multi-prime collisions
    wd_suffix = f"_wd{wd:g}" if wd is not None else ""  # 2.0 -> 'wd2', 0.5 -> 'wd0.5'
    return f"{prefix}dmodel_{D_MODEL}_dmlp_{d_mlp}{wd_suffix}_seed{seed}"


def build_cmd(d_mlp: int, seed: int, epochs: int, tag: str, *,
              metrics_every: int, wd: float | None, p: int | None) -> list[str]:
    cmd = [
        sys.executable, "-u",
        str(REPO_ROOT / "scripts" / "train.py"),
        "--experiment", EXPERIMENT_ID,
        "--override", f"d_model={D_MODEL}",
        "--override", f"d_mlp={d_mlp}",
        "--seed-override", str(seed),
        "--num-epochs", str(epochs),
        "--metrics-every", str(metrics_every),
        "--tag", tag,
    ]
    if wd is not None:
        cmd += ["--override", f"weight_decay={wd}"]
    if p is not None:
        cmd += ["--override", f"p={p}"]
    return cmd


def run_batches(jobs: list[tuple[int, int]], epochs: int, runs_dir: Path, *,
                batch_size: int, metrics_every: int, wd: float | None,
                p: int | None) -> list[tuple[str, int]]:
    """Run (d_mlp, seed) jobs with at most ``batch_size`` concurrent processes."""
    runs_dir.mkdir(parents=True, exist_ok=True)
    results: list[tuple[str, int]] = []
    for i in range(0, len(jobs), batch_size):
        batch = jobs[i:i + batch_size]
        procs = []
        for d_mlp, seed in batch:
            tag = make_tag(d_mlp, seed, wd, p)
            cmd = build_cmd(d_mlp, seed, epochs, tag,
                            metrics_every=metrics_every, wd=wd, p=p)
            logf = open(runs_dir / f"{tag}.log", "w")
            print(f"  launch {tag}")
            procs.append((tag, subprocess.Popen(cmd, stdout=logf,
                          stderr=subprocess.STDOUT, cwd=REPO_ROOT), logf))
        print(f"=== batch {i // batch_size + 1}: {len(procs)} runs in flight ===")
        start = time.time()
        for tag, proc, logf in procs:
            rc = proc.wait()
            logf.close()
            results.append((tag, rc))
            print(f"    {tag}: rc={rc}")
        print(f"  batch done in {(time.time() - start) / 60:.1f} min")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d-mlps", type=str, default=",".join(map(str, DEFAULT_D_MLPS)),
                    help="Comma list and/or inclusive ranges, e.g. '20' or '64,128' or '7-21'.")
    ap.add_argument("--seeds", type=str, default=",".join(map(str, DEFAULT_SEEDS)),
                    help="Comma list and/or inclusive ranges, e.g. '7-21'.")
    ap.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    ap.add_argument("--weight-decay", type=float, default=None,
                    help="Override weight_decay; folded into the tag as _wd<value>.")
    ap.add_argument("--batch-size", type=int, default=1,
                    help="Number of concurrent training processes (default 1 = sequential).")
    ap.add_argument("--metrics-every", type=int, default=50,
                    help="Per-frequency energy logging cadence (needed for basin-commitment analysis).")
    ap.add_argument("--p", type=int, default=None,
                    help="Override the prime (multi-prime sweeps). Omit for config default (p=113).")
    args = ap.parse_args()

    d_mlps = parse_int_list(args.d_mlps)
    seeds = parse_int_list(args.seeds)
    jobs = [(m, s) for m in d_mlps for s in seeds]
    runs_dir = REPO_ROOT / "experiments" / EXPERIMENT_ID / "runs"

    print(f"{len(jobs)} runs: d_mlps={d_mlps} seeds={seeds} "
          f"wd={args.weight_decay} epochs={args.epochs} batch_size={args.batch_size}")
    overall = time.time()
    results = run_batches(jobs, args.epochs, runs_dir,
                          batch_size=args.batch_size, metrics_every=args.metrics_every,
                          wd=args.weight_decay, p=args.p)
    print(f"\nSweep done in {(time.time() - overall) / 60:.1f} min.")
    n_fail = sum(1 for _, rc in results if rc != 0)
    if n_fail:
        print(f"  WARNING: {n_fail}/{len(results)} runs returned nonzero.")


if __name__ == "__main__":
    main()
