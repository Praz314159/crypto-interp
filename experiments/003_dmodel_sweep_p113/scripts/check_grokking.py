"""Scan training runs and report grokked / partial / failed status.

For every run dir under ``runs/`` that has a ``losses.pt``, apply
``crypto_interp.interp.grokking_status`` and print a table. As d_mlp is pushed
below the floor (the compressing-d_mlp sweep), this surfaces the runs that
memorize but never grok -- the interesting failure cases.

Usage:
    python experiments/003_dmodel_sweep_p113/scripts/check_grokking.py
    python experiments/003_dmodel_sweep_p113/scripts/check_grokking.py --grok-thresh 0.1
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from crypto_interp.interp import grokking_status

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"


def classify(status: dict) -> str:
    if status["grokked"]:
        return "grokked"
    if status["memorized_step"] is not None:
        return "memorized-only"   # fit train, never grokked => below the floor / failure
    return "no-fit"               # never even memorized


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", type=str, default=str(RUNS))
    ap.add_argument("--grok-thresh", type=float, default=0.1)
    ap.add_argument("--mem-thresh", type=float, default=0.1)
    args = ap.parse_args()
    runs_dir = Path(args.runs_dir)

    rows = []
    for d in sorted(runs_dir.iterdir()):
        lp = d / "losses.pt"
        if not d.is_dir() or not lp.exists():
            continue
        losses = torch.load(lp, weights_only=False)
        st = grokking_status(losses["train_losses"], losses["test_losses"],
                             mem_thresh=args.mem_thresh, grok_thresh=args.grok_thresh)
        rows.append((d.name, st, classify(st)))

    if not rows:
        print(f"No runs with losses.pt under {runs_dir}")
        return

    width = max(len(name) for name, _, _ in rows)
    print(f"{'run':<{width}}  {'memorized':>9}  {'cliff':>7}  {'final_test':>11}  status")
    print("-" * (width + 40))
    for name, st, cls in rows:
        mem = "-" if st["memorized_step"] is None else str(st["memorized_step"])
        cliff = "-" if st["cliff_step"] is None else str(st["cliff_step"])
        print(f"{name:<{width}}  {mem:>9}  {cliff:>7}  {st['final_test_loss']:>11.3e}  {cls}")

    n = len(rows)
    grok = sum(1 for _, _, c in rows if c == "grokked")
    mem_only = sum(1 for _, _, c in rows if c == "memorized-only")
    no_fit = sum(1 for _, _, c in rows if c == "no-fit")
    print(f"\n{n} runs: {grok} grokked, {mem_only} memorized-only (failure), {no_fit} no-fit")


if __name__ == "__main__":
    main()
