"""Ablate individual characters from W_E at inference time; measure test-loss impact.

For each character k in K, zero out W_E's projection onto cos_k/sin_k and recompute
test loss. Ablations that do no harm are "vestigial"; ablations that hurt are
load-bearing/essential. Uses crypto_interp.interp.essential_characters
(prime-parametric).

Usage:
    python experiments/003_dmodel_sweep_p113/scripts/analyze_vestigial_ablation.py \
        --run-dir experiments/003_dmodel_sweep_p113/runs/dmodel_24_dmlp_32_seed1
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from crypto_interp.interp import char_index, essential_characters, load_run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args()
    run_dir = Path(args.run_dir).resolve()
    ck = sorted(run_dir.glob("checkpoint_*.pt"))
    if not ck:
        raise SystemExit(f"No checkpoint in {run_dir}")
    print(f"Loading {ck[-1]}")
    model, ds, _ = load_run(ck[-1])
    model.eval()
    basis, ci = char_index(ds.p)

    res = essential_characters(model, ds, ci, basis)
    K, base, per = res["K"], res["base_loss"], res["per_char"]
    energies = ", ".join(f"k={k}: {per[k]['energy']:.3f}" for k in K)
    print(f"K = {K}")
    print(f"per-K energies: {energies}")
    print(f"\nBaseline test loss: {base:.4e}")

    print(f"\n{'k':>4} {'order':>5} {'energy':>10} {'ablated test':>14} "
          f"{'Δlog10':>8}  interpretation")
    for k in K:
        v = per[k]
        print(f"{k:>4} {v['order']:>5} {v['energy']:>10.3f} {v['ablated_loss']:>14.4e} "
              f"{v['dlog10']:>+8.3f}  {v['cls']}")

    # Control: ablate the strongest non-K character.
    non_K = [k for k in ci.freqs if k not in set(K)]
    ctrl_k = max(non_K, key=lambda k: per[k]["energy"])
    cv = per[ctrl_k]
    print(f"\nControl: ablate top non-K char k={ctrl_k} (energy {cv['energy']:.3f}): "
          f"test loss {cv['ablated_loss']:.4e}  Δlog10={cv['dlog10']:+.3f}")


if __name__ == "__main__":
    main()
