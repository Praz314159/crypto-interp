"""For each character k, compute the direct ablation contribution:
    Δ_k(a, b, c) = logit_full(c|a, b) - logit_ablated(c|a, b)
where ablated removes χ_k from W_E. This is the EXACT contribution of
character k in W_E to each logit, with no projection assumptions.

If k's contribution is a clean character product χ_k(ab)·χ_k(c)^*, then
Δ_k(a, b, c) reduces to a 1D cosine at frequency k in
Δlog = (log c - log ab) mod (p-1).

For each k we visualize:
  - Δ_k(ab, c) heatmap, dlog-sorted on both axes (character product ⇒ stripes).
  - 1D reduction: average Δ_k along constant Δlog, vs the reference cosine.

All analysis primitives come from crypto_interp.interp (prime-parametric).

Usage:
    python experiments/003_dmodel_sweep_p113/scripts/analyze_ablation_delta.py \
        --run-dir experiments/003_dmodel_sweep_p113/runs/dmodel_24_dmlp_32_seed1 \
        --ks 3,10,20,51,6
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from crypto_interp.interp import (
    char_index,
    compute_logits_grid,
    correlate,
    delta_k,
    discrete_log_table,
    load_run,
    order_of,
    reduce_to_ab,
    reduce_to_diff,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--ks", required=True)
    args = ap.parse_args()
    run_dir = Path(args.run_dir).resolve()
    ck = sorted(run_dir.glob("checkpoint_*.pt"))
    print(f"Loading {ck[-1]}")
    model, ds, _ = load_run(ck[-1])
    model.eval()
    p = ds.p

    basis, ci = char_index(p)
    logits_full = compute_logits_grid(model, ds)

    _, dlog = discrete_log_table(p)
    order_idx = np.argsort([dlog[a] for a in range(1, p)])

    ks = [int(x) for x in args.ks.split(",")]
    print(f"\n{'k':>4} {'order':>5}  {'1D-corr w/ ref cos':>20}  {'||Δ_k|| total':>14}")
    for k in ks:
        delta = delta_k(model, ds, ci, basis, k, logits_full=logits_full)  # (p-1,p-1,vocab)

        # 1D reduction over Δlog (full-vocab columns: column index == value c).
        f_diff = reduce_to_diff(delta, p, value_axis="full")
        # 2D f(ab, c) for the heatmap, restricted to value tokens c=1..p-1.
        f_vis = reduce_to_ab(delta, p)[:, 1:p]

        delta_log = np.arange(p - 1)
        ref_1d = np.cos(2 * np.pi * k * delta_log / (p - 1))
        corr_1d = correlate(f_diff, ref_1d)
        total_norm = float(np.linalg.norm(delta))
        print(f"{k:>4} {order_of(k, p):>5}  {corr_1d:>+20.3f}  {total_norm:>14.3e}")

        f_vis_sorted = f_vis[order_idx][:, order_idx]
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        vmax = max(abs(f_vis.max()), abs(f_vis.min()))
        axes[0].imshow(f_vis, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="equal", origin="upper")
        axes[0].set_xlabel("c"); axes[0].set_ylabel("ab")
        axes[0].set_title(f"Δ_{k}(ab, c) natural index")

        axes[1].imshow(f_vis_sorted, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="equal", origin="upper")
        axes[1].set_xlabel("c (dlog-sorted)"); axes[1].set_ylabel("ab (dlog-sorted)")
        axes[1].set_title(f"Δ_{k}(ab, c) dlog-sorted — character-product ⇒ anti-diagonal stripes")

        ax = axes[2]
        ax.plot(delta_log, f_diff, "b-", lw=1.6, label=f"empirical Δ_{k}(Δlog)")
        scale = (f_diff @ ref_1d) / (ref_1d @ ref_1d + 1e-12)
        ax.plot(delta_log, scale * ref_1d, "r--", lw=1.4,
                label=f"{scale:.2e} · cos[2π·{k}·Δlog/(p-1)]")
        ax.set_xlabel("Δlog = (log c - log ab) mod (p-1)")
        ax.set_ylabel(f"avg Δ_{k}")
        ax.set_title(f"(c) 1D reduction; corr w/ ref = {corr_1d:+.3f}")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

        fig.suptitle(f"Direct ablation contribution Δ_{k}(ab, c) for k={k} "
                     f"(order {order_of(k, p)})", fontsize=11)
        fig.tight_layout()
        out = run_dir / f"delta_k{k}.png"
        fig.savefig(out, dpi=130, bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    main()
