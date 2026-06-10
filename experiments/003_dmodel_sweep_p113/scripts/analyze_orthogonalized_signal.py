"""Visualize character-k product signal in the MLP output, after orthogonalizing
against the larger-magnitude character directions.

Problem: with d_mlp=32, the per-character "write directions" {g_k} in
d_model space are NOT orthogonal. A small-magnitude character like k=20
gets visually swamped by the leakage from k=3 or k=10 when we project naively.

Fix: Gram-Schmidt orthogonalize. For each target k, project f(a, b) onto the
component of g_k orthogonal to the dominant {g_3, g_10, g_51, ...} directions.

Usage:
    python experiments/003_dmodel_sweep_p113/scripts/analyze_orthogonalized_signal.py \
        --run-dir experiments/003_dmodel_sweep_p113/runs/dmodel_24_dmlp_32_seed1 \
        --target 20 --dominants 3,10,51
"""
from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from crypto_interp.interp.bases import (
    multiplicative_fourier_basis,
    discrete_log_table,
)
from crypto_interp.interp.load import load_run


def order_of(k, n=112):
    return n // math.gcd(k, n)


def compute_mlp_output_grid(model, ds):
    p = ds.p
    aa, bb = torch.meshgrid(torch.arange(1, p), torch.arange(1, p), indexing="ij")
    eq = torch.full_like(aa, ds.eq_token)
    inputs = torch.stack([aa, bb, eq], dim=-1).reshape(-1, 3)
    cache = {}
    model.remove_all_hooks()
    model.cache_all(cache)
    with torch.no_grad():
        _ = model(inputs)
    model.remove_all_hooks()
    return cache["blocks.0.hook_mlp_out"][:, -1, :].double().reshape(p - 1, p - 1, -1)


def reference_cos(k, p=113):
    _, dlog = discrete_log_table(p)
    j_a = np.array([dlog[a] for a in range(1, p)])
    theta_a = 2 * np.pi * k * j_a / (p - 1)
    return np.cos(theta_a[:, None] + theta_a[None, :])


def g_k(f_ab, k, p=113):
    """Intrinsic d_model direction for character-k product (cos part)."""
    ref = torch.tensor(reference_cos(k, p), dtype=torch.float64)
    return torch.einsum("ab,abd->d", ref, f_ab)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--target", type=int, required=True)
    ap.add_argument("--dominants", type=str, required=True,
                    help="comma-separated char indices to orthogonalize against")
    args = ap.parse_args()
    run_dir = Path(args.run_dir).resolve()
    ck = sorted(run_dir.glob("checkpoint_*.pt"))
    model, ds, _ = load_run(ck[-1])
    model.eval()
    p = ds.p
    f_ab = compute_mlp_output_grid(model, ds)

    target = args.target
    dominants = [int(x) for x in args.dominants.split(",")]
    print(f"Target: k={target}; orthogonalizing against k ∈ {dominants}")

    g_target = g_k(f_ab, target, p)
    # Also orthogonalize against sin counterparts: build using sin reference.
    _, dlog = discrete_log_table(p)
    j_a = np.array([dlog[a] for a in range(1, p)])
    sin_refs = {}
    for k in dominants + [target]:
        theta_a = 2 * np.pi * k * j_a / (p - 1)
        sin_refs[k] = torch.tensor(np.sin(theta_a[:, None] + theta_a[None, :]),
                                    dtype=torch.float64)
    g_sin_target = torch.einsum("ab,abd->d", sin_refs[target], f_ab)

    # Build Gram-Schmidt basis from dominants (both cos and sin products).
    basis_vecs = []
    for k in dominants:
        gk = g_k(f_ab, k, p)
        gsk = torch.einsum("ab,abd->d", sin_refs[k], f_ab)
        for v in (gk, gsk):
            for u in basis_vecs:
                v = v - (v @ u) * u
            if v.norm() > 1e-10:
                basis_vecs.append(v / v.norm())

    # Project out the dominant basis from g_target and g_sin_target.
    g_t_perp = g_target.clone()
    g_s_perp = g_sin_target.clone()
    for u in basis_vecs:
        g_t_perp = g_t_perp - (g_t_perp @ u) * u
        g_s_perp = g_s_perp - (g_s_perp @ u) * u

    print(f"||g_{target} (raw)||       = {g_target.norm().item():.3e}")
    print(f"||g_{target} (perp)||      = {g_t_perp.norm().item():.3e}")
    print(f"||g_sin_{target} (raw)||   = {g_sin_target.norm().item():.3e}")
    print(f"||g_sin_{target} (perp)||  = {g_s_perp.norm().item():.3e}")
    # The ratio tells us how much of the target's direction is independent of
    # dominants.

    # Plot 3 panels:
    # (a) raw projection onto g_target (the "before" visualization)
    # (b) orthogonalized projection (the "after")
    # (c) reference cos[θ_k(a)+θ_k(b)]
    g_raw_unit = g_target / (g_target.norm() + 1e-12)
    g_perp_unit = g_t_perp / (g_t_perp.norm() + 1e-12)
    sig_raw = torch.einsum("abd,d->ab", f_ab, g_raw_unit).cpu().numpy()
    sig_perp = torch.einsum("abd,d->ab", f_ab, g_perp_unit).cpu().numpy()
    ref = reference_cos(target, p)

    def correlate(x, y):
        x = x.flatten() - x.mean()
        y = y.flatten() - y.mean()
        return float(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-12))

    corr_raw = correlate(sig_raw, ref)
    corr_perp = correlate(sig_perp, ref)
    print(f"\nCorrelation w/ reference:")
    print(f"  raw projection  (onto g_{target}):           {corr_raw:+.3f}")
    print(f"  orthogonalized  (after subtracting dominants): {corr_perp:+.3f}")

    order_idx = np.argsort([dlog[a] for a in range(1, p)])
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    panels = [
        ("raw", sig_raw, f"raw proj onto g_{target} (corr={corr_raw:+.3f})"),
        ("perp", sig_perp, f"after ⟂ against {dominants} (corr={corr_perp:+.3f})"),
        ("ref", ref, f"reference cos[θ_{target}(a)+θ_{target}(b)]"),
    ]
    for col, (key, data, title) in enumerate(panels):
        for row, name in enumerate(["natural", "dlog-sorted"]):
            d = data if name == "natural" else data[order_idx][:, order_idx]
            ax = axes[row, col]
            vmax = max(abs(d.max()), abs(d.min()))
            ax.imshow(d, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                      aspect="equal", origin="upper")
            ax.set_title(f"{title}\n[{name}]", fontsize=8)
    fig.suptitle(
        f"Orthogonalized character-{target} signal "
        f"(order {order_of(target)}) at d_mlp=32 seed=1\n"
        f"||g_perp||/||g_raw|| = {g_t_perp.norm().item()/g_target.norm().item():.3f}",
        fontsize=11,
    )
    fig.tight_layout()
    out = run_dir / f"orthogonalized_k{target}_vs_{'_'.join(map(str, dominants))}.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out.name}")


if __name__ == "__main__":
    main()
