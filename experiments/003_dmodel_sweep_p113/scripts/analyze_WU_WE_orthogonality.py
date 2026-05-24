"""Test whether W_U (unembed) and W_E (embed) carry the same superposition
structure as the MLP write directions.

For each character k, compute:
  W_E_k = W_E[:, value-tokens] @ basis[cos_k]   # input direction
  W_U_k = W_U[:, value-tokens] @ basis[cos_k]   # read direction
  P_k   = MLP write direction (from full-grid char-product projection)

Then for each piggybacker target (e.g., k=20), measure how much of each direction
lies in the span of the dominants {k=3, k=10, k=51}.

If W_E_k and W_U_k for k=20 are *independent* (large perp / raw ratio), but P_k
is dependent (small ratio), then the model puts character-20 in W_E and W_U
along independent directions, but the MLP's writes are along shared dominant
directions. In that case, character-20's logit contribution must flow through
the SKIP path (W_E → resid_post directly, bypassing MLP), not through the MLP.

If all three show the same dependence pattern, then the unembed is *also*
superposed — and the algorithm requires consistent joint superposition.

Usage:
    python experiments/003_dmodel_sweep_p113/scripts/analyze_WU_WE_orthogonality.py \
        --run-dir experiments/003_dmodel_sweep_p113/runs/dmodel_24_dmlp_32_seed1 \
        --targets 20,51,6 --dominants 3,10
"""
from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import numpy as np
import torch

from crypto_interp.interp.bases import (
    multiplicative_fourier_basis,
    discrete_log_table,
)
from crypto_interp.interp.load import load_run


def order_of(k, n=112):
    return n // math.gcd(k, n)


def build_basis_indexed():
    basis, names, _ = multiplicative_fourier_basis(113)
    cos_idx, sin_idx = {}, {}
    for i, nm in enumerate(names):
        m = re.match(r"mul (cos|sin) (\d+)", nm)
        if m:
            kk = int(m.group(2))
            if m.group(1) == "cos":
                cos_idx[kk] = i
            else:
                sin_idx[kk] = i
    return basis.double(), cos_idx, sin_idx


def compute_mlp_grid(model, ds):
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


def char_dirs_from_matrix(W, basis, cos_idx, sin_idx, ks, p=113):
    """For weight matrix W of shape (d_model, vocab), compute the d_model
    directions associated with each character k's cos and sin components."""
    W_v = W[:, :p].double()
    dirs = {}
    for k in ks:
        cos_k = basis[cos_idx[k]].double()
        sin_k = basis[sin_idx[k]].double()
        dirs[(k, "cos")] = W_v @ cos_k
        dirs[(k, "sin")] = W_v @ sin_k
    return dirs


def mlp_dirs(f_ab, ks, p=113):
    """For MLP output, compute the cos and sin product directions for each k."""
    _, dlog = discrete_log_table(p)
    j_a = np.array([dlog[a] for a in range(1, p)])
    dirs = {}
    for k in ks:
        theta_a = 2 * np.pi * k * j_a / (p - 1)
        ref_cos = torch.tensor(np.cos(theta_a[:, None] + theta_a[None, :]),
                                dtype=torch.float64)
        ref_sin = torch.tensor(np.sin(theta_a[:, None] + theta_a[None, :]),
                                dtype=torch.float64)
        dirs[(k, "cos")] = torch.einsum("ab,abd->d", ref_cos, f_ab)
        dirs[(k, "sin")] = torch.einsum("ab,abd->d", ref_sin, f_ab)
    return dirs


def orth_against(target_vec, basis_vecs):
    """Gram-Schmidt: remove components of target_vec along basis_vecs."""
    v = target_vec.clone()
    units = []
    for u in basis_vecs:
        for prev in units:
            u = u - (u @ prev) * prev
        if u.norm() > 1e-10:
            units.append(u / u.norm())
    for u in units:
        v = v - (v @ u) * u
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--targets", required=True,
                    help="comma-separated piggybacker chars to analyze")
    ap.add_argument("--dominants", required=True,
                    help="comma-separated dominant chars to orthogonalize against")
    args = ap.parse_args()
    run_dir = Path(args.run_dir).resolve()
    ck = sorted(run_dir.glob("checkpoint_*.pt"))
    model, ds, _ = load_run(ck[-1])
    model.eval()
    p = ds.p
    basis, cos_idx, sin_idx = build_basis_indexed()
    targets = [int(x) for x in args.targets.split(",")]
    dominants = [int(x) for x in args.dominants.split(",")]
    all_ks = targets + dominants

    # Compute char dirs from W_E, W_U.
    W_E = model.embed.W_E.detach()
    W_U = model.unembed.W_U.detach()
    we_dirs = char_dirs_from_matrix(W_E, basis, cos_idx, sin_idx, all_ks, p)
    wu_dirs = char_dirs_from_matrix(W_U, basis, cos_idx, sin_idx, all_ks, p)
    # MLP write dirs.
    f_ab = compute_mlp_grid(model, ds)
    m_dirs = mlp_dirs(f_ab, all_ks, p)

    # Dominant basis (both cos and sin for each dominant) for each modality.
    print(f"\nOrthogonalizing targets {targets} against dominants {dominants}")
    print(f"Reporting: ||v_perp|| / ||v_raw||  for each modality and target.\n")
    print(f"{'k':>4} {'cos/sin':>7}  {'||W_E_k|| (raw)':>18}  {'||W_E_k (perp)||':>18}  "
          f"{'ratio':>7}  {'||W_U_k|| (raw)':>18}  {'||W_U_k (perp)||':>18}  "
          f"{'ratio':>7}  {'||P_k|| (raw)':>15}  {'||P_k (perp)||':>15}  {'ratio':>7}")
    print("-" * 170)
    for target in targets:
        for part in ("cos", "sin"):
            we_dom = [we_dirs[(d, p2)] for d in dominants for p2 in ("cos", "sin")]
            wu_dom = [wu_dirs[(d, p2)] for d in dominants for p2 in ("cos", "sin")]
            md_dom = [m_dirs[(d, p2)] for d in dominants for p2 in ("cos", "sin")]
            we_raw = we_dirs[(target, part)]
            wu_raw = wu_dirs[(target, part)]
            md_raw = m_dirs[(target, part)]
            we_perp = orth_against(we_raw, we_dom)
            wu_perp = orth_against(wu_raw, wu_dom)
            md_perp = orth_against(md_raw, md_dom)
            print(f"{target:>4} {part:>7}  "
                  f"{we_raw.norm().item():>18.3e}  {we_perp.norm().item():>18.3e}  "
                  f"{we_perp.norm().item()/max(we_raw.norm().item(),1e-30):>7.3f}  "
                  f"{wu_raw.norm().item():>18.3e}  {wu_perp.norm().item():>18.3e}  "
                  f"{wu_perp.norm().item()/max(wu_raw.norm().item(),1e-30):>7.3f}  "
                  f"{md_raw.norm().item():>15.3e}  {md_perp.norm().item():>15.3e}  "
                  f"{md_perp.norm().item()/max(md_raw.norm().item(),1e-30):>7.3f}")

    # Also compute for dominants vs each other (control: should be ratio ≈ 1).
    print(f"\n--- Control: dominants orthogonalized against the OTHER dominants ---")
    for target in dominants:
        others = [d for d in dominants if d != target]
        for part in ("cos", "sin"):
            we_dom = [we_dirs[(d, p2)] for d in others for p2 in ("cos", "sin")]
            wu_dom = [wu_dirs[(d, p2)] for d in others for p2 in ("cos", "sin")]
            md_dom = [m_dirs[(d, p2)] for d in others for p2 in ("cos", "sin")]
            we_raw = we_dirs[(target, part)]
            wu_raw = wu_dirs[(target, part)]
            md_raw = m_dirs[(target, part)]
            we_perp = orth_against(we_raw, we_dom)
            wu_perp = orth_against(wu_raw, wu_dom)
            md_perp = orth_against(md_raw, md_dom)
            print(f"{target:>4} {part:>7}  "
                  f"{we_raw.norm().item():>18.3e}  {we_perp.norm().item():>18.3e}  "
                  f"{we_perp.norm().item()/max(we_raw.norm().item(),1e-30):>7.3f}  "
                  f"{wu_raw.norm().item():>18.3e}  {wu_perp.norm().item():>18.3e}  "
                  f"{wu_perp.norm().item()/max(wu_raw.norm().item(),1e-30):>7.3f}  "
                  f"{md_raw.norm().item():>15.3e}  {md_perp.norm().item():>15.3e}  "
                  f"{md_perp.norm().item()/max(md_raw.norm().item(),1e-30):>7.3f}")


if __name__ == "__main__":
    main()
