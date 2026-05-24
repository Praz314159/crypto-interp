"""Decompose the MLP output at frequency pair (k, k) into character-product vs
character-quotient components.

At each frequency (k, k), the 4 bilinear basis functions are:
  CC: cos θ_k(a) cos θ_k(b)
  CS: cos θ_k(a) sin θ_k(b)
  SC: sin θ_k(a) cos θ_k(b)
  SS: sin θ_k(a) sin θ_k(b)

Two algebraic combinations are special:
  Product (real):   CC - SS = cos[θ_k(a) + θ_k(b)] = cos θ_k(ab)
  Product (imag):   CS + SC = sin[θ_k(a) + θ_k(b)] = sin θ_k(ab)
  Quotient (real):  CC + SS = cos[θ_k(a) - θ_k(b)] = cos θ_k(a/b)
  Quotient (imag):  SC - CS = sin[θ_k(a) - θ_k(b)] = sin θ_k(a/b)

Decompose E[k, k] (the sum over d_model of squared coefficients in each basis)
into product vs quotient parts and report.

Usage:
    python experiments/003_dmodel_sweep_p113/scripts/analyze_kk_decomposition.py \
        --run-dir experiments/003_dmodel_sweep_p113/runs/dmodel_24_dmlp_32_seed1 \
        --ks 3,10,20,51,6
"""
from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import numpy as np
import torch

from crypto_interp.interp.bases import multiplicative_fourier_basis
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
    return basis.double(), names, cos_idx, sin_idx


def compute_mlp_output_grid(model, ds):
    p = ds.p
    a_grid = torch.arange(1, p)
    b_grid = torch.arange(1, p)
    aa, bb = torch.meshgrid(a_grid, b_grid, indexing="ij")
    eq = torch.full_like(aa, ds.eq_token)
    inputs = torch.stack([aa, bb, eq], dim=-1).reshape(-1, 3)
    cache = {}
    model.remove_all_hooks()
    model.cache_all(cache)
    with torch.no_grad():
        _ = model(inputs)
    model.remove_all_hooks()
    mlp_out = cache["blocks.0.hook_mlp_out"][:, -1, :].double()
    return mlp_out.reshape(p - 1, p - 1, -1)


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
    basis, names, cos_idx, sin_idx = build_basis_indexed()

    f_ab = compute_mlp_output_grid(model, ds)            # (p-1, p-1, d_model)
    p = ds.p
    basis_v = basis[:, 1:p].double()                     # restrict to value tokens

    # Project onto each pair of basis rows (cos_a, cos_b), etc.
    # coef[r1, r2, d] = sum_{a,b} basis[r1,a] * basis[r2,b] * f[a,b,d]
    print(f"\n{'k':>4} {'o':>4}  "
          f"{'E[k,k]':>10}  {'E_prod':>10}  {'E_quot':>10}  "
          f"{'%prod':>6}  {'%quot':>6}")
    for k_str in args.ks.split(","):
        k = int(k_str)
        ic, isn = cos_idx[k], sin_idx[k]
        b_cos = basis_v[ic]                              # (p-1,)
        b_sin = basis_v[isn]
        # 4 components.
        CC = torch.einsum("a,b,abd->d", b_cos, b_cos, f_ab)
        CS = torch.einsum("a,b,abd->d", b_cos, b_sin, f_ab)
        SC = torch.einsum("a,b,abd->d", b_sin, b_cos, f_ab)
        SS = torch.einsum("a,b,abd->d", b_sin, b_sin, f_ab)
        # Product (CC-SS for cos[θ_a+θ_b], CS+SC for sin[θ_a+θ_b]).
        P_re = CC - SS
        P_im = CS + SC
        # Quotient (CC+SS for cos[θ_a-θ_b], SC-CS for sin[θ_a-θ_b]).
        Q_re = CC + SS
        Q_im = SC - CS
        # Energy in each direction (sum of squared d_model components).
        # The total E[k,k] = CC² + CS² + SC² + SS²
        # The decomposition (P_re, P_im, Q_re, Q_im) is just an orthogonal basis
        # rotation; their summed squared energies should also equal total/2 each
        # (since P and Q are orthogonal complementary halves).
        E_total = (CC ** 2 + CS ** 2 + SC ** 2 + SS ** 2).sum().item()
        E_prod = 0.5 * (P_re ** 2 + P_im ** 2).sum().item()
        E_quot = 0.5 * (Q_re ** 2 + Q_im ** 2).sum().item()
        # 0.5 factor: rotation orthonormal, but |P|² + |Q|² = 2 * (|CC|² + |SS|² + |CS|² + |SC|²)
        # because (CC-SS)² + (CC+SS)² = 2 CC² + 2 SS², similarly for CS, SC.
        # So divide by 2 to get the right shares.
        print(f"{k:>4} {order_of(k):>4}  "
              f"{E_total:>10.3e}  {E_prod:>10.3e}  {E_quot:>10.3e}  "
              f"{100*E_prod/E_total:>5.1f}%  "
              f"{100*E_quot/E_total:>5.1f}%")


if __name__ == "__main__":
    main()
