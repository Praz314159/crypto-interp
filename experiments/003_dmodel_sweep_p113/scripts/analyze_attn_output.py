"""Test whether the attention output carries the piggybacker characters'
products. Mirrors the MLP-output analysis but on attn_out at the '=' position.

For each character k, compute:
  - g_attn_k = Σ_{a, b} cos[θ_k(a) + θ_k(b)] · attn_out(a, b)  (intrinsic direction)
  - ||g_attn_k|| / ||g_mlp_k|| ratio (compares the "amount of char-k product" in each path)
  - 2D Fourier energy at (k, k) for attn vs mlp

If attention is computing the piggyback products, expect:
  - For dominants {3, 10, 51}: most product energy in MLP, little in attn.
  - For piggybackers {20, 6}: large fraction of product energy in attn, little in MLP.

Usage:
    python experiments/003_dmodel_sweep_p113/scripts/analyze_attn_output.py \
        --run-dir experiments/003_dmodel_sweep_p113/runs/dmodel_24_dmlp_32_seed1 \
        --ks 3,10,20,51,6
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


def compute_grids(model, ds):
    """Run model and return both attn_out and mlp_out at the '=' position,
    each as a (p-1, p-1, d_model) tensor."""
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
    attn_out = cache["blocks.0.hook_attn_out"][:, -1, :].double().reshape(p - 1, p - 1, -1)
    mlp_out = cache["blocks.0.hook_mlp_out"][:, -1, :].double().reshape(p - 1, p - 1, -1)
    return attn_out, mlp_out


def char_product_energy(f_ab, k, p=113):
    """For an (a, b, d_model) tensor f, compute the energy at character-k
    product (real + imaginary parts).
    Returns (||cos_dir||, ||sin_dir||) — magnitudes of intrinsic directions."""
    _, dlog = discrete_log_table(p)
    j_a = np.array([dlog[a] for a in range(1, p)])
    theta_a = 2 * np.pi * k * j_a / (p - 1)
    cos_ref = torch.tensor(np.cos(theta_a[:, None] + theta_a[None, :]),
                            dtype=torch.float64)
    sin_ref = torch.tensor(np.sin(theta_a[:, None] + theta_a[None, :]),
                            dtype=torch.float64)
    g_cos = torch.einsum("ab,abd->d", cos_ref, f_ab)
    g_sin = torch.einsum("ab,abd->d", sin_ref, f_ab)
    return float(g_cos.norm().item()), float(g_sin.norm().item()), g_cos, g_sin


def reconstruct(f_ab, g_dir):
    """Project f_ab onto g_dir / ||g_dir||, returning a (p-1, p-1) signal."""
    if g_dir.norm() < 1e-12:
        return torch.zeros(f_ab.shape[0], f_ab.shape[1]).numpy()
    u = g_dir / g_dir.norm()
    return torch.einsum("abd,d->ab", f_ab, u).cpu().numpy()


def correlate(x, y):
    x = x.flatten() - x.mean()
    y = y.flatten() - y.mean()
    return float(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--ks", required=True)
    args = ap.parse_args()
    run_dir = Path(args.run_dir).resolve()
    ck = sorted(run_dir.glob("checkpoint_*.pt"))
    model, ds, _ = load_run(ck[-1])
    model.eval()
    p = ds.p
    attn_out, mlp_out = compute_grids(model, ds)
    print(f"attn_out shape: {tuple(attn_out.shape)}")
    print(f"mlp_out  shape: {tuple(mlp_out.shape)}")

    print(f"\n{'k':>4} {'o':>4}  "
          f"{'||g_attn_cos||':>14}  {'||g_mlp_cos||':>14}  {'attn/mlp':>9}  "
          f"{'attn-corr':>10}  {'mlp-corr':>10}")
    print("-" * 100)
    ks = [int(x) for x in args.ks.split(",")]
    rows = []
    for k in ks:
        a_c, a_s, ga_c, ga_s = char_product_energy(attn_out, k, p)
        m_c, m_s, gm_c, gm_s = char_product_energy(mlp_out, k, p)
        a_tot = (a_c ** 2 + a_s ** 2) ** 0.5
        m_tot = (m_c ** 2 + m_s ** 2) ** 0.5
        ratio = a_tot / (m_tot + 1e-12)
        _, dlog = discrete_log_table(p)
        j_a = np.array([dlog[a] for a in range(1, p)])
        ref = np.cos(2 * np.pi * k * (j_a[:, None] + j_a[None, :]) / (p - 1))
        sig_attn = reconstruct(attn_out, ga_c)
        sig_mlp = reconstruct(mlp_out, gm_c)
        corr_attn = correlate(sig_attn, ref)
        corr_mlp = correlate(sig_mlp, ref)
        print(f"{k:>4} {order_of(k):>4}  "
              f"{a_c:>14.3e}  {m_c:>14.3e}  {ratio:>9.3f}  "
              f"{corr_attn:>+10.3f}  {corr_mlp:>+10.3f}")
        rows.append((k, a_tot, m_tot, ratio, corr_attn, corr_mlp))

    # Visualize the attn signal for k=20 and k=6 (the piggybackers).
    fig_dir = run_dir
    for k in (20, 6):
        a_c, a_s, ga_c, ga_s = char_product_energy(attn_out, k, p)
        _, dlog = discrete_log_table(p)
        j_a = np.array([dlog[a] for a in range(1, p)])
        ref = np.cos(2 * np.pi * k * (j_a[:, None] + j_a[None, :]) / (p - 1))
        sig_attn = reconstruct(attn_out, ga_c)
        order_idx = np.argsort([dlog[a] for a in range(1, p)])

        fig, axes = plt.subplots(2, 2, figsize=(11, 10))
        for ax, data, title in [
            (axes[0, 0], sig_attn, f"attn k={k} signal, natural"),
            (axes[0, 1], sig_attn[order_idx][:, order_idx], "attn signal, dlog-sorted"),
            (axes[1, 0], ref, "reference cos[θ_k(a)+θ_k(b)], natural"),
            (axes[1, 1], ref[order_idx][:, order_idx], "reference, dlog-sorted"),
        ]:
            vmax = max(abs(data.max()), abs(data.min()))
            ax.imshow(data, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                      aspect="equal", origin="upper")
            ax.set_title(title, fontsize=9)
        corr = correlate(sig_attn, ref)
        fig.suptitle(f"Attention output projected onto intrinsic char-{k} direction "
                     f"(order {order_of(k)}); corr = {corr:+.3f}",
                     fontsize=11)
        fig.tight_layout()
        out = fig_dir / f"attn_signal_k{k}.png"
        fig.savefig(out, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out.name}")


if __name__ == "__main__":
    main()
