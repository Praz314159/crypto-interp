"""For each Δ_k(Δlog) 1D reduction, compute the Fourier spectrum: find what
frequency the signal is actually at. If k=20's ablation contribution shows
peak at frequency 40 (= 2·20), that's evidence the model is using k=20 to
derive a harmonic χ_40 via ReLU squaring.

Usage:
    python experiments/003_dmodel_sweep_p113/scripts/analyze_delta_spectrum.py \
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


def build_basis_indexed():
    basis, names, _ = multiplicative_fourier_basis(113)
    char_idx = {}
    for i, nm in enumerate(names):
        m = re.match(r"mul (cos|sin) (\d+)", nm)
        if m:
            kk = int(m.group(2))
            char_idx.setdefault(kk, []).append(i)
    return basis.double(), char_idx


def ablate_W_E(W_E, k, basis, char_idx, p=113):
    W_E_new = W_E.clone().double()
    W_E_v = W_E_new[:, :p].clone()
    for r in char_idx[k]:
        b = basis[r].double()
        coef = (W_E_v @ b)
        W_E_v = W_E_v - coef[:, None] * b[None, :]
    W_E_new[:, :p] = W_E_v
    return W_E_new


def compute_logits_grid(model, ds, W_E_override=None):
    p = ds.p
    aa, bb = torch.meshgrid(torch.arange(1, p), torch.arange(1, p), indexing="ij")
    eq = torch.full_like(aa, ds.eq_token)
    inputs = torch.stack([aa, bb, eq], dim=-1).reshape(-1, 3)
    if W_E_override is not None:
        orig = model.embed.W_E.detach().clone()
        with torch.no_grad():
            model.embed.W_E.copy_(W_E_override.to(model.embed.W_E.dtype))
    with torch.no_grad():
        logits = model(inputs)[:, -1, :].double()
    if W_E_override is not None:
        with torch.no_grad():
            model.embed.W_E.copy_(orig)
    return logits.reshape(p - 1, p - 1, -1)


def reduce_to_1d(delta_abc, p):
    """Reduce Δ_k(a, b, c) to f(Δlog) where Δlog = log(c) - log(ab) mod (p-1).
    Averages over all (a, b, c) with the same Δlog value."""
    arr = delta_abc.cpu().numpy()
    _, dlog = discrete_log_table(p)
    aa = np.arange(1, p)
    bb = np.arange(1, p)
    ab_grid = (aa[:, None] * bb[None, :]) % p   # (p-1, p-1)
    n = p - 1
    out = np.zeros(n)
    counts = np.zeros(n)
    for a_i in range(p - 1):
        for b_i in range(p - 1):
            ab = ab_grid[a_i, b_i]
            if ab == 0:
                continue
            j_ab = dlog[ab]
            for c in range(1, p):
                j_c = dlog[c]
                delta = (j_c - j_ab) % n
                out[delta] += arr[a_i, b_i, c]
                counts[delta] += 1
    out /= np.where(counts > 0, counts, 1.0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--ks", required=True)
    args = ap.parse_args()
    run_dir = Path(args.run_dir).resolve()
    ck = sorted(run_dir.glob("checkpoint_*.pt"))
    model, ds, _ = load_run(ck[-1])
    model.eval()
    basis, char_idx = build_basis_indexed()
    p = ds.p
    W_E_orig = model.embed.W_E.detach().clone()
    logits_full = compute_logits_grid(model, ds)

    ks = [int(x) for x in args.ks.split(",")]
    fig, axes = plt.subplots(len(ks), 1, figsize=(13, 2.2 * len(ks)), sharex=True)
    if len(ks) == 1:
        axes = [axes]

    for i, k in enumerate(ks):
        W_E_ab = ablate_W_E(W_E_orig, k, basis, char_idx, p)
        logits_ab = compute_logits_grid(model, ds, W_E_override=W_E_ab)
        delta = logits_full - logits_ab
        f_diff = reduce_to_1d(delta, p)

        # Fourier spectrum of f_diff. Real DFT.
        N = len(f_diff)
        # Project onto cos(2π m Δlog / N) and sin(2π m Δlog / N) for m=1..N/2
        dl = np.arange(N)
        cos_proj = np.array([np.dot(f_diff, np.cos(2*np.pi*m*dl/N)) for m in range(1, N//2+1)])
        sin_proj = np.array([np.dot(f_diff, np.sin(2*np.pi*m*dl/N)) for m in range(1, N//2+1)])
        energy = cos_proj**2 + sin_proj**2

        # Top-5 frequencies.
        top = sorted(range(len(energy)), key=lambda i: -energy[i])[:5]
        top_strs = [f"m={top[j]+1} E={energy[top[j]]:.2e}" for j in range(5)]
        print(f"ablating k={k:>3} (order {order_of(k)}):  top freqs: " + "; ".join(top_strs))

        ax = axes[i]
        ax.bar(np.arange(1, N//2+1), energy, color="#1f77b4", alpha=0.85)
        # Mark the "naive" expected frequency m=k.
        if k <= N // 2:
            ax.axvline(k, color="red", ls="--", lw=1.0, alpha=0.7,
                       label=f"naive k={k}")
        # Mark 2k if it's also in range.
        if 2 * k <= N // 2:
            ax.axvline(2 * k, color="orange", ls="--", lw=1.0, alpha=0.7,
                       label=f"2k={2*k}")
        # Mark p-1 - k = 112-k
        if N - k <= N // 2:
            ax.axvline(N - k, color="green", ls="--", lw=1.0, alpha=0.7,
                       label=f"-k={N-k}")
        ax.set_ylabel(f"ablate k={k}")
        ax.set_yscale("log")
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.3, axis="y")

    axes[-1].set_xlabel("Fourier frequency m of Δ_k(Δlog)")
    fig.suptitle(f"Spectrum of Δ_k(Δlog): what frequency does ablating each k destroy?",
                 fontsize=11)
    fig.tight_layout()
    out = run_dir / "delta_spectrum.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
