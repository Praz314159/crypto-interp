"""2D Fourier decomposition of the MLP output as a function of (a, b).

For each character pair (k_a, k_b), compute the energy of the MLP output at
the bilinear basis function χ_{k_a}(a) · χ_{k_b}(b). Diagonal entries (k, k)
correspond to standard character products χ_k(ab); off-diagonals correspond to
cross-frequency bilinear terms (not character products of ab). Diagonal entries
at characters NOT in W_E's K suggest harmonic generation via ReLU.

For each (k_a, k_b), the energy is summed over the four cos/sin pair products
and over d_model dimensions.

Outputs:
  figures/basis_dynamics/fourier2d_<tag>.png

Usage:
    python experiments/003_dmodel_sweep_p113/scripts/analyze_2d_fourier.py \
        --run-dir experiments/003_dmodel_sweep_p113/runs/dmodel_24_dmlp_32_seed1 \
        --tag dmlp32_seed1
"""
from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from crypto_interp.interp.bases import multiplicative_fourier_basis
from crypto_interp.interp.load import load_run

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures" / "basis_dynamics"


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
    return basis.double(), names, char_idx


def compute_mlp_output_grid(model, ds):
    """Run model on all (a, b) ∈ {1,...,p-1}^2 inputs; return MLP output at the
    final position. Returns tensor of shape (p-1, p-1, d_model)."""
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
    # mlp_out: shape (N, n_ctx, d_model); take final position.
    mlp_out = cache["blocks.0.hook_mlp_out"][:, -1, :].double()
    return mlp_out.reshape(p - 1, p - 1, -1)


def fourier2d_energy(f_ab, basis, char_idx, p=113):
    """f_ab: (p-1, p-1, d_model) MLP output.
    Returns: (56, 56) energy matrix E[k_a, k_b] = sum over the 4 cos/sin
    bilinear basis pairs and over d_model dims of the squared projection
    coefficient.

    The basis on each axis is (p, p) but indexed over a, b ∈ {1, ..., p-1} so
    we need to use the rows of basis at the value-token indices, i.e.,
    basis[:, 1:p]. Each character k corresponds to two basis rows (cos and sin).
    """
    # Restrict basis to a ∈ {1, ..., p-1}.
    basis_v = basis[:, 1:p].double()                    # (n_basis, p-1)
    # Project on both axes.
    # f_ab: (p-1, p-1, d_model); want coef[k1, k2, d] = sum_{a,b} basis[k1,a] * basis[k2,b] * f[a,b,d]
    coef = torch.einsum("ka,lb,abd->kld", basis_v, basis_v, f_ab)
    E = (coef ** 2).sum(dim=-1).cpu().numpy()           # (n_basis, n_basis)
    # Aggregate basis indices into characters (cos+sin pair).
    out = np.zeros((56, 56))
    for k_a, rs_a in char_idx.items():
        for k_b, rs_b in char_idx.items():
            patch = E[np.ix_(rs_a, rs_b)]
            out[k_a - 1, k_b - 1] = patch.sum()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    run_dir = Path(args.run_dir).resolve()
    ck = sorted(run_dir.glob("checkpoint_*.pt"))
    print(f"Loading {ck[-1]}")
    model, ds, _ = load_run(ck[-1])
    model.eval()
    basis, names, char_idx = build_basis_indexed()

    f_ab = compute_mlp_output_grid(model, ds)
    print(f"MLP output shape: {tuple(f_ab.shape)}")
    E = fourier2d_energy(f_ab, basis, char_idx, p=ds.p)

    diag = np.diag(E)
    off_total = E.sum() - diag.sum()
    diag_total = diag.sum()
    print(f"\nTotal energy: {E.sum():.4e}")
    print(f"  diagonal:   {diag_total:.4e}  ({100*diag_total/E.sum():.1f}%)")
    print(f"  off-diag:   {off_total:.4e}  ({100*off_total/E.sum():.1f}%)")

    # Top diagonal entries.
    top_diag = np.argsort(diag)[::-1][:12]
    print(f"\nTop-12 diagonal entries (k, k):")
    for i in top_diag:
        k = i + 1
        print(f"  k={k:>3} (order {order_of(k)}): E[{k},{k}] = {diag[i]:.3e}")
    # Top off-diagonal entries.
    E_off = E.copy()
    np.fill_diagonal(E_off, 0)
    flat = E_off.flatten()
    top_off_idx = np.argsort(flat)[::-1][:12]
    print(f"\nTop-12 off-diagonal entries:")
    for idx in top_off_idx:
        ka, kb = idx // 56 + 1, idx % 56 + 1
        print(f"  ({ka:>3}, {kb:>3})  orders ({order_of(ka)}, {order_of(kb)}): "
              f"E = {E_off[ka - 1, kb - 1]:.3e}")

    # Plot.
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    ax = axes[0]
    im = ax.imshow(np.log10(E + 1e-12), cmap="viridis", aspect="equal",
                   origin="lower", extent=[1, 57, 1, 57])
    ax.set_xlabel("k_b")
    ax.set_ylabel("k_a")
    ax.set_title(f"(a) log10 E[k_a, k_b]")
    plt.colorbar(im, ax=ax, shrink=0.85)
    ax.plot([1, 56], [1, 56], "w--", lw=0.5, alpha=0.5)

    ax = axes[1]
    ax.bar(np.arange(1, 57), diag, color="#d62728", alpha=0.85,
           edgecolor="black", linewidth=0.3, label="diagonal E[k,k]")
    # Annotate the K characters.
    K = sorted([k + 1 for k, e in enumerate(diag) if e > 0.05 * diag.max()])
    for k in K:
        ax.text(k, diag[k - 1] * 1.02, str(k), ha="center", fontsize=8,
                fontweight="bold")
    ax.set_xlabel("k")
    ax.set_ylabel("diagonal energy E[k, k]")
    ax.set_yscale("log")
    ax.set_title(f"(b) per-character diagonal energy")
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle(f"2D Fourier decomposition of MLP output ({args.tag})\n"
                 f"diagonal = {100*diag_total/E.sum():.1f}%, "
                 f"off-diag = {100*off_total/E.sum():.1f}%",
                 fontsize=11)
    fig.tight_layout()
    out = FIG_DIR / f"fourier2d_{args.tag}.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
