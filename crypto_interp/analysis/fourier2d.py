"""2D Fourier decomposition of the MLP output as a function of (a, b).

For each character pair (k_a, k_b), compute the energy of the MLP output at the
bilinear basis function χ_{k_a}(a) · χ_{k_b}(b). Diagonal entries (k, k) are
standard character products χ_k(ab); off-diagonals are cross-frequency terms.

Writes ``fourier2d_<tag>.png`` to --out-dir (default: the run dir).

Usage:
    python -m crypto_interp.analysis.fourier2d \
        --run-dir experiments/003_dmodel_sweep_p113/runs/dmodel_24_dmlp_32_seed1 \
        --tag dmlp32_seed1
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from crypto_interp.interp import char_index, compute_activation_grid, load_run, order_of


def fourier2d_energy(f_ab, basis, ci, p):
    """f_ab: (p-1, p-1, d_model). Returns (n_chars, n_chars) energy matrix
    E[k_a-1, k_b-1] summed over the cos/sin bilinear pairs and over d_model."""
    basis_v = basis[:, 1:p].double()                     # (n_basis, p-1)
    coef = torch.einsum("ka,lb,abd->kld", basis_v, basis_v, f_ab)
    E = (coef ** 2).sum(dim=-1).cpu().numpy()            # (n_basis, n_basis)
    nch = max(ci.freqs)
    out = np.zeros((nch, nch))
    for k_a, rs_a in ci.by_char.items():
        for k_b, rs_b in ci.by_char.items():
            out[k_a - 1, k_b - 1] = E[np.ix_(rs_a, rs_b)].sum()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out-dir", default=None, help="Where to write the figure (default: run dir).")
    args = ap.parse_args()
    run_dir = Path(args.run_dir).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else run_dir
    ck = sorted(run_dir.glob("checkpoint_*.pt"))
    print(f"Loading {ck[-1]}")
    model, ds, _ = load_run(ck[-1])
    model.eval()
    p = ds.p
    basis, ci = char_index(p)
    nch = max(ci.freqs)

    f_ab = compute_activation_grid(model, ds, "blocks.0.hook_mlp_out")
    print(f"MLP output shape: {tuple(f_ab.shape)}")
    E = fourier2d_energy(f_ab, basis, ci, p)

    diag = np.diag(E)
    diag_total = diag.sum()
    off_total = E.sum() - diag_total
    print(f"\nTotal energy: {E.sum():.4e}")
    print(f"  diagonal:   {diag_total:.4e}  ({100*diag_total/E.sum():.1f}%)")
    print(f"  off-diag:   {off_total:.4e}  ({100*off_total/E.sum():.1f}%)")

    print("\nTop-12 diagonal entries (k, k):")
    for i in np.argsort(diag)[::-1][:12]:
        k = i + 1
        print(f"  k={k:>3} (order {order_of(k, p)}): E[{k},{k}] = {diag[i]:.3e}")
    E_off = E.copy(); np.fill_diagonal(E_off, 0)
    print("\nTop-12 off-diagonal entries:")
    for idx in np.argsort(E_off.flatten())[::-1][:12]:
        ka, kb = idx // nch + 1, idx % nch + 1
        print(f"  ({ka:>3}, {kb:>3})  orders ({order_of(ka, p)}, {order_of(kb, p)}): "
              f"E = {E_off[ka - 1, kb - 1]:.3e}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    im = axes[0].imshow(np.log10(E + 1e-12), cmap="viridis", aspect="equal",
                        origin="lower", extent=[1, nch + 1, 1, nch + 1])
    axes[0].set_xlabel("k_b"); axes[0].set_ylabel("k_a")
    axes[0].set_title("(a) log10 E[k_a, k_b]")
    plt.colorbar(im, ax=axes[0], shrink=0.85)
    axes[0].plot([1, nch], [1, nch], "w--", lw=0.5, alpha=0.5)

    ax = axes[1]
    ax.bar(np.arange(1, nch + 1), diag, color="#d62728", alpha=0.85,
           edgecolor="black", linewidth=0.3, label="diagonal E[k,k]")
    for k in sorted(k + 1 for k, e in enumerate(diag) if e > 0.05 * diag.max()):
        ax.text(k, diag[k - 1] * 1.02, str(k), ha="center", fontsize=8, fontweight="bold")
    ax.set_xlabel("k"); ax.set_ylabel("diagonal energy E[k, k]")
    ax.set_yscale("log"); ax.set_title("(b) per-character diagonal energy")
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle(f"2D Fourier decomposition of MLP output ({args.tag})\n"
                 f"diagonal = {100*diag_total/E.sum():.1f}%, "
                 f"off-diag = {100*off_total/E.sum():.1f}%", fontsize=11)
    fig.tight_layout()
    out = out_dir / f"fourier2d_{args.tag}.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
