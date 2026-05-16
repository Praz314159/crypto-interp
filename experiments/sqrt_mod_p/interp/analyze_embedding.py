"""First analysis: does the embedding W_E factor through additive or multiplicative
Fourier features?

For each input token a ∈ {0, ..., p-1}, W_E[:, a] is a d_model-dimensional vector.
Stack into a (d_model, p) matrix. Project along the token dimension by each basis.
The resulting (d_model, p) coefficient matrix should be sparse in the "right" basis.

Concretely:
  - Energy at basis row k = sum over d_model of (coef[k, :])^2.
  - Plot: energy at each basis vector. Sparsity → only a few rows have nonzero energy.

For Nanda's modular addition: sparse in additive basis.
Our prediction for modular multiplication: sparse in multiplicative basis.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

import interp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="runs/mul_baseline")
    parser.add_argument("--out", type=str, default=None,
                        help="Path to save the plot. Default: <ckpt-dir>/embedding_basis_energy.png")
    args = parser.parse_args()

    ckpt_path = interp.latest_checkpoint(args.ckpt) if Path(args.ckpt).is_dir() else Path(args.ckpt)
    print(f"Loading: {ckpt_path}")
    model, ds, ckpt = interp.load_run(ckpt_path)
    print(f"  task={ds.task}, p={ds.p}, epoch={ckpt['epoch']}")

    # Pull the embedding for the value tokens (drop the '=' separator and any others).
    p = ds.p
    W_E = model.embed.W_E.detach().double()  # (d_model, vocab)
    # Restrict to the p value tokens.
    W_E_values = W_E[:, :p]  # (d_model, p)
    print(f"W_E_values shape: {tuple(W_E_values.shape)}")

    # Build both bases.
    add_basis, add_names = interp.additive_fourier_basis(p)
    mul_basis, mul_names, g = interp.multiplicative_fourier_basis(p)
    print(f"Primitive root for multiplicative basis: g={g}")

    # Project W_E along the token dim to each basis.
    # coef has shape (p_basis, d_model); rows are basis-vector coefficients.
    add_coef = torch.einsum("kp,dp->kd", add_basis, W_E_values)
    mul_coef = torch.einsum("kp,dp->kd", mul_basis, W_E_values)

    # Energy per basis vector = sum over d_model of coefficient^2.
    add_energy = (add_coef ** 2).sum(dim=1).cpu().numpy()
    mul_energy = (mul_coef ** 2).sum(dim=1).cpu().numpy()

    total_add = add_energy.sum()
    total_mul = mul_energy.sum()
    print(f"Total energy in additive basis: {total_add:.4f}")
    print(f"Total energy in multiplicative basis: {total_mul:.4f}")

    # Sparsity: top-k variance explained.
    def topk_explained(energy, ks=(1, 3, 5, 10, 20)):
        srt = np.sort(energy)[::-1]
        total = srt.sum()
        return [(k, srt[:k].sum() / total) for k in ks]

    print("\nTop-k variance explained, additive basis:")
    for k, frac in topk_explained(add_energy):
        print(f"  top-{k:2d}: {frac:.4f}")
    print("\nTop-k variance explained, multiplicative basis:")
    for k, frac in topk_explained(mul_energy):
        print(f"  top-{k:2d}: {frac:.4f}")

    # Identify dominant basis vectors.
    add_top = np.argsort(add_energy)[::-1][:10]
    mul_top = np.argsort(mul_energy)[::-1][:10]
    print("\nTop-10 additive basis components:")
    for i in add_top:
        print(f"  {i:3d}  {add_names[i]:12s}  energy={add_energy[i]:.4f}")
    print("\nTop-10 multiplicative basis components:")
    for i in mul_top:
        print(f"  {i:3d}  {mul_names[i]:18s}  energy={mul_energy[i]:.4f}")

    # Plot.
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.5), sharex=False)
    axes[0].bar(np.arange(len(add_energy)), add_energy, width=1.0, color='#1f77b4')
    axes[0].set_title(f"Embedding energy in ADDITIVE Fourier basis on Z/{p}")
    axes[0].set_xlabel("basis index (0=const, then cos/sin pairs at increasing k)")
    axes[0].set_ylabel("energy = Σ_d coef²")

    axes[1].bar(np.arange(len(mul_energy)), mul_energy, width=1.0, color='#d62728')
    axes[1].set_title(f"Embedding energy in MULTIPLICATIVE basis on (Z/{p})* "
                      f"(g={g}; index 0 is delta_0)")
    axes[1].set_xlabel("basis index (0=delta_0, then mult-char Fourier on Z/{0})".format(p - 1))
    axes[1].set_ylabel("energy = Σ_d coef²")

    fig.suptitle(f"Embedding W_E sparsity, task={ds.task}, p={p}, epoch={ckpt['epoch']}")
    fig.tight_layout()

    out = Path(args.out) if args.out else Path(ckpt_path).parent / "embedding_basis_energy.png"
    fig.savefig(out, dpi=130)
    print(f"\nSaved plot: {out}")


if __name__ == "__main__":
    main()
