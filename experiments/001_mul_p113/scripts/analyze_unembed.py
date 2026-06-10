"""Unembed structure analysis.

Prediction: the unembed W_U should read off the answer via
   logit(c) ~ Σ_k cos(k · (log_g(a·b) − log_g(c)))
which is maximized at c = a·b mod p.

W_U is a (d_model, vocab) matrix; for each output token c, column W_U[:, c] is
a d_model vector. If we treat the columns indexed by c ∈ {0, ..., p-1} as a
function from Z/p → R^d_model, we expect it to be sparse in the SAME
multiplicative Fourier basis as the embedding — at the same key frequencies
{17, 22, 41, 18}.

Moreover, for each key frequency k, W_U should encode cos(k · log_g(c)) and
sin(k · log_g(c)) in d_model directions that are PAIRED with the embedding's
cos(k · log_g(a)) and sin(k · log_g(a)) directions — so that the dot product
of (residual stream features) with W_U[:, c] gives cos(k(log a + log b − log c)).
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from crypto_interp import interp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="runs/mul_baseline")
    args = parser.parse_args()

    ckpt_path = interp.latest_checkpoint(args.ckpt) if Path(args.ckpt).is_dir() else Path(args.ckpt)
    print(f"Loading: {ckpt_path}")
    model, ds, ckpt = interp.load_run(ckpt_path)
    p = ds.p

    # W_U: (d_model, vocab). Restrict to value-token columns 0..p-1.
    W_U = model.unembed.W_U.detach().double()  # (d_model, vocab)
    W_U_values = W_U[:, :p]  # (d_model, p)
    print(f"W_U_values shape: {tuple(W_U_values.shape)}")

    # Project W_U columns (indexed by output token c) into the multiplicative basis.
    mul_basis, _, g = interp.multiplicative_fourier_basis(p)
    print(f"  primitive root g={g}")

    # coef_U[k, d] = Σ_c basis[k, c] * W_U[d, c]
    coef_U = torch.einsum("kp,dp->kd", mul_basis, W_U_values)
    energy_U = (coef_U ** 2).sum(dim=1).cpu().numpy()

    # Embedding for comparison
    W_E_values = model.embed.W_E.detach().double()[:, :p]
    coef_E = torch.einsum("kp,dp->kd", mul_basis, W_E_values)
    energy_E = (coef_E ** 2).sum(dim=1).cpu().numpy()

    # Rank both by energy
    n = p - 1
    print("\nTop-10 multiplicative-basis components by energy:")
    print(f"{'rank':<5} {'idx':<5} {'name':<14} {'E energy':<12} {'U energy':<12}")
    # Build a "frequency" view: combine cos+sin pair for each k
    freq_E = {}
    freq_U = {}
    for k in range(1, (n - 1) // 2 + 1):
        freq_E[k] = energy_E[2*k] + energy_E[2*k + 1]
        freq_U[k] = energy_U[2*k] + energy_U[2*k + 1]
    if n % 2 == 0:
        freq_E[n // 2] = energy_E[n]
        freq_U[n // 2] = energy_U[n]

    # Top by W_U
    top_U_freqs = sorted(freq_U, key=lambda k: -freq_U[k])[:10]
    top_E_freqs = sorted(freq_E, key=lambda k: -freq_E[k])[:10]

    print("\nTop-10 frequencies by W_U energy:")
    for r, k in enumerate(top_U_freqs):
        print(f"  rank {r+1}: freq {k:3d}, W_U energy={freq_U[k]:.4f}, W_E energy={freq_E[k]:.4f}")

    # Variance-explained
    total_U = sum(freq_U.values()) + (energy_U[0] + energy_U[1])  # include const + delta_0
    print(f"\nVariance explained, W_U:")
    cum = 0
    for r, k in enumerate(top_U_freqs):
        cum += freq_U[k]
        print(f"  top-{r+1:2d} frequencies: {cum/total_U:.4f}")

    # ---- Key cross-check: does W_U use the SAME frequencies as W_E? ----
    key_E = set(top_E_freqs[:5])
    key_U = set(top_U_freqs[:5])
    overlap = key_E & key_U
    print(f"\nKey-frequency overlap: W_E top-5 = {sorted(key_E)}, "
          f"W_U top-5 = {sorted(key_U)}")
    print(f"  Intersection: {sorted(overlap)} ({len(overlap)}/5)")

    # ---- Check: do W_U cos/sin directions PAIR with W_E's? ----
    # The algorithm computes cos(k(log a + log b - log c)) which expands to
    #   cos(k(la+lb)) cos(k lc) + sin(k(la+lb)) sin(k lc).
    # So the W_U direction for "cos(k lc)" should align with the residual-stream
    # direction "cos(k(la+lb))", and similarly for sin. For each key k, check
    # the cosine similarity between the cos/sin directions in W_U vs W_E.
    print(f"\nDirection alignment per key frequency (cos similarity):")
    for k in top_U_freqs[:5]:
        ci, si = 2 * k, 2 * k + 1
        # Vectors in R^d_model
        u_cos = coef_U[ci]  # how cos_k(c) is encoded in d_model
        u_sin = coef_U[si]
        e_cos = coef_E[ci]
        e_sin = coef_E[si]

        def cossim(x, y):
            return float((x @ y) / (x.norm() * y.norm() + 1e-12))

        cs_cc = cossim(u_cos, e_cos)
        cs_ss = cossim(u_sin, e_sin)
        cs_cs = cossim(u_cos, e_sin)
        cs_sc = cossim(u_sin, e_cos)
        print(f"  freq {k:3d}: cos<W_U,W_E>={cs_cc:+.3f}  sin<W_U,W_E>={cs_ss:+.3f}  "
              f"(cross: cos·sin={cs_cs:+.3f}, sin·cos={cs_sc:+.3f})")

    # ---- Plot ----
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7))

    ax1.bar(np.arange(len(energy_E)), energy_E, width=1.0, color='#1f77b4')
    ax1.set_title('W_E embedding energy in multiplicative basis (for reference)')
    ax1.set_xlabel('basis index')
    ax1.set_ylabel('energy')

    ax2.bar(np.arange(len(energy_U)), energy_U, width=1.0, color='#d62728')
    ax2.set_title('W_U unembed energy in multiplicative basis')
    ax2.set_xlabel('basis index')
    ax2.set_ylabel('energy')

    fig.suptitle(f'Unembed vs embedding sparsity, task=mul, p={p}, g={g}')
    fig.tight_layout()
    out = Path(ckpt_path).parent / "unembed_basis_energy.png"
    fig.savefig(out, dpi=130)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
