"""Generate the classic 'restricted vs excluded loss' ablation curve.

For each K = 0..n_pairs:
  - restricted_K = test loss when we KEEP ONLY the top-K key frequency pairs
                    (cos/sin) in the multiplicative Fourier basis.
  - excluded_K  = test loss when we EXCLUDE the top-K key frequencies.

Frequencies are ranked by embedding energy. The plot tells us:
  - How many frequencies the algorithm actually uses (restricted curve plateau).
  - How critical those frequencies are (excluded curve spike).
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from crypto_interp import interp
from crypto_interp.interp import ablate_embedding, evaluate_loss


def rank_frequencies(W_E_values: torch.Tensor, basis: torch.Tensor, p: int):
    """Return frequencies sorted by total cos+sin energy, descending.

    Returns a list of (rank, frequency_k, cos_idx, sin_idx_or_None, energy).
    """
    coef = torch.einsum("kp,dp->kd", basis, W_E_values.to(basis.dtype))
    energy_per_basis = (coef ** 2).sum(dim=1).cpu().numpy()

    n = p - 1
    rows = []
    for k in range(1, (n - 1) // 2 + 1):
        ci, si = 2 * k, 2 * k + 1
        rows.append((k, ci, si, energy_per_basis[ci] + energy_per_basis[si]))
    if n % 2 == 0:
        rows.append((n // 2, n, None, energy_per_basis[n]))
    rows.sort(key=lambda r: -r[3])
    return rows


def ablation_curve(model, ds, basis, ranked_freqs, max_K=None):
    """Return arrays (Ks, restricted_loss, excluded_loss, restricted_acc, excluded_acc)."""
    p = ds.p
    n_pairs = len(ranked_freqs)
    if max_K is None:
        max_K = n_pairs

    Ks, restricted, excluded, r_acc, e_acc = [], [], [], [], []
    for K in range(0, max_K + 1):
        keep_mask = torch.zeros(p, dtype=torch.bool)
        # Always keep delta_0 and const? Nanda doesn't; he treats them as part
        # of the basis. We'll *not* keep them by default, matching his setup.
        for (_, ci, si, _) in ranked_freqs[:K]:
            keep_mask[ci] = True
            if si is not None:
                keep_mask[si] = True

        # Restricted: keep only the top-K key frequencies' cos/sin pair indices.
        m_r = ablate_embedding(model, basis, keep_mask)
        _, te_r, ac_r = evaluate_loss(m_r, ds)

        # Excluded: keep everything EXCEPT the top-K cos/sin pair indices.
        m_e = ablate_embedding(model, basis, ~keep_mask)
        _, te_e, ac_e = evaluate_loss(m_e, ds)

        Ks.append(K)
        restricted.append(te_r)
        excluded.append(te_e)
        r_acc.append(ac_r)
        e_acc.append(ac_e)
        if K <= 10 or K % 5 == 0:
            print(f"K={K:3d}: restricted loss={te_r:.3e} (acc {ac_r:.3f}) | "
                  f"excluded loss={te_e:.3e} (acc {ac_e:.3f})")

    return np.array(Ks), np.array(restricted), np.array(excluded), np.array(r_acc), np.array(e_acc)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="runs/mul_baseline")
    parser.add_argument("--max-K", type=int, default=20,
                        help="Maximum K to compute (default 20 — full curve would be (p-1)/2).")
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    ckpt_path = interp.latest_checkpoint(args.ckpt) if Path(args.ckpt).is_dir() else Path(args.ckpt)
    print(f"Loading: {ckpt_path}")
    model, ds, ckpt = interp.load_run(ckpt_path)

    p = ds.p
    W_E_values = model.embed.W_E.detach().double()[:, :p]
    mul_basis, _, g = interp.multiplicative_fourier_basis(p)
    print(f"  primitive root g={g}, p={p}")

    ranked = rank_frequencies(W_E_values, mul_basis, p)
    print(f"\nTop ranked frequencies (sorted by embedding energy):")
    for r, (k, ci, si, e) in enumerate(ranked[:10]):
        print(f"  rank {r+1}: frequency {k:3d}, energy={e:.4f}")

    # Baseline
    _, te0, ac0 = evaluate_loss(model, ds)
    print(f"\nBaseline test loss: {te0:.3e}, accuracy: {ac0:.4f}")

    print(f"\nComputing ablation curve up to K={args.max_K}...")
    Ks, restricted, excluded, r_acc, e_acc = ablation_curve(
        model, ds, mul_basis, ranked, max_K=args.max_K
    )

    chance = float(np.log(ds.n_answer_tokens))

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Loss curve
    ax1.plot(Ks, restricted, 'o-', label='restricted (keep top-K only)',
             color='#1f77b4', markersize=5)
    ax1.plot(Ks, excluded, 's-', label='excluded (remove top-K)',
             color='#d62728', markersize=5)
    ax1.axhline(te0, color='black', ls='-', lw=0.8, alpha=0.6,
                label=f'baseline (full model) ≈ {te0:.2e}')
    ax1.axhline(chance, color='gray', ls='--', lw=0.8,
                label=f'chance ≈ {chance:.2f}')
    ax1.set_yscale('log')
    ax1.set_xlabel('K (top-K key frequency pairs)')
    ax1.set_ylabel('test cross-entropy loss')
    ax1.set_title('Restricted vs excluded loss as a function of K')
    ax1.legend(loc='center right', fontsize=9)
    ax1.grid(True, which='both', alpha=0.3)

    # Accuracy curve
    ax2.plot(Ks, r_acc, 'o-', label='restricted accuracy',
             color='#1f77b4', markersize=5)
    ax2.plot(Ks, e_acc, 's-', label='excluded accuracy',
             color='#d62728', markersize=5)
    ax2.axhline(ac0, color='black', ls='-', lw=0.8, alpha=0.6,
                label=f'baseline acc ≈ {ac0:.4f}')
    ax2.axhline(1.0 / ds.n_answer_tokens, color='gray', ls='--', lw=0.8,
                label=f'chance ≈ {1.0/ds.n_answer_tokens:.4f}')
    ax2.set_xlabel('K (top-K key frequency pairs)')
    ax2.set_ylabel('test accuracy')
    ax2.set_title('Accuracy under the same ablations')
    ax2.legend(loc='center right', fontsize=9)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(f"Embedding ablation: task={ds.task}, p={p}, "
                 f"basis=multiplicative Fourier on (Z/{p})*")
    fig.tight_layout()

    out = Path(args.out) if args.out else Path(ckpt_path).parent / "ablation_curve.png"
    fig.savefig(out, dpi=130)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
