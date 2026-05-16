"""Embedding ablation: validate the multiplicative-Fourier interpretation.

We run three evaluations on the grokked mul model:
  (a) Baseline — full model, no modification.
  (b) Key-only embedding — restrict W_E to only the top-K multiplicative-basis
      components (k indices and their cos/sin pair partners).
  (c) Anti-key embedding — keep everything EXCEPT the top-K components.

The interpretation predicts:
  - (b) ≈ (a) in loss/accuracy. The key components carry the algorithm.
  - (c) collapses to chance or worse. Removing the key components destroys
        the algorithm.
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from crypto_interp import interp
from crypto_interp.interp import ablate_embedding, evaluate_loss


def find_key_frequencies(W_E_values, basis, top_k_pairs=5):
    """Return a sorted list of (cos_idx, sin_idx, frequency, energy) for the top
    cos/sin frequency pairs in the multiplicative basis.

    The multiplicative basis layout is:
       index 0   : delta_0
       index 1   : mul const
       index 2k  : mul cos k     for k = 1..(n-1)//2  (n = p-1)
       index 2k+1: mul sin k
       index n   : mul cos (n/2)  if n even

    For each frequency k, we sum cos and sin energies to get a single "frequency
    energy"; we return the top-k frequencies.
    """
    coef = torch.einsum("kp,dp->kd", basis, W_E_values.to(basis.dtype))
    energy_per_basis = (coef ** 2).sum(dim=1).cpu().numpy()

    p = W_E_values.shape[1]
    n = p - 1
    # Frequencies k = 1..(n-1)//2 are at indices 2k, 2k+1 (cos, sin)
    # If n is even, k = n/2 has index n (cos only).
    freq_energy = []
    for k in range(1, (n - 1) // 2 + 1):
        cos_idx = 2 * k
        sin_idx = 2 * k + 1
        e = energy_per_basis[cos_idx] + energy_per_basis[sin_idx]
        freq_energy.append((k, cos_idx, sin_idx, e))
    if n % 2 == 0:
        cos_idx = n
        freq_energy.append((n // 2, cos_idx, None, energy_per_basis[cos_idx]))

    freq_energy.sort(key=lambda t: -t[3])
    return freq_energy[:top_k_pairs]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="runs/mul_baseline")
    parser.add_argument("--top-k", type=int, default=5,
                        help="Number of top multiplicative frequency pairs to call 'key'.")
    args = parser.parse_args()

    ckpt_path = interp.latest_checkpoint(args.ckpt) if Path(args.ckpt).is_dir() else Path(args.ckpt)
    print(f"Loading: {ckpt_path}")
    model, ds, ckpt = interp.load_run(ckpt_path)
    print(f"  task={ds.task}, p={ds.p}, epoch={ckpt['epoch']}")

    p = ds.p
    W_E_values = model.embed.W_E.detach().double()[:, :p]

    mul_basis, mul_names, g = interp.multiplicative_fourier_basis(p)
    print(f"  primitive root g={g}")

    # Identify top-K key frequencies (cos/sin pairs)
    key_freqs = find_key_frequencies(W_E_values, mul_basis, top_k_pairs=args.top_k)
    key_indices = set()
    print(f"\nTop-{args.top_k} key frequencies in the multiplicative basis:")
    for k, ci, si, e in key_freqs:
        key_indices.add(ci)
        if si is not None:
            key_indices.add(si)
        sin_str = f"+ idx {si}" if si is not None else "(no sin pair, n/2 frequency)"
        print(f"  frequency {k:3d}: cos idx {ci:3d} {sin_str}, total energy={e:.4f}")

    # Baseline
    tr0, te0, ac0 = evaluate_loss(model, ds)
    print(f"\nBaseline: train_loss={tr0:.3e}, test_loss={te0:.3e}, accuracy={ac0:.4f}")

    # Key-only: keep only the top-K cos/sin pairs (plus delta_0 and const, since those
    # are also part of the embedding's natural structure even if low-energy)
    keep_key = torch.zeros(p, dtype=torch.bool)
    for ci in key_indices:
        keep_key[ci] = True
    model_key = ablate_embedding(model, mul_basis, keep_key)
    tr_k, te_k, ac_k = evaluate_loss(model_key, ds)
    print(f"\nKey-only (top {args.top_k} freqs, {keep_key.sum().item()} basis vectors kept):")
    print(f"  train_loss={tr_k:.3e}, test_loss={te_k:.3e}, accuracy={ac_k:.4f}")

    # Anti-key: keep everything EXCEPT the top-K cos/sin pairs
    keep_anti = ~keep_key
    model_anti = ablate_embedding(model, mul_basis, keep_anti)
    tr_a, te_a, ac_a = evaluate_loss(model_anti, ds)
    print(f"\nAnti-key (everything except top-{args.top_k}, {keep_anti.sum().item()} basis vectors kept):")
    print(f"  train_loss={tr_a:.3e}, test_loss={te_a:.3e}, accuracy={ac_a:.4f}")

    chance = float(np.log(ds.n_answer_tokens))
    print(f"\nChance baseline (uniform) = log({ds.n_answer_tokens}) = {chance:.3f}")

    # Summary
    print("\n--- Summary ---")
    print(f"  Baseline test loss:     {te0:.3e}  (accuracy {ac0:.4f})")
    print(f"  Key-only test loss:     {te_k:.3e}  (accuracy {ac_k:.4f})  "
          f"→ prediction: stays near baseline")
    print(f"  Anti-key test loss:     {te_a:.3e}  (accuracy {ac_a:.4f})  "
          f"→ prediction: crashes to >= chance")


if __name__ == "__main__":
    main()
