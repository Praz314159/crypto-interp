"""Detect 'always-firing' neurons that don't specialize to any frequency.

Nanda's mod-add analysis identified a 6th cluster of neurons whose activations
are not well-explained by any single key frequency. Such neurons are always-on
(positive mean) and contribute via quadratic interactions through the MLP bias.

Detection:
  - For each neuron, compute the fraction of its CENTERED variance that lies
    in the matched bigrams of its top frequency.
  - Neurons below a threshold (default 0.05) are "uncategorized".
  - Also check the mean activation — always-firing neurons have positive mean.

Outputs:
  - Updated neuron-cluster distribution table with the new "uncategorized"
    bucket.
  - Scatter plot: mean activation vs matched-bigram fraction explained, colored
    by cluster.
"""

import argparse
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

import interp
from interp.analyze_neurons import compute_per_neuron_frequency_energy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="runs/mul_baseline")
    parser.add_argument("--threshold", type=float, default=0.05,
                        help="Min fraction-explained at the top freq to be categorized.")
    args = parser.parse_args()

    ckpt_path = interp.latest_checkpoint(args.ckpt) if Path(args.ckpt).is_dir() else Path(args.ckpt)
    print(f"Loading: {ckpt_path}")
    model, ds, ckpt = interp.load_run(ckpt_path)
    p = ds.p

    cache = interp.cache_all(model, ds)
    neuron_acts = cache["blocks.0.mlp.hook_post"][:, -1, :].double()  # (p², d_mlp)
    d_mlp = neuron_acts.shape[1]
    neuron_acts_pp = interp.reshape_pp(neuron_acts, p)

    # Mean activation per neuron (uncentered) — always-firing neurons have high mean
    mean_act = neuron_acts_pp.mean(dim=(0, 1)).cpu().numpy()  # (d_mlp,)
    neuron_acts_centered = neuron_acts_pp - neuron_acts_pp.mean(dim=(0, 1), keepdim=True)
    total_centered_energy = (neuron_acts_centered ** 2).sum(dim=(0, 1)).cpu().numpy()

    mul_basis, _, g = interp.multiplicative_fourier_basis(p)
    coef_2d = interp.project_2d(neuron_acts_centered, mul_basis)
    freq_energy, freq_indices = compute_per_neuron_frequency_energy(coef_2d, p)

    top_freq_energy, top_freq_idx = freq_energy.max(dim=0)
    top_freq_label = np.array([freq_indices[i] for i in top_freq_idx.tolist()])
    frac_explained = (top_freq_energy / torch.tensor(total_centered_energy).clamp(min=1e-12)).cpu().numpy()

    # Classify neurons
    uncategorized = frac_explained < args.threshold
    n_uncat = int(uncategorized.sum())
    print(f"\nUsing threshold frac_explained < {args.threshold} for 'uncategorized':")
    print(f"  total neurons: {d_mlp}")
    print(f"  uncategorized: {n_uncat} ({100*n_uncat/d_mlp:.1f}%)")

    # Distribution by frequency (excluding uncategorized)
    cat_mask = ~uncategorized
    print(f"\nCategorized neurons by frequency:")
    counts = Counter(top_freq_label[cat_mask].tolist())
    for freq, cnt in counts.most_common(10):
        avg_frac = frac_explained[cat_mask & (top_freq_label == freq)].mean()
        print(f"  freq={freq:3d}: {cnt:3d} neurons  (avg frac={avg_frac:.3f})")

    # Look at the "uncategorized" neurons
    if n_uncat > 0:
        unc_mean = mean_act[uncategorized]
        unc_total_centered = total_centered_energy[uncategorized]
        print(f"\nUncategorized neurons statistics:")
        print(f"  mean activation: {unc_mean.mean():.4f} (vs categorized: {mean_act[cat_mask].mean():.4f})")
        print(f"  total centered energy: {unc_total_centered.mean():.4f} "
              f"(vs categorized: {total_centered_energy[cat_mask].mean():.4f})")
        # Are they "always firing"? Check fraction whose mean > 0 (after ReLU)
        always_pos = (unc_mean > unc_mean.std()).sum()
        print(f"  uncategorized with mean activation > 1 std: {always_pos}/{n_uncat}")

    # Plot: mean activation vs frac explained, colored by category
    fig, ax = plt.subplots(figsize=(10, 6))
    key_freqs = [41, 22, 17, 18]  # from prior analysis
    colors = {41: '#d62728', 22: '#1f77b4', 17: '#2ca02c', 18: '#9467bd'}
    for freq, c in colors.items():
        mask = cat_mask & (top_freq_label == freq)
        if mask.sum() > 0:
            ax.scatter(mean_act[mask], frac_explained[mask], color=c,
                       label=f'freq {freq} ({mask.sum()} neurons)', s=16, alpha=0.7)
    other_cat = cat_mask & ~np.isin(top_freq_label, key_freqs)
    if other_cat.sum() > 0:
        ax.scatter(mean_act[other_cat], frac_explained[other_cat],
                   color='gray', label=f'other categorized ({other_cat.sum()})',
                   s=16, alpha=0.5)
    if uncategorized.sum() > 0:
        ax.scatter(mean_act[uncategorized], frac_explained[uncategorized],
                   color='black', marker='x',
                   label=f'uncategorized ({uncategorized.sum()})', s=30)
    ax.axhline(args.threshold, color='black', ls=':', lw=0.8, alpha=0.5,
               label=f'threshold = {args.threshold}')
    ax.set_xlabel('mean activation (uncentered)')
    ax.set_ylabel('fraction of centered variance in top-freq matched bigrams')
    ax.set_title(f'Neuron categorization (task={ds.task}, p={p})\n'
                 f'Black ×: candidates for "always-firing" / quadratic-from-bias cluster')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out = Path(ckpt_path).parent / "neuron_always_firing.png"
    fig.savefig(out, dpi=130)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
