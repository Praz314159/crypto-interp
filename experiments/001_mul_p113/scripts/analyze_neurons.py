"""Neuron-level analysis: do MLP neurons implement the trig-identity step?

The mod-mul algorithm prediction:
  - Each neuron computes (in some linear combination) terms of the form
        cos(k·(log_g a + log_g b))  or  sin(k·(log_g a + log_g b))
    for a single key frequency k.
  - In the 2D multiplicative Fourier basis (k_a, k_b), each such term is
    concentrated on the 4 "frequency-matched" bigrams:
        cos(k(la+lb)) = cos(k·la)cos(k·lb) - sin(k·la)sin(k·lb)
        sin(k(la+lb)) = sin(k·la)cos(k·lb) + cos(k·la)sin(k·lb)
  - So a neuron specialized to frequency k should have energy concentrated on
    the four bigrams (cos k, cos k), (sin k, sin k), (sin k, cos k), (cos k, sin k).

This script:
  1. Caches MLP post-activations on all p² inputs.
  2. Reshapes to (p, p, d_mlp).
  3. Projects each neuron's (p, p) firing pattern into the 2D multiplicative
     Fourier basis.
  4. For each neuron, identifies its dominant frequency (the k whose four
     "matched" bigrams hold the most energy).
  5. Reports the distribution of dominant frequencies and plots example neurons.
"""

import argparse
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from crypto_interp import interp


def matched_bigrams(k: int, n: int) -> list[tuple[int, int]]:
    """Return the 4 basis-index pairs that carry energy for a neuron with frequency k.

    Layout matches multiplicative_fourier_basis():
        cos k → basis index 2k
        sin k → basis index 2k+1
    For k = n/2 (only when n even), sin doesn't exist; return only (cos, cos).
    """
    ci, si = 2 * k, 2 * k + 1
    if n % 2 == 0 and k == n // 2:
        return [(n, n)]
    return [(ci, ci), (si, si), (ci, si), (si, ci)]


def compute_per_neuron_frequency_energy(coef_2d: torch.Tensor, p: int) -> torch.Tensor:
    """Given 2D Fourier coefficients of shape (p_basis, p_basis, d_mlp),
    return a (n_freqs, d_mlp) array where row k is the energy a neuron carries
    at frequency k (summing over the 4 matched bigrams).
    """
    n = p - 1
    d_mlp = coef_2d.shape[-1]
    n_freqs = (n - 1) // 2 + (1 if n % 2 == 0 else 0)
    # Frequencies are k = 1..(n-1)//2, plus k=n/2 if n even
    freq_indices = list(range(1, (n - 1) // 2 + 1))
    if n % 2 == 0:
        freq_indices.append(n // 2)

    out = torch.zeros(len(freq_indices), d_mlp, dtype=coef_2d.dtype, device=coef_2d.device)
    for i, k in enumerate(freq_indices):
        bigrams = matched_bigrams(k, n)
        e = sum((coef_2d[a, b] ** 2) for a, b in bigrams)
        out[i] = e
    return out, freq_indices


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="runs/mul_baseline")
    parser.add_argument("--top-freqs", type=int, default=5,
                        help="How many key frequencies to highlight in the plots.")
    args = parser.parse_args()

    ckpt_path = interp.latest_checkpoint(args.ckpt) if Path(args.ckpt).is_dir() else Path(args.ckpt)
    print(f"Loading: {ckpt_path}")
    model, ds, ckpt = interp.load_run(ckpt_path)
    p = ds.p
    print(f"  task={ds.task}, p={p}, epoch={ckpt['epoch']}")

    # Cache activations
    print("Caching activations on all p² inputs...")
    cache = interp.cache_all(model, ds)

    # MLP post-activations at the final position
    neuron_acts = cache["blocks.0.mlp.hook_post"][:, -1, :].double()  # (p², d_mlp)
    d_mlp = neuron_acts.shape[1]
    neuron_acts_pp = interp.reshape_pp(neuron_acts, p)  # (p, p, d_mlp)
    print(f"  neuron_acts_pp shape: {tuple(neuron_acts_pp.shape)}")

    # Center each neuron (remove DC bias term) — Nanda's standard preprocessing
    neuron_acts_centered = neuron_acts_pp - neuron_acts_pp.mean(dim=(0, 1), keepdim=True)

    # Build multiplicative basis and project
    mul_basis, _, g = interp.multiplicative_fourier_basis(p)
    print(f"  primitive root g={g}")

    coef_2d = interp.project_2d(neuron_acts_centered, mul_basis)  # (p, p, d_mlp)
    print(f"  coef_2d shape: {tuple(coef_2d.shape)}")

    # Total energy per neuron after centering (for normalization)
    total_energy = (coef_2d ** 2).sum(dim=(0, 1))  # (d_mlp,)

    # Per-neuron frequency energy
    freq_energy, freq_indices = compute_per_neuron_frequency_energy(coef_2d, p)  # (n_freqs, d_mlp)

    # For each neuron, fraction of variance explained by its TOP frequency's matched bigrams
    top_freq_energy, top_freq_idx = freq_energy.max(dim=0)
    top_freq_label = torch.tensor([freq_indices[i] for i in top_freq_idx.tolist()])
    frac_explained = top_freq_energy / total_energy.clamp(min=1e-12)

    # How many neurons fall into each frequency
    counts = Counter(top_freq_label.tolist())
    print(f"\nDistribution of neurons by their dominant frequency (top 15):")
    for freq, cnt in counts.most_common(15):
        avg_frac = frac_explained[top_freq_label == freq].mean().item()
        print(f"  freq={freq:3d}:  {cnt:3d} neurons  (avg variance explained by matched bigrams: {avg_frac:.3f})")

    # Identify "key" frequencies from embedding analysis
    W_E_values = model.embed.W_E.detach().double()[:, :p]
    coef_E = torch.einsum("kp,dp->kd", mul_basis, W_E_values)
    energy_E = (coef_E ** 2).sum(dim=1).cpu().numpy()
    n = p - 1
    embed_freq_energy = {}
    for k in range(1, (n - 1) // 2 + 1):
        embed_freq_energy[k] = energy_E[2 * k] + energy_E[2 * k + 1]
    if n % 2 == 0:
        embed_freq_energy[n // 2] = energy_E[n]
    key_freqs = sorted(embed_freq_energy, key=lambda k: -embed_freq_energy[k])[:args.top_freqs]
    print(f"\nKey frequencies (from embedding analysis, top-{args.top_freqs}): {key_freqs}")

    # Cross-check: are the neuron clusters at the same frequencies?
    print(f"Neurons specialized to key frequencies:")
    for k in key_freqs:
        n_neurons = counts.get(k, 0)
        print(f"  freq {k}: {n_neurons} neurons")

    # ----- Plotting -----

    out_dir = Path(ckpt_path).parent

    # Plot 1: histogram of dominant frequencies across neurons
    fig, ax = plt.subplots(figsize=(11, 4.5))
    all_freqs = list(range(1, n // 2 + 1))
    heights = [counts.get(k, 0) for k in all_freqs]
    bars = ax.bar(all_freqs, heights, width=1.0, color='#1f77b4')
    for k in key_freqs:
        if k in all_freqs:
            bars[k - 1].set_color('#d62728')
    ax.set_xlabel('dominant frequency k')
    ax.set_ylabel('# of neurons')
    ax.set_title(f'Neuron dominant-frequency distribution (red = key freqs from embedding)\n'
                 f'task={ds.task}, p={p}, basis=multiplicative Fourier on (Z/{p})*')
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    p1 = out_dir / "neuron_freq_distribution.png"
    fig.savefig(p1, dpi=130)
    print(f"\nSaved {p1}")

    # Plot 2: variance-explained scatter
    fig, ax = plt.subplots(figsize=(11, 4.5))
    is_key = torch.tensor([k.item() in key_freqs for k in top_freq_label])
    ax.scatter(top_freq_label[is_key].numpy(), frac_explained[is_key].numpy(),
               color='#d62728', s=14, label='neurons w/ key dominant freq', alpha=0.8)
    ax.scatter(top_freq_label[~is_key].numpy(), frac_explained[~is_key].numpy(),
               color='#1f77b4', s=14, label='neurons w/ non-key dominant freq', alpha=0.6)
    ax.set_xlabel("neuron's dominant frequency k")
    ax.set_ylabel('fraction of (centered) variance in matched bigrams')
    ax.set_title('Per-neuron: how concentrated is its variance in its dominant frequency?\n'
                 'High value = clean specialization. Red = neurons specialized to key freqs.')
    ax.legend(loc='lower right', fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p2 = out_dir / "neuron_variance_explained.png"
    fig.savefig(p2, dpi=130)
    print(f"Saved {p2}")

    # Plot 3: 2D Fourier heatmaps for a few example neurons specialized to key freqs.
    # Pick the most variance-explained neuron per key frequency.
    fig, axes = plt.subplots(1, min(len(key_freqs), 5), figsize=(3.5 * min(len(key_freqs), 5), 3.6))
    if len(key_freqs) == 1:
        axes = [axes]
    for ax, k in zip(axes, key_freqs[:5]):
        # neurons whose dominant frequency is k
        idxs = (top_freq_label == k).nonzero(as_tuple=True)[0]
        if len(idxs) == 0:
            ax.set_title(f"freq {k}: no neurons")
            ax.axis('off')
            continue
        best = idxs[frac_explained[idxs].argmax()].item()
        heatmap = coef_2d[:, :, best].abs().cpu().numpy()
        im = ax.imshow(heatmap, cmap='magma', interpolation='nearest')
        ax.set_title(f"neuron {best}, dom. freq = {k}\n"
                     f"frac explained = {frac_explained[best]:.3f}")
        ax.set_xlabel('basis index (b axis)')
        ax.set_ylabel('basis index (a axis)')
    fig.suptitle(f'|2D mult-Fourier coefficient| for top example neurons (one per key freq)')
    fig.tight_layout()
    p3 = out_dir / "neuron_2d_fourier_examples.png"
    fig.savefig(p3, dpi=130)
    print(f"Saved {p3}")


if __name__ == "__main__":
    main()
