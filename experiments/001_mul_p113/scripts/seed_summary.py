"""Summarize structural findings across seeds.

For each run directory (e.g. runs/mul_baseline, runs/mul_seed1, ...) print:
  - key frequencies (top-N from embedding)
  - top-N variance explained
  - neuron-cluster sizes per top-N frequency
  - matched-bigram fraction per cluster
  - always-firing cluster size

Use to confirm structural invariance across seeds.
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from crypto_interp import interp
from crypto_interp.interp import compute_per_neuron_frequency_energy


def summarize(run_dir: Path, top_n: int = 5):
    ckpt_path = interp.latest_checkpoint(run_dir)
    model, ds, ckpt = interp.load_run(ckpt_path)
    p = ds.p
    seed = ckpt["config"].get("seed", "?")

    # Baseline accuracy
    inputs = ds.inputs
    labels = ds.labels
    with torch.no_grad():
        logits = model(inputs)[:, -1, : ds.n_answer_tokens]
    acc = float((logits.argmax(-1) == labels).float().mean())

    # Embedding analysis
    W_E_values = model.embed.W_E.detach().double()[:, :p]
    mul_basis, _, g = interp.multiplicative_fourier_basis(p)
    coef_E = torch.einsum("kp,dp->kd", mul_basis, W_E_values)
    energy_E_basis = (coef_E ** 2).sum(dim=1).cpu().numpy()
    n = p - 1
    freq_E = {k: energy_E_basis[2*k] + energy_E_basis[2*k+1]
              for k in range(1, (n - 1) // 2 + 1)}
    if n % 2 == 0:
        freq_E[n // 2] = energy_E_basis[n]
    top_E = sorted(freq_E, key=lambda k: -freq_E[k])
    total_E = sum(freq_E.values()) + energy_E_basis[0] + energy_E_basis[1]

    # Neuron analysis
    cache = interp.cache_all(model, ds)
    acts = cache["blocks.0.mlp.hook_post"][:, -1, :].double()
    acts_pp = interp.reshape_pp(acts, p)
    mean_act = acts_pp.mean(dim=(0, 1)).cpu().numpy()
    acts_c = acts_pp - acts_pp.mean(dim=(0, 1), keepdim=True)
    total_centered_energy = (acts_c ** 2).sum(dim=(0, 1)).cpu().numpy()
    coef_2d = interp.project_2d(acts_c, mul_basis)
    freq_energy_neurons, freq_indices = compute_per_neuron_frequency_energy(coef_2d, p)
    top_e, top_idx = freq_energy_neurons.max(dim=0)
    top_freq_label = np.array([freq_indices[i] for i in top_idx.tolist()])
    frac = (top_e / torch.tensor(total_centered_energy).clamp(min=1e-12)).cpu().numpy()

    dead = (total_centered_energy < 100) & (mean_act < 0.1)
    always_firing = (~dead) & (frac < 0.15) & (mean_act > 0.5) & (total_centered_energy > 1000)

    print(f"\n=== {run_dir.name} (seed={seed}, epoch={ckpt['epoch']}, "
          f"accuracy={acc:.4f}, primitive root g={g}) ===")
    print(f"Top {top_n} embedding frequencies:")
    cum = 0
    for r, k in enumerate(top_E[:top_n]):
        cum += freq_E[k]
        print(f"  rank {r+1}: freq {k:3d}  energy={freq_E[k]:7.4f}  cumulative={cum/total_E:.4f}")

    print(f"Neuron-cluster sizes by top-{top_n} embedding frequencies:")
    for k in top_E[:top_n]:
        cnt = int((top_freq_label == k).sum())
        print(f"  freq {k:3d}: {cnt} neurons")

    print(f"Categories: specialized={int(((~dead) & (~always_firing)).sum())}, "
          f"dead={int(dead.sum())}, always-firing={int(always_firing.sum())}")
    return {
        "seed": seed,
        "g": g,
        "top_frequencies": top_E[:top_n],
        "freq_energies": {k: float(freq_E[k]) for k in top_E[:top_n]},
        "neuron_cluster_sizes": {k: int((top_freq_label == k).sum()) for k in top_E[:top_n]},
        "always_firing": int(always_firing.sum()),
        "dead": int(dead.sum()),
        "accuracy": acc,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=str, nargs="+", required=True,
                        help="Run directories to summarize.")
    parser.add_argument("--top-n", type=int, default=5)
    args = parser.parse_args()

    summaries = []
    for r in args.runs:
        summaries.append(summarize(Path(r), top_n=args.top_n))

    print("\n\n=== Cross-seed summary ===")
    print(f"{'Run':<30}{'g':<5}{'Acc':<10}{'Top frequencies':<35}{'Cluster sizes':<35}{'AF':<5}{'Dead':<5}")
    for r, s in zip(args.runs, summaries):
        freqs = ",".join(map(str, s["top_frequencies"]))
        sizes = ",".join(f"{k}:{s['neuron_cluster_sizes'][k]}" for k in s["top_frequencies"])
        print(f"{Path(r).name:<30}{s['g']:<5}{s['accuracy']:<10.4f}{freqs:<35}{sizes:<35}"
              f"{s['always_firing']:<5}{s['dead']:<5}")


if __name__ == "__main__":
    main()
