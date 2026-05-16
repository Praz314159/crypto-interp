"""Order-class analysis: does the model show Pohlig-Hellman-style decomposition?

For a multiplicative-Fourier basis over Z/(p-1), each character has an
"order" — the multiplicative order in Z/(p-1) of the frequency index k,
equal to (p-1) / gcd(k, p-1).

For Pohlig-Hellman, the natural decomposition uses characters whose orders are
*pure prime powers* — orders divide only one prime power factor of p-1.

For p=113, p-1 = 112 = 2^4 * 7:
  - "Pure 2-part" characters: order in {1, 2, 4, 8, 16}
  - "Pure 7-part" characters: order in {1, 7}
  - "Mixed" characters: order has both 2- and 7-factors (e.g., 14, 28, 56, 112)

A "Pohlig-Hellman-like" model would heavily use pure-prime-power characters
and avoid mixed ones. A random/non-PH model would mostly use mixed characters
(since most characters have full-order = mixed in the smooth case).

This script:
  1. For each model: identifies each neuron's dominant frequency.
  2. Classifies each frequency by its order class.
  3. Reports the fraction of neurons in "pure" vs "mixed" classes.
  4. Aggregates across multiple seeds for statistical confidence.
"""

import argparse
import math
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

import interp
from interp.analyze_neurons import compute_per_neuron_frequency_energy


def factorize(n: int) -> dict[int, int]:
    """Return prime factorization as a dict {prime: exponent}."""
    out = {}
    d = 2
    while d * d <= n:
        while n % d == 0:
            out[d] = out.get(d, 0) + 1
            n //= d
        d += 1
    if n > 1:
        out[n] = out.get(n, 0) + 1
    return out


def classify_order(order: int, factorization: dict[int, int]) -> str:
    """Classify a character order by which prime-power factors of n it touches.

    Returns a label like 'pure_2' or 'pure_7' or 'mixed_2x7'.
    """
    order_factors = factorize(order)
    primes_present = sorted(order_factors.keys())
    if len(primes_present) == 0:
        return "trivial"  # order 1, the constant
    if len(primes_present) == 1:
        return f"pure_{primes_present[0]}"
    return "mixed_" + "x".join(map(str, primes_present))


def analyze_one_model(ckpt_path: Path):
    model, ds, ckpt = interp.load_run(ckpt_path)
    p = ds.p
    n = p - 1
    n_factors = factorize(n)

    cache = interp.cache_all(model, ds)
    neuron_acts = cache["blocks.0.mlp.hook_post"][:, -1, :].double()
    neuron_acts_pp = interp.reshape_pp(neuron_acts, p)
    mean_act = neuron_acts_pp.mean(dim=(0, 1)).cpu().numpy()
    centered = neuron_acts_pp - neuron_acts_pp.mean(dim=(0, 1), keepdim=True)
    total_e = (centered ** 2).sum(dim=(0, 1)).cpu().numpy()

    mul_basis, _, g = interp.multiplicative_fourier_basis(p)
    coef_2d = interp.project_2d(centered, mul_basis)
    freq_energy, freq_list = compute_per_neuron_frequency_energy(coef_2d, p)
    top_e, top_idx = freq_energy.max(dim=0)
    top_freq = np.array([freq_list[i] for i in top_idx.tolist()])
    frac = (top_e / torch.tensor(total_e).clamp(min=1e-12)).cpu().numpy()

    # Classify each neuron by its dominant frequency's order class.
    dead = (total_e < 100) & (mean_act < 0.1)
    always_firing = (~dead) & (frac < 0.15) & (mean_act > 0.5) & (total_e > 1000)

    classes_per_neuron = []
    for ki in range(len(top_freq)):
        if dead[ki]:
            classes_per_neuron.append("dead")
        elif always_firing[ki]:
            classes_per_neuron.append("always_firing")
        else:
            order = n // math.gcd(int(top_freq[ki]), n)
            classes_per_neuron.append(classify_order(order, n_factors))

    cls_counter = Counter(classes_per_neuron)

    seed = ckpt["config"].get("seed", "?")
    return {
        "seed": seed,
        "p": p,
        "n": n,
        "n_factors": n_factors,
        "top_freqs": top_freq.tolist(),
        "class_counts": dict(cls_counter),
        "primitive_root": g,
        "n_specialized": int(((~dead) & (~always_firing)).sum()),
        "n_total": int(len(top_freq)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=str, nargs="+", required=True,
                        help="Run directories or checkpoint files.")
    parser.add_argument("--out", type=str, default="runs/order_class_analysis.png")
    args = parser.parse_args()

    summaries = []
    for r in args.runs:
        rp = Path(r)
        if rp.is_dir():
            ckpt_path = interp.latest_checkpoint(rp)
        else:
            ckpt_path = rp
        s = analyze_one_model(ckpt_path)
        summaries.append(s)
        print(f"\n=== seed={s['seed']}, p={s['p']} (p-1={s['n']}, factors={s['n_factors']}) ===")
        print(f"  primitive root g={s['primitive_root']}, n_specialized={s['n_specialized']}/{s['n_total']}")
        print(f"  Top key freqs: {s['top_freqs'][:10]}")
        print(f"  Order-class counts:")
        # Sort classes: trivial, pure_*, mixed_*, always_firing, dead
        order_keys = sorted(s["class_counts"].keys(),
                            key=lambda k: (0 if k == "trivial" else
                                           1 if k.startswith("pure_") else
                                           2 if k.startswith("mixed_") else
                                           3 if k == "always_firing" else
                                           4))
        for cls in order_keys:
            c = s["class_counts"][cls]
            print(f"    {cls:18s}: {c:4d}  ({100*c/s['n_total']:.1f}%)")

    # Aggregate across seeds
    print("\n\n=== Aggregate across seeds ===")
    all_classes: dict[str, list[int]] = defaultdict(list)
    for s in summaries:
        for cls in set().union(*[set(t["class_counts"].keys()) for t in summaries]):
            all_classes[cls].append(s["class_counts"].get(cls, 0))

    order_keys = sorted(all_classes.keys(),
                        key=lambda k: (0 if k == "trivial" else
                                       1 if k.startswith("pure_") else
                                       2 if k.startswith("mixed_") else
                                       3 if k == "always_firing" else
                                       4))
    print(f"{'class':<20} {'mean':<10} {'std':<10} {'min':<6} {'max':<6}")
    for cls in order_keys:
        vals = np.array(all_classes[cls])
        print(f"{cls:<20} {vals.mean():<10.1f} {vals.std():<10.1f} "
              f"{vals.min():<6d} {vals.max():<6d}")

    # Pohlig-Hellman score: fraction of (non-dead/AF) neurons in "pure" classes
    print("\nPohlig-Hellman score per seed (frac of specialized neurons in pure-subgroup classes):")
    for s in summaries:
        n_pure = sum(c for cls, c in s["class_counts"].items() if cls.startswith("pure_"))
        n_mixed = sum(c for cls, c in s["class_counts"].items() if cls.startswith("mixed_"))
        n_spec = s["n_specialized"]
        ph_score = n_pure / max(n_spec, 1)
        print(f"  seed {s['seed']:2}: pure={n_pure:3}, mixed={n_mixed:3}, specialized={n_spec}, "
              f"PH-score={ph_score:.3f}")

    # Plot: per-seed, bar chart of class counts
    fig, ax = plt.subplots(figsize=(max(10, 0.6 * len(summaries) * len(order_keys) + 4), 5))
    width = 0.8 / len(order_keys)
    x = np.arange(len(summaries))
    palette = {
        "trivial": "#888888",
        "pure_2": "#1f77b4",
        "pure_3": "#aec7e8",
        "pure_5": "#9467bd",
        "pure_7": "#2ca02c",
        "mixed_2x7": "#d62728",
        "mixed_2x3": "#ff7f0e",
        "mixed_2x3x5": "#bcbd22",
        "mixed_2x3x5x7": "#17becf",
        "always_firing": "#7f7f7f",
        "dead": "#000000",
    }
    for i, cls in enumerate(order_keys):
        color = palette.get(cls, "#cccccc")
        vals = [s["class_counts"].get(cls, 0) for s in summaries]
        ax.bar(x + i * width - 0.4 + width / 2, vals, width=width, color=color, label=cls)
    ax.set_xticks(x)
    ax.set_xticklabels([f"seed {s['seed']}" for s in summaries], rotation=0)
    ax.set_ylabel("# neurons")
    ax.set_title(f"Order-class decomposition per seed (p={summaries[0]['p']}, "
                 f"p-1 factors={summaries[0]['n_factors']})")
    ax.legend(loc='upper right', fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print(f"\nSaved plot: {args.out}")


if __name__ == "__main__":
    main()
