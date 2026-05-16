"""Cross-seed trajectory analysis for the multi-seed sweep.

For each seed, loads metrics.pt and answers:
  - Which frequencies are 'key' (top-K by final embedding energy)?
  - When did each key frequency first cross an amplification threshold?
  - Is there a relationship between the algebraic *order* of a frequency
    (its order in Z/(p-1)) and the probability it becomes a key frequency
    and/or how early it amplifies?

Hypotheses (tested with simple statistics):
  H1: Order-2 characters (e.g. frequency 56 on Z/112 — the Legendre character)
      are overrepresented in the top-K key list vs. a uniform-random baseline.
  H2: Low-order characters amplify earlier in training.
  H3: Early-amplified frequencies end up with more neurons (territory effect).
  H5: The number of key frequencies (~4-5) is invariant across seeds.
"""

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def gcd(a: int, b: int) -> int:
    return math.gcd(a, b)


def order_in_Zn(k: int, n: int) -> int:
    """Order of element k in the additive group Z/n. Equals n / gcd(k, n)."""
    return n // gcd(k, n)


def load_sweep(sweep_glob: str) -> list[dict]:
    """Load all metrics.pt files matching the pattern. Returns a list of dicts."""
    out = []
    for p in sorted(Path(".").glob(sweep_glob)):
        metrics_path = p / "metrics.pt"
        if not metrics_path.exists():
            continue
        m = torch.load(metrics_path, weights_only=False)
        m["run_dir"] = p
        out.append(m)
    return out


def find_amplification_epoch(energy_traj: np.ndarray, epochs: list[int],
                              threshold_ratio: float = 0.5) -> int | None:
    """Find first epoch where energy >= threshold_ratio * (max over the trajectory).

    Returns None if the frequency never crosses the threshold.
    """
    max_e = energy_traj.max()
    if max_e <= 0:
        return None
    threshold = threshold_ratio * max_e
    above = np.where(energy_traj >= threshold)[0]
    if len(above) == 0:
        return None
    return epochs[above[0]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--glob", type=str, default="runs/mul_sweep_seed*",
                        help="Glob pattern for sweep run directories.")
    parser.add_argument("--top-K", type=int, default=5,
                        help="Top-K frequencies per seed counted as 'key'.")
    parser.add_argument("--amplify-thresh", type=float, default=0.5,
                        help="Fraction-of-final to count as 'amplified'.")
    parser.add_argument("--out-dir", type=str, default="runs",
                        help="Where to write summary plots.")
    args = parser.parse_args()

    sweep = load_sweep(args.glob)
    if not sweep:
        print(f"No metrics found matching {args.glob}")
        return
    n_seeds = len(sweep)
    p = sweep[0]["config"]["p"]
    n = p - 1
    print(f"Loaded {n_seeds} seeds for p={p} (n=p-1={n})")

    # Per-seed: top-K key freqs (by final energy), amplification time per key freq
    per_seed_info = []
    for m in sweep:
        epochs = list(m["epochs"])
        energies = m["freq_energies"]  # (n_steps, n_freqs)
        final = energies[-1]
        n_freqs = len(final)
        # Frequencies k = 1..(n-1)//2 (and n/2 if n even). Frequency at index i corresponds to k=i+1.
        freq_list = list(range(1, (n - 1) // 2 + 1)) + ([n // 2] if n % 2 == 0 else [])

        ranked = sorted(range(n_freqs), key=lambda i: -final[i])
        key_indices = ranked[:args.top_K]
        key_freqs = [freq_list[i] for i in key_indices]

        amp_times = {}
        for ki, kf in zip(key_indices, key_freqs):
            amp_times[kf] = find_amplification_epoch(
                energies[:, ki], epochs, threshold_ratio=args.amplify_thresh
            )

        per_seed_info.append({
            "seed": m["config"]["seed"],
            "key_freqs": key_freqs,
            "amp_times": amp_times,
            "final_energies": {kf: float(final[ki]) for ki, kf in zip(key_indices, key_freqs)},
            "epochs": epochs,
            "energies": energies,
            "freq_list": freq_list,
        })

    print(f"\nPer-seed key frequencies (top-{args.top_K}):")
    for s in per_seed_info:
        amp_str = ", ".join(f"{kf}@{s['amp_times'][kf]}" for kf in s["key_freqs"])
        print(f"  seed {s['seed']:2d}: freqs = {s['key_freqs']}, amplification = {amp_str}")

    # ---- Aggregate analysis ----
    all_key_freqs = [kf for s in per_seed_info for kf in s["key_freqs"]]
    freq_counter = Counter(all_key_freqs)

    print(f"\nFrequencies appearing as 'key' across seeds:")
    print(f"{'freq':>5} {'count':>6} {'frac':>8} {'gcd(k, n)':>11} {'order':>7}")
    for kf, c in freq_counter.most_common():
        print(f"{kf:5d} {c:6d} {c/n_seeds:8.3f} {gcd(kf, n):11d} {order_in_Zn(kf, n):7d}")

    # ----- Hypothesis tests -----
    print("\n--- Hypothesis tests ---")

    # H1: Order-2 characters overrepresented?
    n_freqs_total = (n - 1) // 2 + (1 if n % 2 == 0 else 0)
    order2_freqs = [kf for kf in range(1, n // 2 + 1) if order_in_Zn(kf, n) == 2]
    expected_per_freq = args.top_K / n_freqs_total
    order2_counts = {kf: freq_counter.get(kf, 0) for kf in order2_freqs}
    print(f"\nH1: Order-2 characters in Z/{n}:")
    print(f"  Frequencies of order 2: {order2_freqs}")
    for kf, c in order2_counts.items():
        observed_rate = c / n_seeds
        print(f"    freq {kf}: observed in {c}/{n_seeds} ({observed_rate:.3f}); "
              f"baseline = {expected_per_freq:.3f}")
    # Simple z-test for each
    for kf, c in order2_counts.items():
        p_baseline = expected_per_freq
        expected = p_baseline * n_seeds
        # Variance of binomial
        var = n_seeds * p_baseline * (1 - p_baseline)
        if var > 0:
            z = (c - expected) / math.sqrt(var)
            print(f"    z-score for freq {kf}: {z:+.2f}")

    # H2: low-order freqs amplify earlier?
    print("\nH2: Amplification time vs order:")
    rows = []
    for s in per_seed_info:
        for kf in s["key_freqs"]:
            t = s["amp_times"][kf]
            if t is not None:
                rows.append((kf, order_in_Zn(kf, n), gcd(kf, n), t))
    if rows:
        orders = np.array([r[1] for r in rows])
        times = np.array([r[3] for r in rows])
        # Spearman rank correlation: simple Pearson on ranks
        order_ranks = np.argsort(np.argsort(orders))
        time_ranks = np.argsort(np.argsort(times))
        corr = np.corrcoef(order_ranks, time_ranks)[0, 1]
        print(f"  Spearman corr(order, amplification time) = {corr:+.3f}")
        print(f"  (Negative = low order → late amplification; positive = low order → early)")
        # Group by order
        unique_orders = sorted(set(orders))
        print(f"  Mean amplification time by order:")
        for o in unique_orders:
            mask = orders == o
            if mask.sum() > 0:
                print(f"    order {o:4d}: n={mask.sum():3d}, mean t = {times[mask].mean():7.0f}")

    # H5: number of "significant" frequencies invariant?
    print("\nH5: How many frequencies are 'significant' per seed?")
    # Define significant as freq with final energy > some fraction of the max
    for s in per_seed_info:
        final = np.array([s["final_energies"][kf] for kf in s["key_freqs"]])
        if len(final) == 0:
            continue
        max_e = final.max()
        n_sig = (final > 0.1 * max_e).sum()
        print(f"  seed {s['seed']:2d}: {n_sig} sig freqs (out of top-{args.top_K}); "
              f"energy ratios = {[f'{e/max_e:.2f}' for e in final]}")

    # ---- Plots ----
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    # Plot 1: heatmap of frequency-amplification time across seeds
    grid_freqs = sorted(set(all_key_freqs))
    fig, ax = plt.subplots(figsize=(max(14, 0.4 * len(grid_freqs) + 2), max(5, 0.35 * n_seeds + 1.5)))
    matrix = np.full((n_seeds, len(grid_freqs)), np.nan)
    for r, s in enumerate(per_seed_info):
        for kf in s["key_freqs"]:
            if kf in grid_freqs and s["amp_times"][kf] is not None:
                c = grid_freqs.index(kf)
                matrix[r, c] = s["amp_times"][kf]
    im = ax.imshow(matrix, cmap='viridis', aspect='auto')
    ax.set_yticks(range(n_seeds))
    ax.set_yticklabels([f"seed {s['seed']}" for s in per_seed_info])
    ax.set_xticks(range(len(grid_freqs)))
    # Two-row label, vertical
    labels = []
    for k in grid_freqs:
        ord_k = order_in_Zn(k, n)
        labels.append(f"{k}\n(ord {ord_k})")
    ax.set_xticklabels(labels, rotation=90, fontsize=8)
    ax.set_title(f'Amplification epoch per key frequency, across {n_seeds} seeds\n'
                 f'(empty = not a top-{args.top_K} frequency for that seed)')
    plt.colorbar(im, ax=ax, label='epoch of amplification')
    fig.tight_layout()
    p1 = out_dir / "trajectory_heatmap.png"
    fig.savefig(p1, dpi=130)
    print(f"\nSaved {p1}")

    # Plot 2: scatter of order vs amplification time
    if rows:
        fig, ax = plt.subplots(figsize=(9, 5))
        orders_arr = np.array([r[1] for r in rows])
        times_arr = np.array([r[3] for r in rows])
        ax.scatter(orders_arr, times_arr, s=30, alpha=0.6)
        ax.set_xlabel(f'order of character in Z/{n}')
        ax.set_ylabel('amplification epoch')
        ax.set_title(f'Amplification time vs character order (Spearman ρ = {corr:+.3f})\n'
                     f'(Each point is one key frequency in one seed)')
        ax.set_xscale('log')
        # Orders must divide n; show only those that actually appear in the data.
        divisors_n = sorted({d for d in range(1, n + 1) if n % d == 0})
        present_orders = sorted(set(int(o) for o in orders_arr))
        tick_orders = [d for d in divisors_n if d in present_orders]
        ax.set_xticks(tick_orders)
        ax.set_xticklabels([str(t) for t in tick_orders])
        ax.get_xaxis().set_minor_locator(plt.NullLocator())
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        p2 = out_dir / "trajectory_order_vs_time.png"
        fig.savefig(p2, dpi=130)
        print(f"Saved {p2}")

    # Plot 3: count of times each frequency was 'key' across seeds
    fig, ax = plt.subplots(figsize=(13, 4))
    freqs_for_bar = sorted(freq_counter)
    counts = [freq_counter[k] for k in freqs_for_bar]
    colors = ['#d62728' if order_in_Zn(k, n) == 2
              else '#ff7f0e' if order_in_Zn(k, n) <= 8
              else '#1f77b4' for k in freqs_for_bar]
    ax.bar(freqs_for_bar, counts, color=colors, width=0.9)
    ax.set_xlabel('frequency k')
    ax.set_ylabel(f'# seeds (out of {n_seeds}) where k is in top-{args.top_K}')
    ax.set_title(f'Frequency selection across seeds (red = order 2, orange = order ≤ 8, blue = other)')
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    p3 = out_dir / "trajectory_freq_counts.png"
    fig.savefig(p3, dpi=130)
    print(f"Saved {p3}")


if __name__ == "__main__":
    main()
