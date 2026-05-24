"""Build and cache the order-multiset dataset for basis-space analysis at p=113.

For each order multiset of size |K| in {3, ..., 7} (drawn from the character
orders of (Z/p)* = {2, 4, 7, 8, 14, 16, 28, 56, 112}), compute:
  - the maximum worst-case logit gap over all realizing bases (exhaustive if
    cheap, sampled with a fixed seed otherwise),
  - one specific basis achieving that gap,
  - cost summary (d_model_cost, d_mlp_cost),
  - the number of distinct bases realizing the multiset.

The cache is saved to ``experiments/003_dmodel_sweep_p113/data/basis_cache.pkl``
as a pandas DataFrame.

This script is slow (a few minutes); subsequent visualizations load the cache
in milliseconds.
"""
from __future__ import annotations

import math
import pickle
import time
from collections import Counter, defaultdict
from itertools import combinations, combinations_with_replacement, product
from pathlib import Path

import numpy as np
import pandas as pd

P = 113
N = P - 1  # 112


# ----- Setup -----

KS = np.arange(1, N)
COS_TABLE = np.cos(2 * np.pi * np.outer(KS, KS) / N)
ORDERS = [2, 4, 7, 8, 14, 16, 28, 56, 112]
PHI = {o: sum(1 for k in range(1, N) if (N // math.gcd(k, N)) == o) for o in ORDERS}
CHARS_BY_ORDER = defaultdict(list)
for k in range(1, N):
    CHARS_BY_ORDER[N // math.gcd(k, N)].append(k)


def gap_for_basis(K_list) -> float:
    """Worst-case logit gap for a basis (list of k values)."""
    L = COS_TABLE[np.array(K_list) - 1].sum(axis=0)
    return float(len(K_list) - L.max())


def best_basis_for_multiset(
    multiset: tuple,
    max_exhaustive: int = 5_000,
    n_samples: int = 4_000,
    rng=None,
) -> tuple:
    """Return (best_gap, best_basis, n_realizing).

    Exhaustive search if the count of realizing bases is <= ``max_exhaustive``,
    else random sampling with ``n_samples`` trials.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    oc = Counter(multiset)
    if any(c > PHI[o] for o, c in oc.items()):
        return None
    options = []
    for o, c in oc.items():
        options.append(list(combinations(CHARS_BY_ORDER[o], c)))
    n_total = 1
    for opts in options:
        n_total *= len(opts)
    best_gap, best_basis = -np.inf, None
    if n_total <= max_exhaustive:
        for combo in product(*options):
            basis = [k for t in combo for k in t]
            g = gap_for_basis(basis)
            if g > best_gap:
                best_gap, best_basis = g, tuple(sorted(basis))
    else:
        for _ in range(n_samples):
            basis = []
            for opts in options:
                basis.extend(opts[rng.integers(0, len(opts))])
            g = gap_for_basis(basis)
            if g > best_gap:
                best_gap, best_basis = g, tuple(sorted(basis))
    return best_gap, best_basis, n_total


def main():
    out_dir = Path(__file__).resolve().parents[1] / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "basis_cache.pkl"

    sizes = [3, 4, 5, 6, 7]
    print(f"Enumerating multisets for sizes {sizes} from orders {ORDERS}")
    rng = np.random.default_rng(0)
    rows = []
    overall_start = time.time()
    for size in sizes:
        sz_start = time.time()
        count_total, count_feasible = 0, 0
        for ms in combinations_with_replacement(ORDERS, size):
            count_total += 1
            oc = Counter(ms)
            if any(c > PHI[o] for o, c in oc.items()):
                continue
            count_feasible += 1
            result = best_basis_for_multiset(ms, rng=rng)
            if result is None:
                continue
            best_gap, best_basis, n_real = result
            has_legendre = (2 in oc)
            n_prim = oc.get(112, 0)
            d_model_cost = 2 * size - (1 if has_legendre else 0)
            d_mlp_cost = sum(ms)
            rows.append({
                "multiset": tuple(sorted(ms, reverse=True)),
                "size": size,
                "best_gap": best_gap,
                "best_basis": best_basis,
                "n_realizing": n_real,
                "d_model_cost": d_model_cost,
                "d_mlp_cost": d_mlp_cost,
                "n_primitive": n_prim,
                "has_legendre": has_legendre,
                "n_order2": oc.get(2, 0),
                "n_order4": oc.get(4, 0),
                "n_order7": oc.get(7, 0),
                "n_order8": oc.get(8, 0),
                "n_order14": oc.get(14, 0),
                "n_order16": oc.get(16, 0),
                "n_order28": oc.get(28, 0),
                "n_order56": oc.get(56, 0),
                "n_order112": oc.get(112, 0),
            })
        sz_elapsed = time.time() - sz_start
        print(f"  size={size}: {count_feasible}/{count_total} feasible multisets, "
              f"{sz_elapsed:.1f}s")

    df = pd.DataFrame(rows)
    print(f"\nTotal rows: {len(df)}")
    print(f"Total time: {time.time() - overall_start:.1f}s")
    print("\nSaving to:", out_path)
    with open(out_path, "wb") as f:
        pickle.dump(df, f)
    print(f"Cache file size: {out_path.stat().st_size / 1024:.1f} KB")

    # Brief summary
    print("\nQuick stats:")
    print(f"  multisets with gap > 0.5:  {(df['best_gap'] > 0.5).sum()}")
    print(f"  multisets with gap > 1.5:  {(df['best_gap'] > 1.5).sum()}")
    print(f"  multisets with gap > 2.5:  {(df['best_gap'] > 2.5).sum()}")
    print(f"  multisets with gap > 3.5:  {(df['best_gap'] > 3.5).sum()}")
    print(f"  multisets containing Legendre: {df['has_legendre'].sum()}")
    print(f"  multisets with ≥1 primitive: {(df['n_primitive'] >= 1).sum()}")


if __name__ == "__main__":
    main()
