"""Lattice-variation analysis: apply the basis/economy verbs at each prime
in our population and tabulate K, character orders, CRT coverage, and helper
multipliers.

Tests two predictions from ``playbooks/lattice_variation.md``:

1. **CRT minimality.** For every grokked seed at every prime, K should
   contain at least one character whose order is divisible by each
   prime-power factor of p−1.
2. **Doubling-economy prime-invariance.** ×2 helper multipliers should
   appear at every prime. (Caveat: the doubling economy is a *capacity*
   effect — at loose d_mlp it may not be invoked even where it would be
   available, so absence at d_mlp=32 isn't falsification.)

Usage:
    python -m crypto_interp.analysis.lattice_variation \\
        --out-dir experiments/lattice_variation/
"""
from __future__ import annotations

import argparse
import csv
import glob
import math
from collections import Counter
from pathlib import Path

from crypto_interp.interp import Session


# Map of prime -> globbed run directories. Edit if layout changes.
PRIME_RUNS = {
    113: "experiments/003_dmodel_sweep_p113/runs/dmodel_24_dmlp_20_wd2_seed*",
    127: "experiments/004_p127/runs/dmodel_24_dmlp_32_seed*",
    181: "experiments/005_p181/runs/dmodel_24_dmlp_32_seed*",
}


def prime_factorize(n: int) -> list[tuple[int, int]]:
    """Return [(prime, exponent), ...] for n's prime factorization."""
    out, d = [], 2
    while d * d <= n:
        e = 0
        while n % d == 0:
            n //= d
            e += 1
        if e:
            out.append((d, e))
        d += 1
    if n > 1:
        out.append((n, 1))
    return out


def crt_check(S: Session) -> dict:
    """Compute K, character orders, factorization of p-1, and CRT coverage."""
    K = sorted(int(k) for k in S.essential()["K"])
    factors = prime_factorize(S.ds.p - 1)
    orders = [S.order(k) for k in K]
    covers = {
        f"{q}^{e}": any(o % (q ** e) == 0 for o in orders)
        for q, e in factors
    }
    return {
        "K": K,
        "orders": orders,
        "factors": factors,
        "covers": covers,
        "all_covered": all(covers.values()),
    }


def analyze_run(run_dir: Path) -> dict | None:
    """Load one run, filter for grokked, return per-seed summary or None."""
    try:
        S = Session.from_run(str(run_dir))
    except Exception as exc:
        return {"run_dir": str(run_dir), "error": f"load: {exc}"}
    try:
        train_loss, test_loss, accuracy = S.evaluate()
    except Exception as exc:
        return {"run_dir": str(run_dir), "error": f"evaluate: {exc}"}
    if test_loss > 1e-2 or accuracy < 0.99:
        return {
            "run_dir": str(run_dir),
            "p": S.ds.p,
            "test_loss": float(test_loss),
            "accuracy": float(accuracy),
            "grokked": False,
        }
    chk = crt_check(S)
    helpers = S.helpers(chk["K"])
    # find_primary_helper_pairs returns one (helper_k, primary_k, mult, energy)
    # per helper (primaries are filtered out inside the verb). mult may be
    # None when the helper has no small-power relationship to its primary.
    helper_mults = [int(m) for (_h, _p, m, _e) in helpers if m is not None]
    return {
        "run_dir": str(run_dir),
        "p": S.ds.p,
        "test_loss": float(test_loss),
        "accuracy": float(accuracy),
        "grokked": True,
        "K": chk["K"],
        "orders": chk["orders"],
        "factors": chk["factors"],
        "covers": chk["covers"],
        "all_covered": chk["all_covered"],
        "helper_mults": helper_mults,
    }


def summarize(per_prime: dict[int, list[dict]]) -> None:
    """Print human-readable summary of CRT-minimality + helper findings."""
    print("\n" + "=" * 72)
    print("LATTICE-VARIATION RESULTS")
    print("=" * 72)
    for p, rows in per_prime.items():
        grokked = [r for r in rows if r.get("grokked")]
        loaded = [r for r in rows if "error" not in r]
        print(f"\n--- p = {p}  (p-1 = "
              f"{' · '.join(f'{q}^{e}' if e > 1 else str(q) for q, e in prime_factorize(p - 1))}) ---")
        print(f"  runs found: {len(rows)}  loaded: {len(loaded)}  grokked: {len(grokked)}")
        if not grokked:
            print("  (no grokked seeds at this prime)")
            continue
        # CRT coverage
        all_covered = sum(1 for r in grokked if r["all_covered"])
        print(f"  CRT minimality:  {all_covered}/{len(grokked)} seeds cover every factor")
        # K stats
        k_sizes = [len(r["K"]) for r in grokked]
        print(f"  |K| distribution:  min={min(k_sizes)} median={sorted(k_sizes)[len(k_sizes)//2]} max={max(k_sizes)}")
        # Helper-mult distribution
        all_mults = [m for r in grokked for m in r["helper_mults"]]
        mult_ct = Counter(all_mults)
        if mult_ct:
            print(f"  helper multipliers:  {dict(sorted(mult_ct.items()))}")
        else:
            print(f"  helper multipliers:  (none observed — loose capacity?)")
        # Per-seed detail
        print(f"  per-seed:")
        for r in sorted(grokked, key=lambda r: r["run_dir"]):
            seed_name = Path(r["run_dir"]).name
            cov = "✓" if r["all_covered"] else "✗"
            miss = [k for k, v in r["covers"].items() if not v]
            miss_s = f" missing={miss}" if miss else ""
            print(f"    {seed_name:42s}  K={r['K']}  mults={r['helper_mults']}  {cov}{miss_s}")


def write_csv(per_prime: dict[int, list[dict]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["p", "run_dir", "grokked", "test_loss", "accuracy",
                    "K", "orders", "p_minus_1_factors", "all_covered",
                    "missing_factors", "helper_mults"])
        for p, rows in per_prime.items():
            for r in rows:
                if "error" in r:
                    w.writerow([p, r["run_dir"], "ERROR", "", "", "", "", "", "", "", r["error"]])
                    continue
                if not r.get("grokked"):
                    w.writerow([p, r["run_dir"], False, r.get("test_loss", ""),
                                r.get("accuracy", ""), "", "", "", "", "", ""])
                    continue
                miss = [k for k, v in r["covers"].items() if not v]
                w.writerow([
                    p, r["run_dir"], True, r["test_loss"], r["accuracy"],
                    " ".join(map(str, r["K"])),
                    " ".join(map(str, r["orders"])),
                    " ".join(f"{q}^{e}" for q, e in r["factors"]),
                    r["all_covered"],
                    " ".join(miss),
                    " ".join(map(str, r["helper_mults"])),
                ])
    print(f"\nWrote {path}")


def write_plots(per_prime: dict[int, list[dict]], out_dir: Path) -> None:
    """Three figures for the paper:
        (1) grok rate by prime
        (2) |K| distribution per prime
        (3) helper-multiplier histogram (the ×2 monopoly)
    """
    import matplotlib.pyplot as plt  # local import: keep CSV path matplotlib-free
    out_dir.mkdir(parents=True, exist_ok=True)
    primes = sorted(per_prime)

    # Common style
    plt.rcParams.update({"figure.dpi": 120, "font.size": 10})

    # ----- (1) grok rate -----
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    xs = list(range(len(primes)))
    rates, counts, total = [], [], []
    for p in primes:
        rows = per_prime[p]
        loaded = [r for r in rows if "error" not in r]
        grokked = [r for r in loaded if r.get("grokked")]
        rates.append(len(grokked) / max(len(loaded), 1))
        counts.append(len(grokked))
        total.append(len(loaded))
    ax.bar(xs, rates, color="steelblue", alpha=0.85)
    for i, (n, t) in enumerate(zip(counts, total)):
        ax.text(i, rates[i] + 0.02, f"{n}/{t}", ha="center", fontsize=9)
    ax.set_xticks(xs, [f"p={p}\n{_p_minus_1_str(p)}" for p in primes])
    ax.set_ylabel("grok rate (test_loss < 1e-2)")
    ax.set_ylim(0, 1.05)
    ax.set_title("Grokking rate by prime")
    fig.tight_layout()
    fig.savefig(out_dir / "grok_rate.png")
    plt.close(fig)

    # ----- (2) |K| distribution per prime -----
    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    width = 0.8 / max(len(primes), 1)
    all_k_sizes = [
        len(r["K"]) for rows in per_prime.values() for r in rows if r.get("grokked")
    ]
    if all_k_sizes:
        kmin, kmax = min(all_k_sizes), max(all_k_sizes)
        bins = list(range(kmin, kmax + 2))
        for i, p in enumerate(primes):
            sizes = [len(r["K"]) for r in per_prime[p] if r.get("grokked")]
            counts_, _ = _hist_counts(sizes, bins)
            xs = [b + (i - (len(primes) - 1) / 2) * width for b in bins[:-1]]
            ax.bar(xs, counts_, width=width, label=f"p={p}")
        ax.set_xticks(bins[:-1])
        ax.set_xlabel("|K|  (essential characters)")
        ax.set_ylabel("# seeds")
        ax.set_title("Essential-K size distribution")
        ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "k_size_distribution.png")
    plt.close(fig)

    # ----- (3) helper-mult histogram (the ×2 monopoly) -----
    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    mult_by_prime: dict[int, Counter] = {}
    for p in primes:
        muls = [m for r in per_prime[p] if r.get("grokked") for m in r["helper_mults"]]
        mult_by_prime[p] = Counter(muls)
    all_mults = sorted({m for c in mult_by_prime.values() for m in c} | {2, 3})
    width = 0.8 / max(len(primes), 1)
    for i, p in enumerate(primes):
        ys = [mult_by_prime[p].get(m, 0) for m in all_mults]
        xs = [m + (i - (len(primes) - 1) / 2) * width for m in all_mults]
        ax.bar(xs, ys, width=width, label=f"p={p}")
    ax.set_xticks(all_mults)
    ax.set_xlabel("helper multiplier  (k_helper = mult · k_primary, mod p−1)")
    ax.set_ylabel("# helpers observed")
    ax.set_title("Helper-multiplier histogram (predicting: only mult=2)")
    ax.legend(fontsize=9)
    # Visual emphasis on the absence of mult=3
    if 3 in all_mults:
        ax.axvspan(2.5, 3.5, alpha=0.06, color="red", zorder=0)
    fig.tight_layout()
    fig.savefig(out_dir / "helper_multiplier_hist.png")
    plt.close(fig)

    print(f"Wrote 3 figures to {out_dir}/")


def _p_minus_1_str(p: int) -> str:
    factors = prime_factorize(p - 1)
    return "·".join(f"{q}^{e}" if e > 1 else str(q) for q, e in factors)


def _hist_counts(values, bins):
    import numpy as np
    counts, edges = np.histogram(values, bins=bins)
    return counts.tolist(), edges


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="experiments/lattice_variation",
                    help="Where to write the summary CSV + figures.")
    ap.add_argument("--no-plots", action="store_true",
                    help="Skip figure generation (CSV only).")
    args = ap.parse_args()

    per_prime: dict[int, list[dict]] = {}
    for p, pattern in PRIME_RUNS.items():
        dirs = sorted(glob.glob(pattern))
        print(f"[p={p}]  {len(dirs)} run dirs matching {pattern}")
        rows = []
        for d in dirs:
            r = analyze_run(Path(d))
            if r is not None:
                rows.append(r)
        per_prime[p] = rows

    summarize(per_prime)
    out_dir = Path(args.out_dir)
    write_csv(per_prime, out_dir / "lattice_variation.csv")
    if not args.no_plots:
        write_plots(per_prime, out_dir)


if __name__ == "__main__":
    main()
