"""Cross-prime kernel/noise comparison.

For every run across the configured primes, compute the empirical CE
(L_emp), the symmetric-kernel CE (L_sym; see `interp.theory`), the kernel
margin, and the algebraic floor on L_sym implied by the run's essential
character set K (zero iff the lcm of character orders covers p-1).
Each run is then bucketed by thresholds on (L_emp, L_sym) and the results
are written as a CSV plus two scatter plots: L_emp vs L_sym across primes,
and predicted floor vs observed L_sym for non-grokked runs.

Usage:
    python -m crypto_interp.analysis.cross_prime_kernel_noise \\
        --out-dir outputs/cross_prime_kernel_noise/all
"""
from __future__ import annotations

import argparse
import csv
import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from crypto_interp.interp import Session
from crypto_interp.interp.theory import theory_baseline


PRIME_RUNS = {
    113: "experiments/003_dmodel_sweep_p113/runs/dmodel_24_dmlp_20_wd2_seed*",
    127: "experiments/004_p127/runs/dmodel_24_dmlp_32_seed[0-9]*",
    181: "experiments/005_p181/runs/dmodel_24_dmlp_32_seed[0-9]*",
}


def prime_factorize(n: int) -> list[tuple[int, int]]:
    out, d = [], 2
    while d * d <= n:
        e = 0
        while n % d == 0:
            n //= d; e += 1
        if e:
            out.append((d, e))
        d += 1
    if n > 1:
        out.append((n, 1))
    return out


def order_of(k: int, p: int) -> int:
    return (p - 1) // np.gcd(k, p - 1) if k != 0 else 1


def crt_minimality_floor(K: list[int], p: int) -> float:
    """The algebraic floor on L_sym for a given K.

    If K satisfies CRT minimality (lcm of orders == p−1), the floor is 0:
    a perfect approximate-CRT readout with K alone can in principle reach
    zero loss.  Otherwise the floor is bounded below by log of the size
    of the largest fiber of the joint character map a ↦ (χ_k(a))_{k∈K}.
    Specifically, if the kernel intersection has size m > 1, then m
    elements collapse to the same character vector, and any classifier
    achieves at most 1/m on those — so CE per such (a,b) is at least
    log m.  Averaged over (a,b), this gives log m as the floor.
    """
    n = p - 1
    orders = [order_of(int(k), p) for k in K]
    lcm = 1
    for o in orders:
        lcm = (lcm * o) // np.gcd(lcm, o)
    # Size of the kernel intersection in (Z/p)*.
    fiber_size = n // lcm
    if fiber_size <= 1:
        return 0.0
    return float(np.log(fiber_size))


def classify_regime(L_emp: float, L_sym: float, kernel_margin: float) -> str:
    """Assign one of three labels."""
    grokked_threshold = 1e-2
    if L_sym >= 0.5 and L_emp >= 0.5 and L_emp - L_sym < 0.3:
        return "CRT-fail"
    if L_emp < grokked_threshold:
        return "grokked"
    if L_sym < grokked_threshold:
        return "algorithmic-but-noisy"
    return "intermediate"


def kernel_margin(kappa_obs: np.ndarray) -> float:
    return float(kappa_obs[0] - kappa_obs[1:].max())


def analyze_one(run_dir: Path) -> dict | None:
    try:
        S = Session.from_run(str(run_dir))
        train_l, test_l, acc = S.evaluate()
    except Exception as e:
        return {"run_dir": str(run_dir), "error": str(e)}
    try:
        r = theory_baseline(S)
    except Exception as e:
        return {"run_dir": str(run_dir), "error": f"theory_baseline: {e}"}

    p = r["p"]
    K = r["K"]
    floor = crt_minimality_floor(K, p)
    margin = kernel_margin(r["kappa_obs"])
    regime = classify_regime(r["L_empirical"], r["L_symmetric"], margin)
    return {
        "run_dir": str(run_dir),
        "p": p,
        "test_loss_eval": float(test_l),
        "accuracy_eval": float(acc),
        "K": K,
        "|K|": len(K),
        "L_emp": r["L_empirical"],
        "L_sym": r["L_symmetric"],
        "kernel_margin": margin,
        "predicted_floor": floor,
        "regime": regime,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="outputs/cross_prime_kernel_noise/all")
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for p, pattern in PRIME_RUNS.items():
        dirs = sorted(d for d in glob.glob(pattern)
                      if Path(d).is_dir() and not d.endswith(".log"))
        print(f"[p={p}]  {len(dirs)} dirs")
        for d in dirs:
            r = analyze_one(Path(d))
            if r is None or "error" in r:
                if r and "error" in r:
                    print(f"  skip {Path(d).name}: {r['error']}")
                continue
            all_rows.append(r)

    # Print per-prime summary
    print("\n" + "=" * 78)
    print("CROSS-PRIME KERNEL/NOISE SUMMARY")
    print("=" * 78)
    for p in sorted({r["p"] for r in all_rows}):
        rows = [r for r in all_rows if r["p"] == p]
        regime_ct = {}
        for r in rows:
            regime_ct[r["regime"]] = regime_ct.get(r["regime"], 0) + 1
        print(f"\np = {p}  ({len(rows)} seeds)")
        for reg in ["grokked", "algorithmic-but-noisy", "CRT-fail", "intermediate"]:
            n = regime_ct.get(reg, 0)
            if n:
                print(f"  {reg:>24s}: {n}")
        # Algebraic-floor check
        nong = [r for r in rows if r["regime"] != "grokked"]
        if nong:
            print(f"  non-grokked seeds: predicted vs observed L_sym floor:")
            for r in sorted(nong, key=lambda r: -r["predicted_floor"]):
                print(f"    {Path(r['run_dir']).name:42s}  "
                      f"K={r['K']}  pred={r['predicted_floor']:.4g}  "
                      f"L_sym={r['L_sym']:.4g}  L_emp={r['L_emp']:.4g}  "
                      f"[{r['regime']}]")

    # CSV
    csv_path = out_dir / "cross_prime_kernel_noise.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["p", "run_dir", "regime", "test_loss_eval", "accuracy_eval",
                    "|K|", "K", "L_emp", "L_sym", "kernel_margin",
                    "predicted_floor"])
        for r in all_rows:
            w.writerow([r["p"], r["run_dir"], r["regime"], r["test_loss_eval"],
                        r["accuracy_eval"], r["|K|"], " ".join(map(str, r["K"])),
                        r["L_emp"], r["L_sym"], r["kernel_margin"],
                        r["predicted_floor"]])
    print(f"\nWrote {csv_path}")

    # Plots
    plot_population(all_rows, out_dir)


def plot_population(rows: list[dict], out_dir: Path) -> None:
    # 1. Scatter of L_emp vs L_sym, colored by prime
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

    ax = axes[0]
    color_by_p = {113: "C0", 127: "C1", 181: "C2"}
    marker_by_regime = {"grokked": "o", "algorithmic-but-noisy": "s",
                        "CRT-fail": "x", "intermediate": "^"}
    for p in [113, 127, 181]:
        for reg in ["grokked", "algorithmic-but-noisy", "CRT-fail", "intermediate"]:
            sel = [r for r in rows if r["p"] == p and r["regime"] == reg]
            if not sel:
                continue
            xs = np.clip([r["L_sym"] for r in sel], 1e-10, None)
            ys = np.clip([r["L_emp"] for r in sel], 1e-10, None)
            ax.scatter(xs, ys, c=color_by_p[p], marker=marker_by_regime[reg],
                       s=60, alpha=0.75, edgecolor="black", linewidth=0.4,
                       label=f"p={p} {reg}")
    # y = x reference
    lim = (1e-10, 10)
    ax.plot(lim, lim, "k--", alpha=0.3, label="y = x")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(*lim); ax.set_ylim(*lim)
    ax.set_xlabel("L_symmetric (algorithm)")
    ax.set_ylabel("L_empirical (model behavior)")
    ax.set_title("Cross-prime regime scatter")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=7, loc="lower right", ncol=2)

    # 2. Predicted floor vs observed L_sym at non-grokked seeds
    ax = axes[1]
    for p in [113, 127, 181]:
        sel = [r for r in rows if r["p"] == p and r["regime"] != "grokked"
               and r["predicted_floor"] > 0]
        if not sel:
            continue
        xs = [r["predicted_floor"] for r in sel]
        ys = [r["L_sym"] for r in sel]
        ax.scatter(xs, ys, c=color_by_p[p], s=80, alpha=0.85,
                   edgecolor="black", linewidth=0.4, label=f"p={p}")
        # Annotate with seed numbers
        for r in sel:
            seed = Path(r["run_dir"]).name.split("seed")[-1]
            ax.annotate(f" s{seed}", (r["predicted_floor"], r["L_sym"]), fontsize=8)
    # y = x reference
    ax.plot([0, 5], [0, 5], "k--", alpha=0.3, label="y = x  (prediction)")
    ax.set_xlabel("Predicted algebraic floor on L_sym  (log p_1 in kernel intersection)")
    ax.set_ylabel("Observed L_sym")
    ax.set_title("CRT-failure: does the algebra predict the floor?")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    out_path = out_dir / "cross_prime_regimes.png"
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
