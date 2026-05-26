"""Decompose the asymmetric noise term into mechanistically meaningful modes.

The model's logits decompose as

    L_model(x_a, x_b, x_c) = κ_obs(x_a + x_b - x_c) + ε(x_a, x_b, x_c)

where ``κ_obs`` is the translation-symmetric kernel (the algorithm) and ``ε`` is
the asymmetric residual. By construction, ε has zero mean conditional on the
offset d = x_a + x_b - x_c.

The 3D Fourier decomposition of ε along (x_a, x_b, x_c) lives on
Z/(p-1)^3. Modes are indexed by (k_a, k_b, k_c). Translation-invariant modes
(those satisfying k_a + k_b - k_c ≡ 0) carry the kernel; by construction the
residual has these modes at zero.

Everything else — the asymmetric modes — is grouped into mechanistically
distinct subsets:

    "single_a"       : modes with k_b = k_c = 0, k_a ≠ 0
                       → output depends on a alone (no use of b)
    "single_b"       : modes with k_a = k_c = 0, k_b ≠ 0
                       → output depends on b alone (no use of a)
    "single_c"       : modes with k_a = k_b = 0, k_c ≠ 0
                       → class-prior bias
    "pair_ac"        : k_b = 0, k_a, k_c nonzero, NOT in the algorithmic family
                       → half-attention / a-c shortcut
    "pair_bc"        : k_a = 0, k_b, k_c nonzero, similarly
                       → half-attention / b-c shortcut
    "pair_ab"        : k_c = 0, k_a, k_b nonzero
                       → input-only interaction, never reaches output
    "off_resonance"  : k_a = k_b ≠ 0 but k_c ≠ k_a + k_b mod (p-1)
                       → MLP cluster computes the right input product but
                         reads out at the wrong output character
    "generic"        : everything else

The integrals report ``Σ |Ê(k_a, k_b, k_c)|² / N`` in each subset — the L²
contribution of that subset to the residual.

Usage:
    python -m crypto_interp.analysis.noise_decomposition \\
        --run-dir experiments/003_dmodel_sweep_p113/runs/dmodel_24_dmlp_20_wd2_seed14
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from crypto_interp.interp import Session
from crypto_interp.interp.bases import discrete_log_table


def reindex_logits_by_dlog(logits_grid: np.ndarray, p: int) -> np.ndarray:
    """Convert ``logits_grid[a-1, b-1, c]`` over a, b ∈ (Z/p)*, c ∈ Z/p into
    ``L[x_a, x_b, x_c]`` indexed by dlogs — shape (p-1, p-1, p-1).
    """
    n = p - 1
    _g, dlog = discrete_log_table(p)
    L = np.asarray(logits_grid)[..., 1:p].astype(np.float64)  # drop c=0
    L_dlog = np.zeros((n, n, n), dtype=np.float64)
    # invert dlog: residue at dlog x is (g^x mod p) — for the index mapping.
    # We need: L_dlog[x_a, x_b, x_c] = L[a-1, b-1, c-1] where dlog(a)=x_a etc.
    inv_dlog = np.zeros(n, dtype=np.int64)  # inv_dlog[x] = (residue with dlog x) - 1
    for residue, x in dlog.items():
        inv_dlog[x] = residue - 1
    for xa in range(n):
        for xb in range(n):
            L_dlog[xa, xb, :] = L[inv_dlog[xa], inv_dlog[xb], inv_dlog]
    return L_dlog


def compute_residual(L_dlog: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Subtract the symmetric kernel κ_obs(d) from L_dlog.

    Returns (residual ε, kernel κ_obs).
    """
    n = L_dlog.shape[0]
    # Offset tensor (n, n, n)
    d = (np.arange(n)[:, None, None] + np.arange(n)[None, :, None]
         - np.arange(n)[None, None, :]) % n
    # κ_obs(d) = mean of L_dlog over (x_a, x_b, x_c) at fixed d.
    kappa = np.zeros(n, dtype=np.float64)
    counts = np.zeros(n, dtype=np.int64)
    np.add.at(kappa, d.ravel(), L_dlog.ravel())
    np.add.at(counts, d.ravel(), 1)
    kappa = kappa / np.maximum(counts, 1)
    eps = L_dlog - kappa[d]
    return eps, kappa


def classify_modes(n: int) -> dict[str, np.ndarray]:
    """Boolean masks of shape (n, n, n) selecting each mode class.

    The "algorithm" subspace — modes carrying the translation-symmetric
    kernel κ(x_a + x_b - x_c) — has Fourier support at (k, k, n-k) for
    k = 0..n-1 (and equivalently their conjugates (n-k, n-k, k)). These
    are exactly the modes where ``k_a == k_b`` AND ``k_a + k_c ≡ 0 (mod n)``.
    By construction the residual ε = L - κ_obs has zero mass here.

    Off-resonance: input characters match (k_a = k_b) but the output
    character k_c is wrong (k_c ≠ -k_a mod n).
    """
    ka = np.arange(n)[:, None, None]
    kb = np.arange(n)[None, :, None]
    kc = np.arange(n)[None, None, :]

    # k_a + k_c ≡ 0 mod n  ⇔  k_c == (n - k_a) mod n.
    algorithm = (ka == kb) & ((ka + kc) % n == 0)
    off_resonance = (ka == kb) & ~algorithm  # includes both k_a=0 and k_a≠0

    single_a = (kb == 0) & (kc == 0) & (ka != 0)
    single_b = (ka == 0) & (kc == 0) & (kb != 0)
    single_c = (ka == 0) & (kb == 0) & (kc != 0)

    pair_ac = (kb == 0) & (ka != 0) & (kc != 0)
    pair_bc = (ka == 0) & (kb != 0) & (kc != 0)
    pair_ab = (kc == 0) & (ka != 0) & (kb != 0) & (ka != kb)

    classified = (single_a | single_b | single_c
                  | pair_ac | pair_bc | pair_ab
                  | off_resonance | algorithm)
    generic = ~classified

    return {
        "algorithm":       algorithm,
        "single_a":        single_a,
        "single_b":        single_b,
        "single_c":        single_c,
        "pair_ac":         pair_ac,
        "pair_bc":         pair_bc,
        "pair_ab":         pair_ab,
        "off_resonance":   off_resonance,
        "generic":         generic,
    }


def noise_at_answer_variance(eps: np.ndarray, p: int) -> tuple[float, float]:
    """Variance of ε(a, b, ab) over (a, b) pairs — the noise *at the target*.

    This is the noise that directly enters the CE through the target-class
    logit. The kernel value at d=0 is the same for every (a, b), so
    ε(a, b, ab) = L_model(a, b, ab) - κ(0). Its variance should correlate
    tightly with the L_emp - L_sym gap.

    Returns (variance, mean). Mean should be ~0 by construction.
    """
    n = eps.shape[0]
    # All (x_a, x_b) pairs; the target's dlog is x_a + x_b mod n.
    xa = np.arange(n)[:, None]
    xb = np.arange(n)[None, :]
    xc_target = (xa + xb) % n   # dlog of c = a*b
    noise_at_target = eps[xa, xb, xc_target]  # shape (n, n)
    return float(noise_at_target.var()), float(noise_at_target.mean())


def top_modes(power: np.ndarray, masks: dict[str, np.ndarray],
              n_top: int = 12) -> list[dict]:
    """Top-N individual modes by power, each tagged with its class."""
    n = power.shape[0]
    flat = power.ravel()
    order = np.argsort(flat)[::-1][:n_top]
    out = []
    for idx in order:
        ka, kb, kc = np.unravel_index(idx, power.shape)
        cls = "?"
        for name, m in masks.items():
            if m[ka, kb, kc]:
                cls = name
                break
        out.append({
            "k_a": int(ka), "k_b": int(kb), "k_c": int(kc),
            "power": float(power[ka, kb, kc]),
            "fold_kc_neg": int((n - ka) % n),   # what kc would have been if algorithmic
            "class": cls,
        })
    return out


def decompose(run_dir: Path) -> dict:
    S = Session.from_run(str(run_dir))
    p = S.ds.p
    logits = S.logits_grid.detach().cpu().numpy()
    L_dlog = reindex_logits_by_dlog(logits, p)
    eps, kappa = compute_residual(L_dlog)
    n = p - 1

    # 3D FFT of the residual (normalized so total Σ |E|² = (1/n) Σ |eps|²).
    E = np.fft.fftn(eps) / n
    power = np.abs(E) ** 2
    total_power = float(power.sum())

    masks = classify_modes(n)
    by_class = {name: float(power[m].sum()) for name, m in masks.items()}

    # The metric that should correlate with the L_emp - L_sym gap:
    target_var, target_mean = noise_at_answer_variance(eps, p)

    # Top modes for mechanistic insight.
    tops = top_modes(power, masks, n_top=12)

    # Kernel power (1D) for SNR.
    kernel_power = float(np.sum(kappa ** 2))  # Σ_d |κ(d)|²
    kernel_in_3d = n * n * kernel_power      # spread across n² values of d⁻¹ orbit

    from crypto_interp.interp.theory import (
        empirical_test_loss_from_logits, kernel_loss,
    )
    L_emp = empirical_test_loss_from_logits(logits, p)
    L_sym = kernel_loss(kappa)

    return {
        "run_dir": str(run_dir),
        "p": p,
        "n": n,
        "L_empirical": L_emp,
        "L_symmetric": L_sym,
        "total_residual_power": total_power,
        "by_class": by_class,
        "masks": masks,
        "power": power,
        "noise_at_target_var": target_var,
        "noise_at_target_mean": target_mean,
        "kernel_3d_power": kernel_in_3d,
        "snr_3d": kernel_in_3d / max(total_power, 1e-300),
        "top_modes": tops,
    }


def print_summary(d: dict) -> None:
    print(f"\nResidual decomposition for {Path(d['run_dir']).name}  (p={d['p']})")
    print(f"  L_empirical = {d['L_empirical']:.4g}")
    print(f"  L_symmetric = {d['L_symmetric']:.4g}")
    print(f"  total residual power = {d['total_residual_power']:.4g}")
    print("\n  per-class L² contributions (fraction of total residual power):")
    total = d["total_residual_power"]
    classes = ["algorithm", "single_a", "single_b", "single_c",
               "pair_ac", "pair_bc", "pair_ab",
               "off_resonance", "generic"]
    for c in classes:
        val = d["by_class"][c]
        frac = val / total if total > 0 else 0.0
        bar = "█" * int(50 * frac)
        print(f"    {c:>15s}  {val:>12.4g}  {frac*100:>6.2f}%  {bar}")


def plot_summary(d: dict, out_path: Path) -> None:
    classes = ["single_a", "single_b", "single_c",
               "pair_ac", "pair_bc", "pair_ab",
               "off_resonance", "generic"]
    vals = [d["by_class"][c] for c in classes]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = ["C0", "C0", "C1", "C2", "C2", "C3", "C4", "gray"]
    ax.bar(classes, vals, color=colors)
    ax.set_ylabel("L² power in residual")
    ax.set_yscale("log")
    ax.set_title(f"Asymmetric-noise decomposition — {Path(d['run_dir']).name}\n"
                 f"L_emp={d['L_empirical']:.3g}, L_sym={d['L_symmetric']:.3g}")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    ax.grid(True, axis="y", alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_population(results: list[dict], out_path: Path) -> None:
    """Three-panel population summary:
        (a) stacked bar of noise composition per seed
        (b) scatter: Var[ε(a,b,ab)] vs (L_emp − L_sym)
        (c) scatter: total noise power vs L_emp
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))

    classes_order = ["off_resonance", "single_c", "generic",
                     "pair_ac", "pair_bc", "single_a", "single_b", "pair_ab"]
    color_map = {
        "off_resonance": "#1f77b4",
        "single_c":      "#ff7f0e",
        "generic":       "#bcbd22",
        "pair_ac":       "#2ca02c",
        "pair_bc":       "#98df8a",
        "single_a":      "#d62728",
        "single_b":      "#ff9896",
        "pair_ab":       "#9467bd",
    }
    seed_labels = [Path(r["run_dir"]).name.split("seed")[-1] for r in results]

    # (a) stacked bar — fractional composition
    ax = axes[0]
    bottoms = np.zeros(len(results))
    for cls in classes_order:
        vals = np.array([r["by_class"][cls] / r["total_residual_power"] for r in results])
        ax.bar(seed_labels, vals * 100, bottom=bottoms * 100,
               label=cls, color=color_map[cls])
        bottoms += vals
    ax.set_ylabel("% of total residual power")
    ax.set_xlabel("seed")
    ax.set_title("Noise composition by Fourier subspace")
    ax.legend(fontsize=8, loc="upper right", bbox_to_anchor=(1.02, 1.0))
    ax.set_ylim(0, 105)

    # (b) Var[ε at target] vs (L_emp − L_sym)
    ax = axes[1]
    xs = np.array([r["noise_at_target_var"] for r in results])
    ys = np.array([r["L_empirical"] - r["L_symmetric"] for r in results])
    # Mark grokked vs noisy vs failed by color
    colors = []
    for r in results:
        seed = int(Path(r["run_dir"]).name.split("seed")[-1])
        if seed in {7, 8, 10, 12}: colors.append("green")
        elif seed in {13, 14}:     colors.append("orange")
        elif seed == 9:            colors.append("red")
        else:                      colors.append("gray")
    ax.loglog(xs, ys, "o", markersize=0)
    for x, y, c, s in zip(xs, ys, colors, seed_labels):
        ax.scatter([x], [y], c=c, s=80, edgecolor="black", linewidth=0.6)
        ax.annotate(f" {s}", (x, y), fontsize=9)
    # Reference line: y = x/2 (Gaussian prediction)
    if len(xs) > 0:
        xx = np.geomspace(max(xs.min(), 1e-12), xs.max(), 100)
        ax.plot(xx, xx / 2, "--", color="gray", alpha=0.5, label="y = x / 2 (Gaussian)")
    ax.set_xlabel("Var[ε(a, b, ab)]  (noise at target)")
    ax.set_ylabel("L_empirical − L_symmetric")
    ax.set_title("Target noise predicts the CE gap")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=9)

    # (c) total residual power vs L_empirical
    ax = axes[2]
    xs = np.array([r["total_residual_power"] for r in results])
    ys = np.array([r["L_empirical"] for r in results])
    for x, y, c, s in zip(xs, ys, colors, seed_labels):
        ax.scatter([x], [y], c=c, s=80, edgecolor="black", linewidth=0.6)
        ax.annotate(f" {s}", (x, y), fontsize=9)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("total residual power Σ|ε|²")
    ax.set_ylabel("L_empirical")
    ax.set_title("Total noise power doesn't predict grokking")
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"\nWrote population summary {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=None,
                    help="Single run (alternative to --seeds).")
    ap.add_argument("--runs-root", default="experiments/003_dmodel_sweep_p113/runs")
    ap.add_argument("--tag", default="dmodel_24_dmlp_20_wd2")
    ap.add_argument("--seeds", nargs="+", type=int, default=None)
    ap.add_argument("--out-dir", default="experiments/noise_decomposition")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.run_dir:
        runs = [Path(args.run_dir)]
    elif args.seeds:
        runs = [Path(args.runs_root) / f"{args.tag}_seed{s}" for s in args.seeds]
    else:
        raise SystemExit("provide --run-dir or --seeds")

    results = []
    for r in runs:
        if not r.exists():
            print(f"skip missing {r}")
            continue
        d = decompose(r)
        results.append(d)
        print_summary(d)
        # Top modes
        print("\n  Top 8 individual modes by power:")
        print(f"    {'k_a':>4} {'k_b':>4} {'k_c':>4}  {'power':>11}  {'class':>15}  notes")
        for tm in d["top_modes"][:8]:
            algkc = tm["fold_kc_neg"]
            note = f"alg-kc={algkc}" if tm["k_a"] == tm["k_b"] else ""
            print(f"    {tm['k_a']:>4} {tm['k_b']:>4} {tm['k_c']:>4}  {tm['power']:>11.4g}  {tm['class']:>15}  {note}")
        plot_summary(d, out_dir / f"{Path(r).name}.png")
        print(f"\n  noise_at_target_var = {d['noise_at_target_var']:.4g}  "
              f"L_emp - L_sym = {d['L_empirical'] - d['L_symmetric']:.4g}")
        print(f"  prediction L_emp - L_sym ≈ var/2 = {d['noise_at_target_var']/2:.4g}")
        print(f"  total power = {d['total_residual_power']:.4g}   "
              f"SNR = {d['snr_3d']:.4g}")

    if len(results) > 1:
        plot_population(results, out_dir / "population.png")


if __name__ == "__main__":
    main()
