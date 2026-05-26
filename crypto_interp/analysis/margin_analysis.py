"""Per-example margin analysis.

For each (a, b) pair, compute:
    target_logit       = L(a, b, c = a*b)
    max_alternative    = max_{c ≠ a*b} L(a, b, c)
    margin             = target - max_alternative
    predicted          = argmax_c L(a, b, c)
    correct            = (predicted == a*b)

We want to test whether grokked vs noisy seeds differ in the *per-example*
margin distribution, not in aggregate noise quantities. Hypothesis: noisy seeds
have small or negative margins on a ~2-5% subset of (a, b) pairs where the
noise wins; grokked seeds have positive margins everywhere with comfortable
spread.

Plus: decompose the margin as

    margin(a, b) = kernel_margin + noise_margin(a, b)

where ``kernel_margin = κ(0) - max_{d≠0} κ(d)`` is constant across (a, b)
and ``noise_margin`` is the per-example noise contribution. This shows
exactly how the noise term consumes the kernel's margin headroom.

Usage:
    python -m crypto_interp.analysis.margin_analysis \\
        --seeds 7 8 10 12 9 13 14 \\
        --runs-root experiments/003_dmodel_sweep_p113/runs \\
        --tag dmodel_24_dmlp_20_wd2 \\
        --out-dir experiments/margin_analysis
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from crypto_interp.interp import Session
from crypto_interp.interp.bases import discrete_log_table
from crypto_interp.interp.theory import observed_kernel


def margin_breakdown(run_dir: Path) -> dict:
    S = Session.from_run(str(run_dir))
    p = S.ds.p
    logits = S.logits_grid.detach().cpu().numpy()[..., :p].astype(np.float64)
    # logits[a-1, b-1, c] for a, b ∈ {1, …, p−1}, c ∈ {0, …, p−1}
    # Targets:
    a_idx = np.arange(1, p)[:, None]
    b_idx = np.arange(1, p)[None, :]
    target_c = (a_idx * b_idx) % p   # shape (p−1, p−1)
    n = p - 1

    target_logit = np.take_along_axis(logits, target_c[..., None], axis=-1).squeeze(-1)
    # Mask out target before taking the max over c.
    logits_mask = logits.copy()
    np.put_along_axis(logits_mask, target_c[..., None], -np.inf, axis=-1)
    max_alt_logit = logits_mask.max(axis=-1)
    margin = target_logit - max_alt_logit
    predicted_c = logits.argmax(axis=-1)
    correct_mask = (predicted_c == target_c)
    accuracy = float(correct_mask.mean())
    n_wrong = int((~correct_mask).sum())

    # Kernel margin for reference: compute κ_obs and its margin.
    from crypto_interp.analysis.noise_decomposition import reindex_logits_by_dlog
    L_dlog = reindex_logits_by_dlog(S.logits_grid.detach().cpu().numpy(), p)
    # Re-derive the offset-indexed kernel from L_dlog by averaging at fixed offset.
    d = (np.arange(n)[:, None, None] + np.arange(n)[None, :, None]
         - np.arange(n)[None, None, :]) % n
    kappa = np.zeros(n)
    counts = np.zeros(n)
    np.add.at(kappa, d.ravel(), L_dlog.ravel())
    np.add.at(counts, d.ravel(), 1)
    kappa = kappa / np.maximum(counts, 1)
    kernel_margin = float(kappa[0] - kappa[1:].max())

    # If model errors are present, summarize where they live.
    wrong_pairs = []
    if n_wrong > 0:
        ai, bi = np.where(~correct_mask)
        for i, j in zip(ai[:50], bi[:50]):
            wrong_pairs.append({
                "a": int(i + 1), "b": int(j + 1),
                "target": int(target_c[i, j]),
                "predicted": int(predicted_c[i, j]),
                "target_logit": float(target_logit[i, j]),
                "max_alt_logit": float(max_alt_logit[i, j]),
                "margin": float(margin[i, j]),
            })

    return {
        "run_dir": str(run_dir),
        "p": p,
        "accuracy": accuracy,
        "n_wrong": n_wrong,
        "n_total": int(correct_mask.size),
        "margin": margin,
        "kernel_margin": kernel_margin,
        "target_logit": target_logit,
        "max_alt_logit": max_alt_logit,
        "wrong_pairs": wrong_pairs,
        "correct_mask": correct_mask,
    }


def print_one(d: dict) -> None:
    name = Path(d["run_dir"]).name
    print(f"\n=== {name} (p={d['p']}) ===")
    print(f"  accuracy:        {d['accuracy']:.4%}")
    print(f"  wrong examples:  {d['n_wrong']} / {d['n_total']}")
    print(f"  kernel margin (constant across (a,b)): {d['kernel_margin']:.4g}")
    m = d["margin"]
    print(f"  margin distribution:")
    print(f"    min:        {m.min():.4g}")
    print(f"    1st pct:    {np.percentile(m, 1):.4g}")
    print(f"    median:     {np.median(m):.4g}")
    print(f"    99th pct:   {np.percentile(m, 99):.4g}")
    print(f"    max:        {m.max():.4g}")
    if d["wrong_pairs"]:
        print(f"  first 10 incorrect (a, b) → predicted vs target  (margin):")
        for w in d["wrong_pairs"][:10]:
            print(f"    a={w['a']:>3}, b={w['b']:>3}  pred={w['predicted']:>3}  "
                  f"target={w['target']:>3}  margin={w['margin']:+.3g}")


def plot_one(d: dict, ax_margin, ax_logits) -> None:
    """Two-panel per-seed plot: margin distribution + scatter of (target, max_alt)."""
    m = d["margin"].ravel()
    # Histogram of margin
    ax_margin.hist(m, bins=80, color="steelblue", edgecolor="black", linewidth=0.4)
    ax_margin.axvline(0, color="red", linestyle="--", label="0 (error boundary)")
    ax_margin.axvline(d["kernel_margin"], color="green", linestyle=":",
                      label=f"kernel margin = {d['kernel_margin']:.2g}")
    ax_margin.set_xlabel("margin = target − max_alt")
    ax_margin.set_ylabel("# of (a, b) pairs")
    ax_margin.set_title(f"{Path(d['run_dir']).name}\nacc={d['accuracy']:.4%}  "
                        f"n_wrong={d['n_wrong']}")
    ax_margin.legend(fontsize=9)
    ax_margin.grid(True, alpha=0.3)

    # Scatter target_logit vs max_alt_logit
    correct = d["correct_mask"].ravel()
    tl = d["target_logit"].ravel()
    ma = d["max_alt_logit"].ravel()
    if (~correct).any():
        ax_logits.scatter(tl[~correct], ma[~correct], s=4, color="red",
                          alpha=0.7, label=f"wrong (n={d['n_wrong']})")
    ax_logits.scatter(tl[correct], ma[correct], s=2, color="C0",
                      alpha=0.3, label="correct")
    lo, hi = min(tl.min(), ma.min()), max(tl.max(), ma.max())
    ax_logits.plot([lo, hi], [lo, hi], "k--", alpha=0.3, label="margin=0")
    ax_logits.set_xlabel("target logit  L(a, b, ab)")
    ax_logits.set_ylabel("max-alternative logit  max_{c≠ab} L")
    ax_logits.set_title("target vs max_alt logits")
    ax_logits.legend(fontsize=9)
    ax_logits.grid(True, alpha=0.3)


def plot_population(results: list[dict], out_dir: Path) -> None:
    n_seeds = len(results)
    fig, axes = plt.subplots(n_seeds, 2, figsize=(13, 3.2 * n_seeds), squeeze=False)
    for i, d in enumerate(results):
        plot_one(d, axes[i][0], axes[i][1])
    fig.tight_layout()
    out = out_dir / "population_margins.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"\nWrote {out}")

    # Also: aggregate scatter — accuracy vs margin spread
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    # (a) all margin distributions overlaid (log y)
    ax = axes[0]
    for d in results:
        seed = int(Path(d["run_dir"]).name.split("seed")[-1])
        if seed in {7, 8, 10, 12}: color, regime = "green", "grokked"
        elif seed in {13, 14}:     color, regime = "darkorange", "noisy"
        elif seed == 9:            color, regime = "red", "CRT-fail"
        else:                      color, regime = "gray", "?"
        m = d["margin"].ravel()
        ax.hist(m, bins=60, alpha=0.4, color=color, label=f"seed {seed} ({regime})",
                density=True)
    ax.axvline(0, color="black", linestyle="--", alpha=0.5)
    ax.set_xlabel("margin = target − max_alt")
    ax.set_ylabel("density")
    ax.set_title("Per-example margin distributions")
    ax.set_yscale("log")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)

    # (b) negative-margin tail: zoom in
    ax = axes[1]
    for d in results:
        seed = int(Path(d["run_dir"]).name.split("seed")[-1])
        if seed in {7, 8, 10, 12}: color = "green"
        elif seed in {13, 14}:     color = "darkorange"
        elif seed == 9:            color = "red"
        else:                      color = "gray"
        m = d["margin"].ravel()
        # Plot sorted ascending — the worst tail
        sorted_m = np.sort(m)
        ax.plot(sorted_m[:500], color=color, alpha=0.7, label=f"seed {seed}")
    ax.axhline(0, color="black", linestyle="--", alpha=0.5)
    ax.set_xlabel("rank (worst 500 examples)")
    ax.set_ylabel("margin")
    ax.set_title("Bottom 500 margins per seed")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (c) summary: accuracy vs L_emp
    ax = axes[2]
    for d in results:
        seed = int(Path(d["run_dir"]).name.split("seed")[-1])
        if seed in {7, 8, 10, 12}: color, regime = "green", "grokked"
        elif seed in {13, 14}:     color, regime = "darkorange", "noisy"
        elif seed == 9:            color, regime = "red", "CRT-fail"
        else:                      color, regime = "gray", "?"
        ax.scatter([d["accuracy"]], [d["n_wrong"]], c=color, s=120,
                   edgecolor="black", linewidth=0.6)
        ax.annotate(f" s{seed}", (d["accuracy"], d["n_wrong"]), fontsize=10)
    ax.set_xlabel("argmax accuracy")
    ax.set_ylabel("# wrong examples  (out of (p−1)² = 12544)")
    ax.set_yscale("symlog")
    ax.set_title("Accuracy & error count")
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    out = out_dir / "population_summary.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"Wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, required=True)
    ap.add_argument("--runs-root", default="experiments/003_dmodel_sweep_p113/runs")
    ap.add_argument("--tag", default="dmodel_24_dmlp_20_wd2")
    ap.add_argument("--out-dir", default="experiments/margin_analysis")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for s in args.seeds:
        r = Path(args.runs_root) / f"{args.tag}_seed{s}"
        if not r.exists():
            print(f"skip missing {r}")
            continue
        d = margin_breakdown(r)
        results.append(d)
        print_one(d)

    if results:
        plot_population(results, out_dir)


if __name__ == "__main__":
    main()
