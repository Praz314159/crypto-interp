"""Where errors live: per-example margins + per-token attribution.

For each (a, b) pair, compute:
    target_logit       = L(a, b, c = a*b)
    max_alt_logit      = max_{c ≠ a*b} L(a, b, c)
    margin             = target − max_alt   (positive ⇒ correct)
    predicted          = argmax_c L(a, b, c)
    correct            = (predicted == a*b)

Aggregate two ways:
    1. Per-example margin distribution — bulk of (a, b) pairs lives where?
       Algorithm-but-noisy seeds have median margin similar to grokked seeds
       but a thin tail of catastrophically negative margins.
    2. Per-token error attribution — which input tokens carry the errors?
       Errors typically concentrate on 1–4 specific token columns of W_E,
       symmetric in A vs B position (implicating the embedding directly).

Also reports the kernel margin κ(0) − max_{d≠0} κ(d) per seed: the
algorithm's own headroom, constant across (a, b). CRT-failure seeds are
diagnosed by kernel margin ≈ 0.

Usage:
    python -m crypto_interp.analysis.error_localization \\
        --seeds 7 8 10 12 9 13 14 \\
        --runs-root experiments/003_dmodel_sweep_p113/runs \\
        --tag dmodel_24_dmlp_20_wd2 \\
        --out-dir experiments/error_localization
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from crypto_interp.interp import Session


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def analyze_run(run_dir: Path) -> dict:
    """Single shared pass through the logits grid; both margin and per-token views."""
    S = Session.from_run(str(run_dir))
    p = S.ds.p
    n = p - 1
    logits = S.logits_grid.detach().cpu().numpy()[..., :p].astype(np.float64)

    # Targets c = a*b mod p, for a, b ∈ {1, …, p-1}.
    a_idx = np.arange(1, p)[:, None]
    b_idx = np.arange(1, p)[None, :]
    target = (a_idx * b_idx) % p

    target_logit = np.take_along_axis(logits, target[..., None], axis=-1).squeeze(-1)
    logits_mask = logits.copy()
    np.put_along_axis(logits_mask, target[..., None], -np.inf, axis=-1)
    max_alt = logits_mask.max(axis=-1)
    margin = target_logit - max_alt
    predicted = logits.argmax(axis=-1)
    wrong = predicted != target

    # Per-token error rates.
    err_by_a = wrong.mean(axis=1)
    err_by_b = wrong.mean(axis=0)
    # Per-target-class error rate.
    err_by_target = np.zeros(p)
    target_count = np.zeros(p, dtype=np.int64)
    np.add.at(err_by_target, target.ravel(), wrong.ravel().astype(np.int64))
    np.add.at(target_count, target.ravel(), 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        err_by_target = err_by_target / np.maximum(target_count, 1)

    # Kernel margin for reference.
    from crypto_interp.analysis.noise_decomposition import reindex_logits_by_dlog
    L_dlog = reindex_logits_by_dlog(S.logits_grid.detach().cpu().numpy(), p)
    d = (np.arange(n)[:, None, None] + np.arange(n)[None, :, None]
         - np.arange(n)[None, None, :]) % n
    kappa = np.zeros(n)
    counts = np.zeros(n)
    np.add.at(kappa, d.ravel(), L_dlog.ravel())
    np.add.at(counts, d.ravel(), 1)
    kappa = kappa / np.maximum(counts, 1)
    kernel_margin = float(kappa[0] - kappa[1:].max())

    # First 50 incorrect (a, b) for diagnostic printing.
    wrong_pairs = []
    if int(wrong.sum()) > 0:
        ai, bi = np.where(wrong)
        for i, j in zip(ai[:50], bi[:50]):
            wrong_pairs.append({
                "a": int(i + 1), "b": int(j + 1),
                "target": int(target[i, j]),
                "predicted": int(predicted[i, j]),
                "target_logit": float(target_logit[i, j]),
                "max_alt_logit": float(max_alt[i, j]),
                "margin": float(margin[i, j]),
            })

    return {
        "run_dir": str(run_dir),
        "p": p,
        "n_wrong": int(wrong.sum()),
        "n_total": int(wrong.size),
        "accuracy": float(1 - wrong.mean()),
        # Margin view
        "margin": margin,
        "target_logit": target_logit,
        "max_alt_logit": max_alt,
        "correct_mask": ~wrong,
        "kernel_margin": kernel_margin,
        "wrong_pairs": wrong_pairs,
        # Per-token view
        "err_by_a": err_by_a,
        "err_by_b": err_by_b,
        "err_by_target": err_by_target,
        "wrong_mask": wrong,
        "pred": predicted,
        "target": target,
    }


def find_outlier_tokens(err: np.ndarray, threshold_sigma: float = 3.0) -> list[tuple[int, float]]:
    """Tokens whose error rate is > mean + threshold_sigma · std. Returned 1-indexed."""
    mean, std = err.mean(), err.std()
    out = [(i + 1, float(e)) for i, e in enumerate(err)
           if e > mean + threshold_sigma * std and e > 0]
    out.sort(key=lambda x: -x[1])
    return out


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_one(d: dict) -> None:
    name = Path(d["run_dir"]).name
    print(f"\n=== {name} (p={d['p']}) ===")
    print(f"  accuracy = {d['accuracy']:.4%}   "
          f"wrong = {d['n_wrong']} / {d['n_total']}   "
          f"kernel margin = {d['kernel_margin']:.4g}")

    m = d["margin"]
    print("  margin distribution:")
    print(f"    min: {m.min():.4g}   1%: {np.percentile(m, 1):.4g}   "
          f"median: {np.median(m):.4g}   "
          f"99%: {np.percentile(m, 99):.4g}   max: {m.max():.4g}")

    if d["n_wrong"] == 0:
        print("  (no errors — nothing to localize)")
        return

    a_out = find_outlier_tokens(d["err_by_a"])
    b_out = find_outlier_tokens(d["err_by_b"])
    t_out_arr = d["err_by_target"][1:]
    t_mean, t_std = t_out_arr.mean(), t_out_arr.std()
    t_out = [(i, float(e)) for i, e in enumerate(d["err_by_target"])
             if i > 0 and e > t_mean + 3 * t_std and e > 0]
    t_out.sort(key=lambda x: -x[1])

    print(f"  >3σ outlier tokens by err rate:")
    print(f"    as A: {a_out[:8]}")
    print(f"    as B: {b_out[:8]}")
    print(f"    as target c: {t_out[:8]}")
    a_set = {x for x, _ in a_out}
    b_set = {x for x, _ in b_out}
    print(f"    A∩B (A↔B-symmetric outliers): {sorted(a_set & b_set)}")

    print(f"  first 10 incorrect (a, b):")
    for w in d["wrong_pairs"][:10]:
        print(f"    a={w['a']:>3} b={w['b']:>3}  pred={w['predicted']:>3}  "
              f"target={w['target']:>3}  margin={w['margin']:+.3g}")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

_REGIME_COLOR = {
    7: "green", 8: "green", 10: "green", 12: "green",
    13: "darkorange", 14: "darkorange",
    9: "red",
}
_REGIME_LABEL = {
    7: "grokked", 8: "grokked", 10: "grokked", 12: "grokked",
    13: "noisy", 14: "noisy",
    9: "CRT-fail",
}


def _seed_color_regime(run_dir: str) -> tuple[str, str]:
    seed = int(Path(run_dir).name.split("seed")[-1])
    return _REGIME_COLOR.get(seed, "gray"), _REGIME_LABEL.get(seed, "?")


def plot_one(d: dict, out_path: Path) -> None:
    """Per-seed 4-panel: margin histogram, target-vs-alt scatter, error heatmap, per-token bars."""
    color, regime = _seed_color_regime(d["run_dir"])
    fig, axes = plt.subplots(2, 2, figsize=(13, 9.6))

    # (a) margin histogram
    ax = axes[0][0]
    m = d["margin"].ravel()
    ax.hist(m, bins=80, color="steelblue", edgecolor="black", linewidth=0.4)
    ax.axvline(0, color="red", linestyle="--", label="0 (error boundary)")
    ax.axvline(d["kernel_margin"], color="green", linestyle=":",
               label=f"kernel margin = {d['kernel_margin']:.2g}")
    ax.set_xlabel("margin = target − max_alt")
    ax.set_ylabel("# of (a, b) pairs")
    ax.set_title(f"{Path(d['run_dir']).name} ({regime})\n"
                 f"acc={d['accuracy']:.4%}  n_wrong={d['n_wrong']}")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # (b) target vs max_alt scatter
    ax = axes[0][1]
    correct = d["correct_mask"].ravel()
    tl = d["target_logit"].ravel()
    ma = d["max_alt_logit"].ravel()
    if (~correct).any():
        ax.scatter(tl[~correct], ma[~correct], s=4, color="red", alpha=0.7,
                   label=f"wrong (n={d['n_wrong']})")
    ax.scatter(tl[correct], ma[correct], s=2, color="C0", alpha=0.3, label="correct")
    lo, hi = min(tl.min(), ma.min()), max(tl.max(), ma.max())
    ax.plot([lo, hi], [lo, hi], "k--", alpha=0.3, label="margin=0")
    ax.set_xlabel("target logit  L(a, b, ab)")
    ax.set_ylabel("max-alternative logit")
    ax.set_title("target vs max-alt logits")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # (c) error heatmap
    ax = axes[1][0]
    p = d["p"]
    ax.imshow(d["wrong_mask"], cmap="Reds", aspect="auto",
              extent=[1, p, p, 1], interpolation="nearest")
    ax.set_xlabel("b")
    ax.set_ylabel("a")
    ax.set_title(f"where errors happen ({d['n_wrong']} / {(p-1)**2})")

    # (d) per-token error rate
    ax = axes[1][1]
    x = np.arange(1, p)
    ax.bar(x - 0.2, d["err_by_a"], width=0.4, color="steelblue", label="err by a")
    ax.bar(x + 0.2, d["err_by_b"], width=0.4, color="orange", label="err by b")
    ax.set_xlabel("token")
    ax.set_ylabel("error rate")
    ax.set_title("per-token error rate (A vs B)")
    ax.legend()
    ax.set_xlim(0, p)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_population(results: list[dict], out_dir: Path) -> None:
    """Population summary — 3 panels: overlaid margins, bottom-tail, accuracy."""
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))

    # (a) overlaid margin distributions (log y)
    ax = axes[0]
    for d in results:
        color, regime = _seed_color_regime(d["run_dir"])
        seed = int(Path(d["run_dir"]).name.split("seed")[-1])
        ax.hist(d["margin"].ravel(), bins=60, alpha=0.4, color=color,
                label=f"seed {seed} ({regime})", density=True)
    ax.axvline(0, color="black", linestyle="--", alpha=0.5)
    ax.set_xlabel("margin")
    ax.set_ylabel("density")
    ax.set_title("Per-example margin distributions")
    ax.set_yscale("log")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)

    # (b) bottom-500 tails
    ax = axes[1]
    for d in results:
        color, _ = _seed_color_regime(d["run_dir"])
        seed = int(Path(d["run_dir"]).name.split("seed")[-1])
        sorted_m = np.sort(d["margin"].ravel())
        ax.plot(sorted_m[:500], color=color, alpha=0.7, label=f"seed {seed}")
    ax.axhline(0, color="black", linestyle="--", alpha=0.5)
    ax.set_xlabel("rank (worst 500 examples)")
    ax.set_ylabel("margin")
    ax.set_title("Bottom 500 margins per seed")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (c) accuracy + error count
    ax = axes[2]
    for d in results:
        color, _ = _seed_color_regime(d["run_dir"])
        seed = int(Path(d["run_dir"]).name.split("seed")[-1])
        ax.scatter([d["accuracy"]], [d["n_wrong"]], c=color, s=120,
                   edgecolor="black", linewidth=0.6)
        ax.annotate(f" s{seed}", (d["accuracy"], d["n_wrong"]), fontsize=10)
    ax.set_xlabel("argmax accuracy")
    ax.set_ylabel(f"# wrong examples  (out of (p−1)²)")
    ax.set_yscale("symlog")
    ax.set_title("Accuracy & error count")
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out_dir / "population_summary.png", dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, required=True)
    ap.add_argument("--runs-root", default="experiments/003_dmodel_sweep_p113/runs")
    ap.add_argument("--tag", default="dmodel_24_dmlp_20_wd2")
    ap.add_argument("--out-dir", default="experiments/error_localization")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for s in args.seeds:
        r = Path(args.runs_root) / f"{args.tag}_seed{s}"
        if not r.exists():
            print(f"skip missing {r}")
            continue
        d = analyze_run(r)
        results.append(d)
        print_one(d)
        plot_one(d, out_dir / f"{r.name}.png")

    if results:
        plot_population(results, out_dir)
        print(f"\nWrote {len(results)} per-seed figures + population summary to {out_dir}/")


if __name__ == "__main__":
    main()
