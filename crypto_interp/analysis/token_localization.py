"""Per-token error localization.

For each (a, b) pair with the model's prediction wrong, attribute the error to:
    • input token a
    • input token b
    • target class c
    • the predicted class

Aggregate to find tokens that are anomalously bad. Then for each anomaly,
compare the token's embedding (in the character basis) to the population
average — looking for a structural defect at specific characters.

Usage:
    python -m crypto_interp.analysis.token_localization \\
        --seeds 7 8 10 12 13 14 \\
        --runs-root experiments/003_dmodel_sweep_p113/runs \\
        --tag dmodel_24_dmlp_20_wd2 \\
        --out-dir experiments/token_localization
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from crypto_interp.interp import Session


def analyze_one(run_dir: Path) -> dict:
    S = Session.from_run(str(run_dir))
    p = S.ds.p
    logits = S.logits_grid.detach().cpu().numpy()[..., :p].astype(np.float64)

    a_idx = np.arange(1, p)[:, None]
    b_idx = np.arange(1, p)[None, :]
    target = (a_idx * b_idx) % p
    pred = logits.argmax(axis=-1)
    wrong = pred != target

    # Per-input-token error rates.
    err_by_a = wrong.mean(axis=1)   # shape (p-1,) — error rate when a = a_idx[i]
    err_by_b = wrong.mean(axis=0)   # shape (p-1,) — error rate when b = b_idx[j]
    # Per-target-class error rate (across all (a,b) that should produce class c).
    err_by_target = np.zeros(p)
    target_count = np.zeros(p, dtype=np.int64)
    np.add.at(err_by_target, target.ravel(), wrong.ravel().astype(np.int64))
    np.add.at(target_count, target.ravel(), 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        err_by_target = err_by_target / np.maximum(target_count, 1)

    # Per-token embedding norm — simple diagnostic. Token x corresponds to
    # W_E[:, x]. For multiplicative inputs we look at columns x = 1..p-1.
    W_E = S.model.embed.W_E[:, :p].detach().cpu().numpy()  # shape (d_model, p)
    W_E_mul = W_E[:, 1:p]  # shape (d_model, p-1) for tokens 1..p-1
    embed_norm = np.linalg.norm(W_E_mul, axis=0)  # shape (p-1,) — per token

    return {
        "run_dir": str(run_dir),
        "p": p,
        "n_wrong": int(wrong.sum()),
        "accuracy": float(1 - wrong.mean()),
        "err_by_a": err_by_a,
        "err_by_b": err_by_b,
        "err_by_target": err_by_target,
        "wrong_mask": wrong,
        "pred": pred,
        "target": target,
        "embed_norm": embed_norm,
    }


def find_outlier_tokens(err: np.ndarray, threshold_sigma: float = 3.0) -> list[int]:
    """Tokens whose error rate is > mean + threshold_sigma·std."""
    mean, std = err.mean(), err.std()
    out = []
    for i, e in enumerate(err):
        if e > mean + threshold_sigma * std and e > 0:
            out.append((i + 1, float(e)))  # +1 because index 0 in err corresponds to token 1
    out.sort(key=lambda x: -x[1])
    return out


def print_one(d: dict) -> None:
    name = Path(d["run_dir"]).name
    print(f"\n=== {name} (p={d['p']}) ===")
    print(f"  total wrong: {d['n_wrong']} / {(d['p']-1)**2}  (acc {d['accuracy']:.4%})")
    if d['n_wrong'] == 0:
        print("  (no errors — nothing to localize)")
        return

    print("\n  per-input-token A error rate:")
    a_out = find_outlier_tokens(d["err_by_a"])
    print(f"    mean = {d['err_by_a'].mean():.4%},  std = {d['err_by_a'].std():.4%}")
    print(f"    >3σ outliers ({len(a_out)}): {a_out[:10]}")

    print("\n  per-input-token B error rate:")
    b_out = find_outlier_tokens(d["err_by_b"])
    print(f"    mean = {d['err_by_b'].mean():.4%},  std = {d['err_by_b'].std():.4%}")
    print(f"    >3σ outliers ({len(b_out)}): {b_out[:10]}")

    print("\n  per-target-class error rate:")
    t_out_arr = d["err_by_target"][1:]  # skip class 0
    mean, std = t_out_arr.mean(), t_out_arr.std()
    t_out = []
    for i, e in enumerate(d["err_by_target"]):
        if i == 0:
            continue
        if e > mean + 3 * std and e > 0:
            t_out.append((i, float(e)))
    t_out.sort(key=lambda x: -x[1])
    print(f"    mean = {mean:.4%},  std = {std:.4%}")
    print(f"    >3σ outliers ({len(t_out)}): {t_out[:10]}")

    # Cross-tabulation: do the bad-a tokens overlap with the bad-b tokens?
    a_set = {x for x, _ in a_out}
    b_set = {x for x, _ in b_out}
    overlap = a_set & b_set
    print(f"\n  outlier overlap A ∩ B: {sorted(overlap)[:10]} (n={len(overlap)})")
    print(f"  union A ∪ B: {len(a_set | b_set)} tokens")


def plot_population(results: list[dict], out_dir: Path) -> None:
    """For each seed, a 3-panel figure: err_by_a, err_by_b, err_by_target."""
    n_seeds = len(results)
    fig, axes = plt.subplots(n_seeds, 3, figsize=(14, 2.8 * n_seeds), squeeze=False)
    for i, d in enumerate(results):
        seed = int(Path(d["run_dir"]).name.split("seed")[-1])
        if seed in {7, 8, 10, 12}: color = "green"
        elif seed in {13, 14}:     color = "darkorange"
        elif seed == 9:            color = "red"
        else:                      color = "gray"

        ax = axes[i][0]
        x = np.arange(1, d["p"])
        ax.bar(x, d["err_by_a"], color=color, alpha=0.85, width=1.0)
        ax.set_xlabel("input token a")
        ax.set_ylabel("err rate")
        ax.set_title(f"seed {seed}: error by a-token")
        ax.set_xlim(0, d["p"])

        ax = axes[i][1]
        ax.bar(x, d["err_by_b"], color=color, alpha=0.85, width=1.0)
        ax.set_xlabel("input token b")
        ax.set_ylabel("err rate")
        ax.set_title(f"seed {seed}: error by b-token")
        ax.set_xlim(0, d["p"])

        ax = axes[i][2]
        # err_by_target is indexed 0..p-1; skip 0.
        ax.bar(np.arange(1, d["p"]), d["err_by_target"][1:],
               color=color, alpha=0.85, width=1.0)
        ax.set_xlabel("target class c")
        ax.set_ylabel("err rate")
        ax.set_title(f"seed {seed}: error by target")
        ax.set_xlim(0, d["p"])

    fig.tight_layout()
    out = out_dir / "population_token_errors.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"\nWrote {out}")


def plot_seed14_zoom(d: dict, out_dir: Path) -> None:
    """For seed 14 specifically: plot wrong (a, b) pairs as a heatmap + the
    embedding norms of the worst tokens."""
    p = d["p"]
    wrong = d["wrong_mask"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # (a) heatmap of wrong (a, b) pairs
    ax = axes[0]
    ax.imshow(wrong, cmap="Reds", aspect="auto",
              extent=[1, p, p, 1], interpolation="nearest")
    ax.set_xlabel("b")
    ax.set_ylabel("a")
    ax.set_title(f"{Path(d['run_dir']).name}: where errors happen\n"
                 f"({d['n_wrong']} wrong out of {(p-1)**2})")

    # (b) error rate by a, b on same axis
    ax = axes[1]
    x = np.arange(1, p)
    ax.bar(x - 0.2, d["err_by_a"], width=0.4, color="steelblue", label="err by a")
    ax.bar(x + 0.2, d["err_by_b"], width=0.4, color="orange", label="err by b")
    ax.set_xlabel("token")
    ax.set_ylabel("error rate")
    ax.set_title("Per-token error rate: A vs B")
    ax.legend()
    ax.set_xlim(0, p)
    fig.tight_layout()
    name = Path(d["run_dir"]).name
    out = out_dir / f"{name}_zoom.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"Wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, required=True)
    ap.add_argument("--runs-root", default="experiments/003_dmodel_sweep_p113/runs")
    ap.add_argument("--tag", default="dmodel_24_dmlp_20_wd2")
    ap.add_argument("--out-dir", default="experiments/token_localization")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for s in args.seeds:
        r = Path(args.runs_root) / f"{args.tag}_seed{s}"
        if not r.exists():
            print(f"skip missing {r}")
            continue
        d = analyze_one(r)
        results.append(d)
        print_one(d)

    if results:
        plot_population(results, out_dir)
        # Zoom on the most informative noisy seeds
        for d in results:
            seed = int(Path(d['run_dir']).name.split("seed")[-1])
            if seed in {13, 14}:
                plot_seed14_zoom(d, out_dir)


if __name__ == "__main__":
    main()
