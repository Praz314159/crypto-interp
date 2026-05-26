"""Test: do the failing tokens have anomalous embeddings outside the K subspace?

The approximate-CRT algorithm requires W_E to lie in the span of characters K
— i.e., each row of W_E (viewed as a function on (Z/p)*) should be a linear
combination of cos_k, sin_k for k ∈ K. Energy outside that span is structural
leakage that the MLP/unembed cannot fully suppress.

Per-token test:
    1. Project each row of W_E_mul onto the K-character subspace via OLS regression.
    2. Compute the residual W_E_mul − reconstruction.
    3. For each token x, |residual[:, x]|² is the per-token "K-leakage energy".
    4. Correlate per-token K-leakage with per-token error rate.

If the failing tokens (e.g., 56, 98 in seed 14; 43, 68, 71, 109 in seed 13)
have disproportionately large K-leakage, we've found the mechanistic cause:
specific embedding columns leak energy onto non-K characters, producing
asymmetric noise that exceeds the kernel margin only at those columns.

Usage:
    python -m crypto_interp.analysis.embedding_residual \\
        --seeds 13 14 7 12 \\
        --runs-root experiments/003_dmodel_sweep_p113/runs \\
        --tag dmodel_24_dmlp_20_wd2 \\
        --out-dir experiments/embedding_residual
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from crypto_interp.interp import Session


def per_token_k_leakage(S: Session, K: list[int]) -> dict:
    """Per-token energy outside the K-character subspace.

    Returns a dict with:
        leakage_norm: shape (p-1,), per-token residual norm squared
        clean_norm:   shape (p-1,), per-token in-K norm squared
        total_norm:   shape (p-1,), per-token total embedding norm squared
        leakage_frac: shape (p-1,), leakage / total per token
    """
    p = S.ds.p
    n = p - 1
    # Embedding for tokens 1..p-1, ordered by RESIDUE (not dlog).
    W_E = S.model.embed.W_E[:, 1:p].detach().cpu().numpy()  # (d_model, p-1)

    # The harness's basis is ORDERED BY DLOG. We need to remap W_E columns into
    # dlog order so the regression columns match the cos/sin functions.
    from crypto_interp.interp.bases import discrete_log_table
    _g, dlog = discrete_log_table(p)
    # dlog_arr[i] = dlog of token (i+1) in residue order.
    dlog_arr = np.array([dlog[i + 1] for i in range(n)], dtype=np.int64)
    # Permutation: W_E_dlog[:, j] = W_E[:, residue with dlog j] - 1.
    # I.e. perm such that for each j, dlog of perm[j]+1 equals j.
    inv_dlog = np.zeros(n, dtype=np.int64)
    for residue_idx, x in enumerate(dlog_arr):
        inv_dlog[x] = residue_idx
    W_E_dlog = W_E[:, inv_dlog]  # (d_model, p-1) in dlog order

    # Build the cos/sin basis matrix B of shape (n, 2*|K|), each column a basis
    # function evaluated at dlog j = 0..n-1.
    K_pos = [k for k in K if k > 0]
    cols = []
    col_labels = []
    for k in K_pos:
        cols.append(np.cos(2 * np.pi * k * np.arange(n) / n))
        col_labels.append(f"cos_{k}")
        cols.append(np.sin(2 * np.pi * k * np.arange(n) / n))
        col_labels.append(f"sin_{k}")
    # Plus the constant
    cols.append(np.ones(n))
    col_labels.append("const")
    B = np.stack(cols, axis=1)  # shape (n, 2*|K_pos| + 1)

    # Solve W_E_dlog ≈ A @ B.T  i.e. find A (d_model, n_basis) s.t. W_E_dlog ≈ A @ B.T
    # Equivalently: for each row of W_E_dlog, fit a linear combination of the basis cols.
    # Closed-form: A.T = (B.T B)^-1 B.T W_E_dlog.T
    # OLS row-by-row.
    BtB_inv = np.linalg.pinv(B.T @ B)
    A = (BtB_inv @ B.T @ W_E_dlog.T).T   # shape (d_model, n_basis)
    reconstruction_dlog = A @ B.T        # shape (d_model, n)
    residual_dlog = W_E_dlog - reconstruction_dlog

    # Per-token norms, indexed by dlog.
    leakage_per_dlog = np.sum(residual_dlog ** 2, axis=0)         # (n,)
    clean_per_dlog = np.sum(reconstruction_dlog ** 2, axis=0)     # (n,)
    total_per_dlog = np.sum(W_E_dlog ** 2, axis=0)                # (n,)

    # Map back to residue order: position i in residue order has dlog dlog_arr[i].
    leakage_per_residue = leakage_per_dlog[dlog_arr]
    clean_per_residue   = clean_per_dlog[dlog_arr]
    total_per_residue   = total_per_dlog[dlog_arr]
    leakage_frac        = leakage_per_residue / np.maximum(total_per_residue, 1e-12)

    # Also: total leakage power (sum across tokens) — sanity check.
    total_leakage_power = float(leakage_per_dlog.sum())
    total_signal_power = float(clean_per_dlog.sum())

    return {
        "leakage_per_token": leakage_per_residue,
        "clean_per_token": clean_per_residue,
        "total_per_token": total_per_residue,
        "leakage_frac": leakage_frac,
        "K": K_pos,
        "total_leakage_power": total_leakage_power,
        "total_signal_power": total_signal_power,
    }


def analyze_run(run_dir: Path) -> dict:
    S = Session.from_run(str(run_dir))
    K = list(S.essential()["K"])
    leakage = per_token_k_leakage(S, K)

    # Per-token error rate from logits
    p = S.ds.p
    logits = S.logits_grid.detach().cpu().numpy()[..., :p]
    a_idx = np.arange(1, p)[:, None]
    b_idx = np.arange(1, p)[None, :]
    target = (a_idx * b_idx) % p
    pred = logits.argmax(axis=-1)
    wrong = pred != target
    err_by_a = wrong.mean(axis=1)
    err_by_b = wrong.mean(axis=0)
    # Combined: token's error rate regardless of position.
    err_per_token = (err_by_a + err_by_b) / 2

    return {
        "run_dir": str(run_dir),
        "p": p,
        "K": K,
        "leakage": leakage,
        "err_per_token": err_per_token,
        "n_wrong": int(wrong.sum()),
    }


def print_one(d: dict) -> None:
    name = Path(d["run_dir"]).name
    lk = d["leakage"]
    print(f"\n=== {name} ===")
    print(f"  K = {d['K']}")
    print(f"  Total signal power (in-K):  {lk['total_signal_power']:.4g}")
    print(f"  Total leakage power (out-of-K): {lk['total_leakage_power']:.4g}")
    print(f"  global leakage frac: {lk['total_leakage_power']/(lk['total_signal_power']+lk['total_leakage_power']):.3%}")
    if d["n_wrong"] == 0:
        print("  (no errors — grokked)")
    else:
        print(f"  n_wrong = {d['n_wrong']}")

    # Per-token: correlate err_per_token with leakage_frac
    err = d["err_per_token"]
    lf = lk["leakage_frac"]
    valid = (err.std() > 0) and (lf.std() > 0)
    if valid:
        corr = np.corrcoef(err, lf)[0, 1]
        print(f"  Pearson corr(err_rate, leakage_frac) = {corr:+.4f}")
        # Spearman would be more robust to outliers
        from scipy.stats import spearmanr
        rho, pval = spearmanr(err, lf)
        print(f"  Spearman corr = {rho:+.4f}  (p={pval:.3g})")

    # Print the top 5 leaky tokens
    top_leaky = np.argsort(lf)[::-1][:8]
    print("\n  Top 8 leakiest tokens (token=residue, frac=leakage/total, err_rate):")
    for i in top_leaky:
        print(f"    token {i+1:>3}  leakage_frac={lf[i]:.4%}  "
              f"err_rate={err[i]:.4%}  total_norm²={lk['total_per_token'][i]:.4g}")


def plot_one(d: dict, ax_scatter, ax_bars) -> None:
    err = d["err_per_token"]
    lf = d["leakage"]["leakage_frac"]
    seed = int(Path(d["run_dir"]).name.split("seed")[-1])
    if seed in {7, 8, 10, 12}: color, regime = "green", "grokked"
    elif seed in {13, 14}:     color, regime = "darkorange", "noisy"
    elif seed == 9:            color, regime = "red", "CRT-fail"
    else:                      color, regime = "gray", "?"

    # Scatter: leakage_frac vs error rate
    ax_scatter.scatter(lf * 100, err * 100, s=24, c=color, alpha=0.6, edgecolor="black", linewidth=0.3)
    # Annotate the top-3 by err rate
    top_err = np.argsort(err)[::-1][:3]
    for i in top_err:
        if err[i] > 0:
            ax_scatter.annotate(f" tok {i+1}", (lf[i]*100, err[i]*100), fontsize=8)
    ax_scatter.set_xlabel("K-leakage fraction (%)")
    ax_scatter.set_ylabel("err rate (%)")
    ax_scatter.set_title(f"seed {seed} ({regime}): leakage vs errors")
    ax_scatter.grid(True, alpha=0.3)

    # Bars: per-token leakage_frac
    p = d["p"]
    x = np.arange(1, p)
    ax_bars.bar(x, lf * 100, color=color, alpha=0.7, width=1.0)
    ax_bars.set_xlabel("token (residue)")
    ax_bars.set_ylabel("K-leakage (%)")
    ax_bars.set_title(f"seed {seed}: per-token K-leakage")
    ax_bars.set_xlim(0, p)


def plot_population(results: list[dict], out_dir: Path) -> None:
    n = len(results)
    fig, axes = plt.subplots(n, 2, figsize=(13, 3.2 * n), squeeze=False)
    for i, d in enumerate(results):
        plot_one(d, axes[i][0], axes[i][1])
    fig.tight_layout()
    out = out_dir / "population.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"\nWrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, required=True)
    ap.add_argument("--runs-root", default="experiments/003_dmodel_sweep_p113/runs")
    ap.add_argument("--tag", default="dmodel_24_dmlp_20_wd2")
    ap.add_argument("--out-dir", default="experiments/embedding_residual")
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

    if results:
        plot_population(results, out_dir)


if __name__ == "__main__":
    main()
