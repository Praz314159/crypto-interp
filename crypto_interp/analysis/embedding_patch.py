"""Causal patch test: replace failing tokens' embeddings with their K-projection,
re-evaluate accuracy, and see if the error rate collapses.

For each token x, the K-projection of W_E[:, x] is the part of the embedding
that lies in span{cos_k, sin_k : k ∈ K}. The residual is the "out-of-K
leakage." If leakage causes the failures in seed 14, then replacing
W_E[:, 56] (and 98) with their K-projections should restore accuracy.

We sweep a leakage-fraction threshold T: tokens with leakage_frac ≥ T are
patched, others left alone. T=100% means no patch (baseline). T=0% means
patch every token. The interesting threshold is where seed 14 recovers — if
it does at, say, T=25% (catching tokens 56 and 98), the localization story
is complete.

Usage:
    python -m crypto_interp.analysis.embedding_patch \\
        --seeds 13 14 7 12 \\
        --runs-root experiments/003_dmodel_sweep_p113/runs \\
        --tag dmodel_24_dmlp_20_wd2 \\
        --out-dir experiments/embedding_patch
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from crypto_interp.interp import Session
from crypto_interp.interp.bases import discrete_log_table


def compute_k_projection(W_E_mul: np.ndarray, K: list[int], p: int) -> tuple[np.ndarray, np.ndarray]:
    """Project each column of W_E_mul onto span{cos_k, sin_k : k ∈ K, k>0} + const.

    W_E_mul is shape (d_model, p-1) in *residue* order (column i = token i+1).
    Returns (projection, residual) both shape (d_model, p-1) in residue order.
    """
    n = p - 1
    _g, dlog = discrete_log_table(p)
    dlog_arr = np.array([dlog[i + 1] for i in range(n)], dtype=np.int64)
    inv_dlog = np.zeros(n, dtype=np.int64)
    for residue_idx, x in enumerate(dlog_arr):
        inv_dlog[x] = residue_idx

    # Reorder W_E_mul into dlog order.
    W_dlog = W_E_mul[:, inv_dlog]  # (d_model, n)
    K_pos = [k for k in K if k > 0]
    cols = []
    for k in K_pos:
        cols.append(np.cos(2 * np.pi * k * np.arange(n) / n))
        cols.append(np.sin(2 * np.pi * k * np.arange(n) / n))
    cols.append(np.ones(n))
    B = np.stack(cols, axis=1)   # (n, 2|K_pos|+1)

    # OLS row-by-row: A = (B^T B)^-1 B^T W_dlog.T
    BtB_inv = np.linalg.pinv(B.T @ B)
    A = (BtB_inv @ B.T @ W_dlog.T).T   # (d_model, n_basis)
    reconstruction_dlog = A @ B.T       # (d_model, n)
    residual_dlog = W_dlog - reconstruction_dlog

    # Reorder back to residue order.
    reconstruction = np.empty_like(W_E_mul)
    residual = np.empty_like(W_E_mul)
    for residue_idx in range(n):
        reconstruction[:, residue_idx] = reconstruction_dlog[:, dlog_arr[residue_idx]]
        residual[:, residue_idx] = residual_dlog[:, dlog_arr[residue_idx]]
    return reconstruction, residual


def evaluate_with_patched_embed(S: Session, new_W_E_mul: np.ndarray) -> tuple[float, float, int]:
    """Swap W_E[:, 1:p] for new_W_E_mul, recompute accuracy and CE, restore."""
    p = S.ds.p
    W_E = S.model.embed.W_E
    original = W_E[:, 1:p].detach().clone()
    try:
        with torch.no_grad():
            W_E[:, 1:p] = torch.tensor(new_W_E_mul, dtype=W_E.dtype, device=W_E.device)
        # Invalidate cached grids on session, then recompute.
        S._logits_grid = None
        logits = S.logits_grid.detach().cpu().numpy()[..., :p].astype(np.float64)
        a_idx = np.arange(1, p)[:, None]
        b_idx = np.arange(1, p)[None, :]
        target = (a_idx * b_idx) % p
        pred = logits.argmax(axis=-1)
        n_wrong = int((pred != target).sum())
        accuracy = float(1 - n_wrong / target.size)

        # CE
        m = logits.max(axis=-1, keepdims=True)
        lse = m.squeeze(-1) + np.log(np.sum(np.exp(logits - m), axis=-1))
        target_logit = np.take_along_axis(logits, target[..., None], axis=-1).squeeze(-1)
        ce = float((lse - target_logit).mean())
        return accuracy, ce, n_wrong
    finally:
        with torch.no_grad():
            W_E[:, 1:p] = original
        S._logits_grid = None


def patch_experiment(run_dir: Path, thresholds: list[float]) -> dict:
    S = Session.from_run(str(run_dir))
    p = S.ds.p
    K = list(S.essential()["K"])

    # Baseline (no patch)
    W_E_mul = S.model.embed.W_E[:, 1:p].detach().cpu().numpy().copy()
    baseline_acc, baseline_ce, baseline_wrong = evaluate_with_patched_embed(S, W_E_mul)

    # K-projection
    proj, resid = compute_k_projection(W_E_mul, K, p)
    leakage_per_token = np.sum(resid ** 2, axis=0)  # (p-1,)
    total_per_token = np.sum(W_E_mul ** 2, axis=0)
    leakage_frac = leakage_per_token / np.maximum(total_per_token, 1e-12)

    # For each threshold, patch tokens with leakage_frac >= T (T in fraction units 0..1)
    rows = []
    for T in thresholds:
        mask = leakage_frac >= T
        n_patched = int(mask.sum())
        patched = W_E_mul.copy()
        patched[:, mask] = proj[:, mask]
        acc, ce, n_wrong = evaluate_with_patched_embed(S, patched)
        rows.append({
            "T": float(T),
            "n_patched": n_patched,
            "patched_tokens": [int(i + 1) for i in np.where(mask)[0]],
            "acc": acc,
            "ce": ce,
            "n_wrong": n_wrong,
        })

    return {
        "run_dir": str(run_dir),
        "p": p,
        "K": K,
        "baseline": {
            "acc": baseline_acc,
            "ce": baseline_ce,
            "n_wrong": baseline_wrong,
        },
        "thresholds": rows,
        "leakage_frac": leakage_frac,
    }


def print_one(d: dict) -> None:
    name = Path(d["run_dir"]).name
    print(f"\n=== {name} ===")
    print(f"  K = {d['K']}")
    b = d["baseline"]
    print(f"  baseline:  acc={b['acc']:.4%}  CE={b['ce']:.4g}  n_wrong={b['n_wrong']}")
    print(f"\n  Patch threshold T (leakage_frac ≥ T) → patched tokens, new accuracy:")
    print(f"  {'T':>6}  {'n_patched':>9}  {'acc':>10}  {'CE':>11}  {'n_wrong':>7}  patched")
    for r in d["thresholds"]:
        pt = r["patched_tokens"][:10]
        more = f"... (+{len(r['patched_tokens'])-10})" if len(r['patched_tokens']) > 10 else ""
        print(f"  {r['T']*100:>5.1f}%  {r['n_patched']:>9}  "
              f"{r['acc']:>10.4%}  {r['ce']:>11.4g}  {r['n_wrong']:>7}  {pt} {more}")


def plot_population(results: list[dict], out_dir: Path) -> None:
    n_seeds = len(results)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

    # (a) accuracy vs threshold
    ax = axes[0]
    for d in results:
        seed = int(Path(d["run_dir"]).name.split("seed")[-1])
        if seed in {7, 8, 10, 12}: color, regime = "green", "grokked"
        elif seed in {13, 14}:     color, regime = "darkorange", "noisy"
        elif seed == 9:            color, regime = "red", "CRT-fail"
        else:                      color, regime = "gray", "?"
        Ts = [r["T"] for r in d["thresholds"]]
        accs = [r["acc"] for r in d["thresholds"]]
        ax.plot([1.0] + Ts, [d["baseline"]["acc"]] + accs, "o-", color=color,
                label=f"seed {seed} ({regime})", linewidth=2)
    ax.set_xlabel("leakage threshold T  (patch all tokens with frac ≥ T)")
    ax.set_ylabel("argmax accuracy")
    ax.set_title("Patching high-leakage tokens with their K-projection")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    ax.set_xlim(1.05, -0.05)  # T sweeps from 1 (no patch) down to 0 (all)

    # (b) CE vs threshold (log scale)
    ax = axes[1]
    for d in results:
        seed = int(Path(d["run_dir"]).name.split("seed")[-1])
        if seed in {7, 8, 10, 12}: color = "green"
        elif seed in {13, 14}:     color = "darkorange"
        elif seed == 9:            color = "red"
        else:                      color = "gray"
        Ts = [r["T"] for r in d["thresholds"]]
        ces = [r["ce"] for r in d["thresholds"]]
        ax.semilogy([1.0] + Ts, [d["baseline"]["ce"]] + ces, "o-", color=color,
                    label=f"seed {seed}", linewidth=2)
    ax.set_xlabel("leakage threshold T")
    ax.set_ylabel("CE (log)")
    ax.set_title("CE under embedding patch")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=9)
    ax.set_xlim(1.05, -0.05)
    fig.tight_layout()
    out = out_dir / "patch_curve.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"\nWrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, required=True)
    ap.add_argument("--runs-root", default="experiments/003_dmodel_sweep_p113/runs")
    ap.add_argument("--tag", default="dmodel_24_dmlp_20_wd2")
    ap.add_argument("--out-dir", default="experiments/embedding_patch")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Sweep from "patch nothing" to "patch everything".
    thresholds = [0.50, 0.40, 0.30, 0.25, 0.20, 0.15, 0.10, 0.05, 0.0]

    results = []
    for s in args.seeds:
        r = Path(args.runs_root) / f"{args.tag}_seed{s}"
        if not r.exists():
            print(f"skip missing {r}")
            continue
        d = patch_experiment(r, thresholds)
        results.append(d)
        print_one(d)

    if results:
        plot_population(results, out_dir)


if __name__ == "__main__":
    main()
