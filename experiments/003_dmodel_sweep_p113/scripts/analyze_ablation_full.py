"""Full per-character ablation sweep.

For a trained model, zero out W_E's projection onto each of the 56 multiplicative
characters (one at a time) and measure the resulting test loss. Plot the
distribution of "essentialness" vs character order and vs the character's
energy in W_E.

This answers: is the "K is sparse" claim correct (only ~4 characters used) or
is there a smooth tail of load-bearing characters down to the noise floor?

Outputs:
  data/basis_dynamics/ablation_full_<tag>.csv
  figures/basis_dynamics/ablation_full_<tag>.png
"""
from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from crypto_interp.interp.bases import multiplicative_fourier_basis
from crypto_interp.interp.load import load_run
from crypto_interp.training.loop import cross_entropy_high_precision

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures" / "basis_dynamics"
OUT_DIR = ROOT / "data" / "basis_dynamics"

ORDER_COLOR = {
    2: "#9467bd", 4: "#8c564b", 7: "#bcbd22", 8: "#17becf",
    14: "#e377c2", 16: "#1f77b4", 28: "#2ca02c", 56: "#ff7f0e", 112: "#d62728",
}


def order_of(k, n=112):
    return n // math.gcd(k, n)


def build_basis_indexed():
    basis, names, _ = multiplicative_fourier_basis(113)
    char_idx = {}
    for i, nm in enumerate(names):
        m = re.match(r"mul (cos|sin) (\d+)", nm)
        if m:
            kk = int(m.group(2))
            char_idx.setdefault(kk, []).append(i)
    return basis.double(), names, char_idx


def char_energy(W_E_dp, basis, char_idx, n_chars=56):
    coef = torch.einsum("kp,dp->kd", basis, W_E_dp.double())
    E = (coef ** 2).sum(dim=1).cpu().numpy()
    out = np.zeros(n_chars)
    for k, rs in char_idx.items():
        out[k - 1] = float(E[rs].sum())
    return out


def ablate_character(W_E_full, basis, char_idx, k_remove, p=113):
    W_E_new = W_E_full.clone().double()
    W_E_v = W_E_new[:, :p].clone()
    rows = char_idx[k_remove]
    for r in rows:
        b = basis[r].double()
        coef = (W_E_v @ b)
        W_E_v = W_E_v - coef[:, None] * b[None, :]
    W_E_new[:, :p] = W_E_v
    return W_E_new


def compute_test_loss(model, ds):
    mask = ds.test_mask.bool()
    with torch.no_grad():
        logits = model(ds.inputs[mask])[:, -1, :ds.n_answer_tokens]
    return float(cross_entropy_high_precision(logits, ds.labels[mask], True).item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    run_dir = Path(args.run_dir).resolve()
    tag = args.tag or run_dir.name
    ck = sorted(run_dir.glob("checkpoint_*.pt"))
    print(f"Loading {ck[-1]}")
    model, ds, _ = load_run(ck[-1])
    model.eval()

    basis, names, char_idx = build_basis_indexed()
    p = ds.p
    W_E_orig = model.embed.W_E.detach().clone()

    char_E = char_energy(W_E_orig[:, :p], basis, char_idx)
    K_top = sorted([k + 1 for k, e in enumerate(char_E) if e >= 0.05 * char_E.max()])
    print(f"Top-K (5% threshold): {K_top}")

    base = compute_test_loss(model, ds)
    print(f"Baseline test loss: {base:.4e}  (log10 = {np.log10(base):.3f})")

    rows = []
    for k in range(1, 57):
        W_E_ab = ablate_character(W_E_orig, basis, char_idx, k, p=p)
        with torch.no_grad():
            model.embed.W_E.copy_(W_E_ab.to(model.embed.W_E.dtype))
        ablated = compute_test_loss(model, ds)
        with torch.no_grad():
            model.embed.W_E.copy_(W_E_orig)
        rows.append(dict(
            k=k, order=order_of(k),
            energy=char_E[k - 1],
            ablated=ablated,
            delta_log=np.log10(ablated) - np.log10(base),
            in_top_K=k in K_top,
        ))

    rows.sort(key=lambda r: r["delta_log"], reverse=True)
    print(f"\n{'k':>4} {'o':>4} {'energy':>10} {'ablated':>12} {'Δlog10':>8} {'topK':>5}")
    for r in rows:
        print(f"{r['k']:>4} {r['order']:>4} {r['energy']:>10.3f} "
              f"{r['ablated']:>12.4e} {r['delta_log']:>+8.3f} "
              f"{'Y' if r['in_top_K'] else '.':>5}")

    # CSV.
    csv_path = OUT_DIR / f"ablation_full_{tag}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            r["in_top_K"] = "Y" if r["in_top_K"] else "N"
            w.writerow(r)
    print(f"\nSaved {csv_path}")

    # Plot: energy vs Δlog10 ablation impact.
    rows.sort(key=lambda r: r["k"])  # restore by-k order
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

    ax = axes[0]
    for r in rows:
        c = ORDER_COLOR.get(r["order"], "#aaaaaa")
        marker = "*" if r["in_top_K"] else "o"
        ms = 14 if r["in_top_K"] else 7
        ax.scatter(r["energy"], r["delta_log"], color=c,
                   marker=marker, s=ms ** 2, alpha=0.85,
                   edgecolor="black", linewidth=0.5)
        if r["delta_log"] > 1.5 or r["in_top_K"]:
            ax.annotate(f"k={r['k']}", (r["energy"], r["delta_log"]),
                        xytext=(4, 4), textcoords="offset points", fontsize=7)
    ax.set_xscale("log")
    ax.set_xlabel("W_E energy in character k")
    ax.set_ylabel("Δlog10(test loss) after ablating k")
    ax.set_title(f"(a) per-character ablation impact ({tag})")
    ax.grid(True, alpha=0.3)
    handles = [plt.Line2D([0], [0], marker="o", color="w",
                          markerfacecolor=c, markersize=8,
                          markeredgecolor="black", label=f"order {o}")
               for o, c in ORDER_COLOR.items()]
    handles.append(plt.Line2D([0], [0], marker="*", color="w",
                              markerfacecolor="white", markersize=12,
                              markeredgecolor="black", label="top-K (5% thresh)"))
    ax.legend(handles=handles, fontsize=7, loc="lower right", ncol=2)

    ax = axes[1]
    rows_sorted = sorted(rows, key=lambda r: r["delta_log"], reverse=True)
    xs = np.arange(len(rows_sorted))
    colors = [ORDER_COLOR.get(r["order"], "#aaaaaa") for r in rows_sorted]
    bars = ax.bar(xs, [r["delta_log"] for r in rows_sorted], color=colors,
                  edgecolor="black", linewidth=0.4)
    for i, r in enumerate(rows_sorted):
        if r["in_top_K"]:
            ax.text(i, r["delta_log"] + 0.1, f"k={r['k']}",
                    fontsize=7, ha="center", fontweight="bold")
    ax.axhline(np.log10(0.1) - np.log10(base), color="red", ls="--", lw=0.7,
               alpha=0.5, label="loss=0.1 (random model)")
    ax.set_xlabel("characters sorted by ablation impact")
    ax.set_ylabel("Δlog10(test loss) after ablating k")
    ax.set_title(f"(b) ranked ablation impact")
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(fontsize=8, loc="upper right")

    fig.suptitle(f"Full per-character ablation, baseline test loss = {base:.2e}",
                 fontsize=11)
    fig.tight_layout()
    out = FIG_DIR / f"ablation_full_{tag}.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
