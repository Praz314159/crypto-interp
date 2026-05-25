"""Full per-character ablation sweep.

For a trained model, ablate each multiplicative character from W_E one at a time
and measure the resulting test loss. Plot "essentialness" (Δlog10 test loss) vs
character order and vs the character's W_E energy. Answers whether K is genuinely
sparse or has a smooth tail of load-bearing characters.

Uses crypto_interp.interp.essential_characters (prime-parametric).

Writes ``ablation_full_<tag>.{csv,png}`` to --out-dir (default: the run dir).

Usage:
    python -m crypto_interp.analysis.ablation_full \
        --run-dir experiments/003_dmodel_sweep_p113/runs/dmodel_24_dmlp_32_seed1
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from crypto_interp.interp import char_index, essential_characters, load_run

ORDER_COLOR = {
    2: "#9467bd", 4: "#8c564b", 7: "#bcbd22", 8: "#17becf",
    14: "#e377c2", 16: "#1f77b4", 28: "#2ca02c", 56: "#ff7f0e", 112: "#d62728",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--out-dir", default=None, help="Where to write csv/png (default: run dir).")
    args = ap.parse_args()
    run_dir = Path(args.run_dir).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else run_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.tag or run_dir.name
    ck = sorted(run_dir.glob("checkpoint_*.pt"))
    print(f"Loading {ck[-1]}")
    model, ds, _ = load_run(ck[-1])
    model.eval()
    basis, ci = char_index(ds.p)

    res = essential_characters(model, ds, ci, basis)
    base, per, K_top = res["base_loss"], res["per_char"], set(res["K"])
    print(f"Top-K (5% threshold): {sorted(K_top)}")
    print(f"Baseline test loss: {base:.4e}  (log10 = {np.log10(base):.3f})")

    rows = [dict(k=k, order=per[k]["order"], energy=per[k]["energy"],
                 ablated=per[k]["ablated_loss"], delta_log=per[k]["dlog10"],
                 in_top_K=k in K_top) for k in ci.freqs]

    rows.sort(key=lambda r: r["delta_log"], reverse=True)
    print(f"\n{'k':>4} {'o':>4} {'energy':>10} {'ablated':>12} {'Δlog10':>8} {'topK':>5}")
    for r in rows:
        print(f"{r['k']:>4} {r['order']:>4} {r['energy']:>10.3f} "
              f"{r['ablated']:>12.4e} {r['delta_log']:>+8.3f} "
              f"{'Y' if r['in_top_K'] else '.':>5}")

    csv_path = out_dir / f"ablation_full_{tag}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            rr = dict(r); rr["in_top_K"] = "Y" if r["in_top_K"] else "N"
            w.writerow(rr)
    print(f"\nSaved {csv_path}")

    rows.sort(key=lambda r: r["k"])
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

    ax = axes[0]
    for r in rows:
        c = ORDER_COLOR.get(r["order"], "#aaaaaa")
        ax.scatter(r["energy"], r["delta_log"], color=c,
                   marker="*" if r["in_top_K"] else "o",
                   s=(14 if r["in_top_K"] else 7) ** 2, alpha=0.85,
                   edgecolor="black", linewidth=0.5)
        if r["delta_log"] > 1.5 or r["in_top_K"]:
            ax.annotate(f"k={r['k']}", (r["energy"], r["delta_log"]),
                        xytext=(4, 4), textcoords="offset points", fontsize=7)
    ax.set_xscale("log")
    ax.set_xlabel("W_E energy in character k")
    ax.set_ylabel("Δlog10(test loss) after ablating k")
    ax.set_title(f"(a) per-character ablation impact ({tag})")
    ax.grid(True, alpha=0.3)
    handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=c,
                          markersize=8, markeredgecolor="black", label=f"order {o}")
               for o, c in ORDER_COLOR.items()]
    handles.append(plt.Line2D([0], [0], marker="*", color="w", markerfacecolor="white",
                              markersize=12, markeredgecolor="black", label="top-K (5%)"))
    ax.legend(handles=handles, fontsize=7, loc="lower right", ncol=2)

    ax = axes[1]
    rows_sorted = sorted(rows, key=lambda r: r["delta_log"], reverse=True)
    xs = np.arange(len(rows_sorted))
    ax.bar(xs, [r["delta_log"] for r in rows_sorted],
           color=[ORDER_COLOR.get(r["order"], "#aaaaaa") for r in rows_sorted],
           edgecolor="black", linewidth=0.4)
    for i, r in enumerate(rows_sorted):
        if r["in_top_K"]:
            ax.text(i, r["delta_log"] + 0.1, f"k={r['k']}", fontsize=7,
                    ha="center", fontweight="bold")
    ax.axhline(np.log10(0.1) - np.log10(base), color="red", ls="--", lw=0.7,
               alpha=0.5, label="loss=0.1 (random model)")
    ax.set_xlabel("characters sorted by ablation impact")
    ax.set_ylabel("Δlog10(test loss) after ablating k")
    ax.set_title("(b) ranked ablation impact")
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(fontsize=8, loc="upper right")

    fig.suptitle(f"Full per-character ablation, baseline test loss = {base:.2e}", fontsize=11)
    fig.tight_layout()
    out = out_dir / f"ablation_full_{tag}.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
