"""Analyze the (d_mlp × seed) sweep runs.

For each run matching ``dmodel_24_dmlp_<M>_seed<S>``, identify K from the final
checkpoint's W_E (multiplicative-Fourier basis), report d_mlp_cost and primitive
count, and compare cliff time. Prime is inferred from the embedding width, so
this works for sweeps at any prime.

Outputs:
  data/basis_dynamics/dmlp_sweep_summary.csv
  figures/basis_dynamics/dmlp_K_summary.png
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from crypto_interp.interp import char_energy, char_index, find_cliff, order_of


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", required=True, help="Experiment runs/ directory to scan.")
    ap.add_argument("--out-dir", default=None,
                    help="Where to write csv/png (default: <runs-dir>/../figures/basis_dynamics).")
    args = ap.parse_args()
    runs_dir = Path(args.runs_dir).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else runs_dir.parent / "figures" / "basis_dynamics"
    out_dir.mkdir(parents=True, exist_ok=True)

    pat = re.compile(r"dmodel_24_dmlp_(\d+)_seed(\d+)$")
    runs = []
    for d in sorted(runs_dir.iterdir()):
        m = pat.match(d.name)
        if not m or not d.is_dir():
            continue
        ck = sorted(d.glob("checkpoint_*.pt"))
        if ck:
            runs.append((int(m.group(1)), int(m.group(2)), d, ck[-1]))
    # Baseline d_mlp=512 seeds for comparison.
    for s in (1, 2, 3):
        d = runs_dir / f"dmodel_24_seed{s}"
        ck = sorted(d.glob("checkpoint_*.pt"))
        if ck:
            runs.append((512, s, d, ck[-1]))
    runs.sort()

    basis = ci = p = None
    rows = []
    for d_mlp, seed, run_dir, last_ck in runs:
        state = torch.load(last_ck, weights_only=False, map_location="cpu")
        W_E = state["model_state"]["embed.W_E"].double()
        if p is None:
            p = W_E.shape[1] - 1               # vocab = p + 1
            basis, ci = char_index(p)
        char_E = char_energy(W_E[:, :p], basis, ci)
        K = sorted(int(k + 1) for k, e in enumerate(char_E) if e >= 0.05 * char_E.max())
        orders = sorted((order_of(k, p) for k in K), reverse=True)
        n_prim = sum(1 for o in orders if o == p - 1)
        d_mlp_cost = sum(orders)
        has_leg = 2 in orders

        cliff = None
        losses_path = run_dir / "losses.pt"
        if losses_path.exists():
            losses = torch.load(losses_path, weights_only=False)
            cliff = find_cliff(losses["test_losses"])

        rows.append(dict(d_mlp=d_mlp, seed=seed, K=K, K_size=len(K), orders=orders,
                         n_primitive=n_prim, d_mlp_cost=d_mlp_cost,
                         has_legendre=has_leg, cliff=cliff))

    print(f"{'d_mlp':>5} {'seed':>4}  {'cliff':>6} {'|K|':>3} {'cost':>5} "
          f"{'#prim':>5} {'Leg':>3}   orders                       K")
    for r in rows:
        cliff_s = "-" if r["cliff"] is None else str(r["cliff"])
        leg = "Y" if r["has_legendre"] else "."
        print(f"{r['d_mlp']:>5} {r['seed']:>4}  {cliff_s:>6} {r['K_size']:>3} "
              f"{r['d_mlp_cost']:>5} {r['n_primitive']:>5} {leg:>3}   "
              f"{str(r['orders']):<28}  {r['K']}")

    csv_path = out_dir / "dmlp_sweep_summary.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["d_mlp", "seed", "K", "K_size", "orders", "n_primitive",
                    "d_mlp_cost", "has_legendre", "cliff"])
        for r in rows:
            w.writerow([r["d_mlp"], r["seed"], "|".join(map(str, r["K"])), r["K_size"],
                        "|".join(map(str, r["orders"])), r["n_primitive"], r["d_mlp_cost"],
                        "Y" if r["has_legendre"] else "N",
                        r["cliff"] if r["cliff"] is not None else ""])
    print(f"\nSaved {csv_path}")

    dmlps = sorted(set(r["d_mlp"] for r in rows))
    by_d = {d: [r for r in rows if r["d_mlp"] == d] for d in dmlps}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, key, title, ylab in [
        (axes[0], "K_size", "|K| vs d_mlp budget", "|K|"),
        (axes[1], "d_mlp_cost", "actual d_mlp_cost vs budget", "Σ order(k)"),
        (axes[2], "n_primitive", "# primitive chars vs budget", "# order-(p-1) in K"),
    ]:
        xs = [d for d in dmlps for _ in by_d[d]]
        ys = [r[key] for d in dmlps for r in by_d[d]]
        ax.scatter(xs, ys, s=80, alpha=0.75, edgecolor="black", linewidth=0.4)
        ax.set_xscale("log", base=2)
        ax.set_xticks(dmlps); ax.set_xticklabels([str(d) for d in dmlps])
        ax.set_xlabel("d_mlp budget"); ax.set_ylabel(ylab); ax.set_title(title)
        ax.grid(True, alpha=0.3)
        if key == "d_mlp_cost":
            ax.plot(dmlps, dmlps, "k--", lw=0.7, alpha=0.5, label="cost = budget")
            ax.legend(fontsize=8)
    fig.suptitle(f"d_mlp sweep summary ({len(rows)} runs)", fontsize=11)
    fig.tight_layout()
    out = out_dir / "dmlp_K_summary.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
