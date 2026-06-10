"""Test the gating hypothesis: does the attention pattern depend on the
piggybacker characters (k=20, k=6) of the input tokens?

The 1-layer transformer's attention at the "=" position assigns weights to
positions 0 (a), 1 (b), 2 (=). The query at "=" depends only on W_E[eq_token]
+ W_pos[2] — constant across (a, b). The keys at positions 0 and 1 depend on
W_E[a] and W_E[b]. So attn[h, =, a-pos] = softmax(Q_= · K[a])_a is a function
of a only (per head); similarly attn[h, =, b-pos] is a function of b only.

For each head and each input position (a, b), we collect the attention weight
to position 0 (call it w_a(a)) and to position 1 (w_b(b)), then project these
onto the multiplicative-Fourier basis. Significant character-k content in w_a
or w_b means the attention is using χ_k(input) as a gating feature.

Outputs:
  figures/basis_dynamics/attn_gating_<tag>.png  (per-head attn spectra)
  data/basis_dynamics/attn_gating_<tag>.csv

Usage:
    python experiments/003_dmodel_sweep_p113/scripts/analyze_attn_gating.py \
        --run-dir experiments/003_dmodel_sweep_p113/runs/dmodel_24_dmlp_32_seed1 \
        --tag dmlp32_seed1
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

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures" / "basis_dynamics"
OUT_DIR = ROOT / "data" / "basis_dynamics"


def order_of(k, n=112):
    return n // math.gcd(k, n)


ORDER_COLOR = {
    2: "#9467bd", 4: "#8c564b", 7: "#bcbd22", 8: "#17becf",
    14: "#e377c2", 16: "#1f77b4", 28: "#2ca02c", 56: "#ff7f0e", 112: "#d62728",
}


def build_basis_indexed():
    basis, names, _ = multiplicative_fourier_basis(113)
    char_idx = {}
    for i, nm in enumerate(names):
        m = re.match(r"mul (cos|sin) (\d+)", nm)
        if m:
            kk = int(m.group(2))
            char_idx.setdefault(kk, []).append(i)
    return basis.double(), char_idx


def compute_attn_pattern(model, ds):
    """Return attention pattern at the '=' position over the full (a, b) grid.
    Shape: (n_heads, p-1, p-1, 3) where last axis is key-position
    (0=a, 1=b, 2==)."""
    p = ds.p
    aa, bb = torch.meshgrid(torch.arange(1, p), torch.arange(1, p), indexing="ij")
    eq = torch.full_like(aa, ds.eq_token)
    inputs = torch.stack([aa, bb, eq], dim=-1).reshape(-1, 3)
    cache = {}
    model.remove_all_hooks()
    model.cache_all(cache)
    with torch.no_grad():
        _ = model(inputs)
    model.remove_all_hooks()
    # hook_attn: shape (batch, n_heads, query_pos, key_pos)
    attn = cache["blocks.0.attn.hook_attn"].double()
    # query pos = 2 (= token), all key positions
    attn_eq = attn[:, :, 2, :]                        # (N, n_heads, 3)
    n_heads = attn_eq.shape[1]
    return attn_eq.reshape(p - 1, p - 1, n_heads, 3).permute(2, 0, 1, 3)


def project_char_spectrum_1d(signal_per_token, basis, char_idx, p=113):
    """signal_per_token: (p-1,) array indexed by a ∈ {1, ..., p-1}.
    Return per-character energy: list length 56."""
    basis_v = basis[:, 1:p].double()
    coef = basis_v @ torch.tensor(signal_per_token, dtype=torch.float64)
    E = (coef ** 2).cpu().numpy()
    out = np.zeros(56)
    for k, rs in char_idx.items():
        out[k - 1] = float(E[rs].sum())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    run_dir = Path(args.run_dir).resolve()
    ck = sorted(run_dir.glob("checkpoint_*.pt"))
    model, ds, _ = load_run(ck[-1])
    model.eval()
    basis, char_idx = build_basis_indexed()
    p = ds.p

    attn_pattern = compute_attn_pattern(model, ds)        # (heads, p-1, p-1, 3)
    n_heads = attn_pattern.shape[0]
    print(f"Attention pattern shape: {tuple(attn_pattern.shape)}")
    print(f"({n_heads} heads, {p-1}×{p-1} input grid, 3 key positions)")

    # Sanity: attention weights should sum to 1 across key positions.
    sums = attn_pattern.sum(dim=-1)
    print(f"Sum check: min/max sum = {sums.min().item():.4f} / {sums.max().item():.4f}")

    # The weight to position a (key=0) is a function of a only;
    # the weight to position b (key=1) is a function of b only.
    # We verify this empirically: w_a(a, b) should be ~constant in b for each a.
    print(f"\n{'head':>4}  {'std_b(w_a) / mean':>20}  {'std_a(w_b) / mean':>20}")
    for h in range(n_heads):
        w_a = attn_pattern[h, :, :, 0].cpu().numpy()  # (p-1, p-1), indexed (a, b)
        w_b = attn_pattern[h, :, :, 1].cpu().numpy()
        rel_std_a = (w_a.std(axis=1) / (w_a.mean(axis=1) + 1e-12)).mean()
        rel_std_b = (w_b.std(axis=0) / (w_b.mean(axis=0) + 1e-12)).mean()
        print(f"{h:>4}  {rel_std_a:>20.4f}  {rel_std_b:>20.4f}")
    # If small (< 0.1), each weight is essentially a function of one input.

    # Average across the "other" input to get w_a(a) and w_b(b) per head.
    w_a_per_head = attn_pattern[:, :, :, 0].mean(dim=2).cpu().numpy()  # (heads, p-1)
    w_b_per_head = attn_pattern[:, :, :, 1].mean(dim=1).cpu().numpy()  # (heads, p-1)

    rows = []
    for h in range(n_heads):
        for which, signal in [("w_a", w_a_per_head[h]), ("w_b", w_b_per_head[h])]:
            spec = project_char_spectrum_1d(signal, basis, char_idx, p=p)
            # Identify top characters by spectrum.
            top = sorted(range(56), key=lambda i: -spec[i])[:5]
            rows.append(dict(
                head=h, which=which,
                top1_k=top[0] + 1, top1_E=spec[top[0]],
                top2_k=top[1] + 1, top2_E=spec[top[1]],
                top3_k=top[2] + 1, top3_E=spec[top[2]],
                spec=spec,
            ))

    # Print summary.
    print(f"\nPer-head character spectrum of attention-to-(a) and attention-to-(b):")
    print(f"{'head':>4} {'which':>5}  "
          f"{'top1 (k, E)':>20} {'top2 (k, E)':>20} {'top3 (k, E)':>20}")
    for r in rows:
        print(f"{r['head']:>4} {r['which']:>5}  "
              f"k={r['top1_k']:<3} E={r['top1_E']:>10.3e}  "
              f"k={r['top2_k']:<3} E={r['top2_E']:>10.3e}  "
              f"k={r['top3_k']:<3} E={r['top3_E']:>10.3e}")

    # Plot per-head character spectra as bar chart.
    fig, axes = plt.subplots(n_heads, 2, figsize=(14, 3 * n_heads), sharex=True)
    for h in range(n_heads):
        for j, which in enumerate(("w_a", "w_b")):
            ax = axes[h, j]
            r = [r for r in rows if r["head"] == h and r["which"] == which][0]
            spec = r["spec"]
            colors = [ORDER_COLOR.get(order_of(k + 1), "#aaaaaa") for k in range(56)]
            ax.bar(np.arange(1, 57), spec, color=colors,
                   edgecolor="black", linewidth=0.3)
            for k in (3, 6, 10, 20, 51):
                if spec[k - 1] > 0.01 * max(spec):
                    ax.text(k, spec[k - 1] * 1.02, f"k={k}", ha="center", fontsize=7,
                            fontweight="bold")
            ax.set_title(f"head {h}, attn weight to {which[2:]}-position", fontsize=9)
            ax.set_yscale("log")
            ax.grid(True, alpha=0.3, axis="y")
            if h == n_heads - 1:
                ax.set_xlabel("character k")
    fig.suptitle(f"Attention-pattern character spectra ({args.tag})\n"
                 f"Energy in cos_k + sin_k components of attn-from-= weights",
                 fontsize=11)
    fig.tight_layout()
    out = FIG_DIR / f"attn_gating_{args.tag}.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {out}")

    # CSV.
    csv_path = OUT_DIR / f"attn_gating_{args.tag}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["head", "which", "k", "order", "E_in_attn"])
        for r in rows:
            for k in range(56):
                w.writerow([r["head"], r["which"], k + 1, order_of(k + 1),
                            f"{r['spec'][k]:.6e}"])
    print(f"Saved {csv_path}")


if __name__ == "__main__":
    main()
