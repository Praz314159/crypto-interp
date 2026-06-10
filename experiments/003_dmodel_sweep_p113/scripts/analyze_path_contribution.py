"""Path-decomposition at convergence: which paths through the residual block
carry the character-product contributions in the trained model's logits?

For the 1-layer transformer:
  resid_pre  = embed(x)
  resid_mid  = resid_pre + attn(resid_pre)
  resid_post = resid_mid + mlp(resid_mid)
  logits     = unembed(resid_post)

We split the forward pass into four paths by selectively detaching attn_out and
mlp_out (so they contribute as constants without backprop through them — but
for forward-only analysis we instead REPLACE them with their mean-over-inputs
value to remove their input-dependent signal, or simply ZERO them).

  full     = A + B + C + D    (all paths active)
  no_attn  = A + B            (zero out attn_out: skip + MLP-only path)
  no_mlp   = A + C            (zero out mlp_out:  skip + attn-only path)
  bare     = A                (zero both: pure linear skip path)

For each configuration, compute:
  - Test loss
  - Logit spectrum on c-axis projected onto multiplicative basis
  - Per-character contribution magnitude

Outputs:
  data/basis_dynamics/path_contribution_<tag>.csv
  figures/basis_dynamics/path_contribution_<tag>.png

Usage:
    python experiments/003_dmodel_sweep_p113/scripts/analyze_path_contribution.py \
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
    return basis.double(), char_idx


def forward_with_path_ablation(model, inputs, zero_attn, zero_mlp):
    """Run the 1-layer transformer with optional zeroing of attn_out and mlp_out."""
    block = model.blocks[0]
    e = model.embed(inputs)
    e = model.pos_embed(e)
    attn_out = block.attn(e)
    if zero_attn:
        attn_out = torch.zeros_like(attn_out)
    resid_mid = e + attn_out
    mlp_out = block.mlp(resid_mid)
    if zero_mlp:
        mlp_out = torch.zeros_like(mlp_out)
    resid_post = resid_mid + mlp_out
    return model.unembed(resid_post)


def char_spectrum_logits(logits, basis, char_idx, p, n_ans):
    """logits: (N, n_ans). Project on c-axis (over the answer-token axis,
    restricted to a ∈ 1..p-1) onto multiplicative-Fourier basis.
    Returns per-character mean-squared-energy across inputs."""
    L = logits[:, 1:p].double()                       # (N, p-1)
    basis_v = basis[:, 1:p].double()                  # (n_basis, p-1)
    coef = (L @ basis_v.T)                            # (N, n_basis)
    E_per_basis = (coef ** 2).mean(dim=0).cpu().numpy()  # (n_basis,)
    out = np.zeros(56)
    for k, rs in char_idx.items():
        out[k - 1] = float(E_per_basis[rs].sum())
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

    test_in = ds.inputs[ds.test_mask.bool()]
    test_lab = ds.labels[ds.test_mask.bool()]

    print(f"\nPath-decomposition at convergence (test loss + char spectrum):")
    print(f"{'path':>10}  {'test loss':>14}  {'logit char-spectrum top-5':>40}")
    results = {}
    for tag, za, zm in [
        ("full",    False, False),
        ("no_attn", True,  False),
        ("no_mlp",  False, True),
        ("bare",    True,  True),
    ]:
        with torch.no_grad():
            logits = forward_with_path_ablation(model, test_in, za, zm)
        last = logits[:, -1, :ds.n_answer_tokens]
        loss = float(cross_entropy_high_precision(last, test_lab, True).item())
        # Char spectrum on all (a, b) inputs at the answer position.
        inputs_all = ds.inputs                          # all 12769 (a,b,=) triples
        with torch.no_grad():
            logits_all = forward_with_path_ablation(model, inputs_all, za, zm)
        spec = char_spectrum_logits(logits_all[:, -1, :], basis, char_idx,
                                     p, ds.n_answer_tokens)
        results[tag] = dict(loss=loss, spec=spec)
        top5 = sorted(range(56), key=lambda i: -spec[i])[:5]
        print(f"{tag:>10}  {loss:>14.4e}  "
              f"{', '.join(f'k={k+1}:{spec[k]:.1e}' for k in top5)}")

    # Now also compute the DIFFERENCE spectra: how much does removing each path
    # change the logit content at each character?
    # Δ(no_attn → full) = full - no_attn  ≈ attention contribution
    # Δ(no_mlp → full) = full - no_mlp    ≈ MLP contribution
    contrib_attn = results["full"]["spec"] - results["no_attn"]["spec"]
    contrib_mlp  = results["full"]["spec"] - results["no_mlp"]["spec"]
    contrib_skip = results["bare"]["spec"]

    print(f"\nPer-character logit-spectrum contribution by path:")
    print(f"{'k':>4} {'o':>4}  "
          f"{'skip (bare)':>14}  {'+attn':>14}  {'+mlp':>14}  "
          f"{'fraction':>10}")
    for k in (3, 10, 20, 51, 6, 30, 13, 41):
        a = contrib_attn[k - 1]
        m = contrib_mlp[k - 1]
        s = contrib_skip[k - 1]
        total = abs(a) + abs(m) + abs(s)
        frac_str = (f"attn={100*abs(a)/total:.0f}% "
                    f"mlp={100*abs(m)/total:.0f}% "
                    f"skip={100*abs(s)/total:.0f}%") if total > 1e-12 else "n/a"
        print(f"{k:>4} {order_of(k):>4}  "
              f"{s:>14.3e}  {a:>14.3e}  {m:>14.3e}  {frac_str}")

    # Plot.
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    ax = axes[0]
    xs = np.arange(1, 57)
    ax.bar(xs, results["bare"]["spec"], color="#9467bd", alpha=0.6, label="skip (bare)")
    ax.bar(xs, contrib_attn, color="#2ca02c", alpha=0.6,
           bottom=results["bare"]["spec"], label="attn (full - no_attn)")
    ax.bar(xs, contrib_mlp, color="#d62728", alpha=0.6,
           bottom=results["bare"]["spec"] + contrib_attn,
           label="mlp (full - no_mlp)")
    ax.set_yscale("log")
    ax.set_xlabel("character k")
    ax.set_ylabel("logit-spectrum energy")
    ax.set_title("(a) per-character logit contribution by path")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    ax = axes[1]
    losses = [results[t]["loss"] for t in ("full", "no_attn", "no_mlp", "bare")]
    labels = ["full", "no_attn (A+B)", "no_mlp (A+C)", "bare (A)"]
    bars = ax.bar(labels, losses, color=["#1f77b4", "#d62728", "#2ca02c", "#9467bd"])
    ax.set_yscale("log")
    ax.set_ylabel("test loss")
    ax.set_title("(b) test loss under path ablation")
    for b, v in zip(bars, losses):
        ax.text(b.get_x() + b.get_width() / 2, v * 1.1,
                f"{v:.2e}", ha="center", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle(f"Path contribution at convergence ({args.tag})", fontsize=11)
    fig.tight_layout()
    out = FIG_DIR / f"path_contribution_{args.tag}.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
