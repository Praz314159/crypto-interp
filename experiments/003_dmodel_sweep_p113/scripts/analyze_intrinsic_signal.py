"""Visualize the character-k product signal in the MLP output, using the
intrinsic d_model direction that the MLP actually writes to (rather than the
unembed's read direction).

For each (a, b), compute the MLP output f(a, b) ∈ R^d_model. The intrinsic
character-k direction is
    g_k[d] = Σ_{a,b} cos[θ_k(a) + θ_k(b)] · f(a, b)[d]
The "character-k signal" at (a, b) is then ⟨f(a,b), g_k / ||g_k||⟩.

We compare this signal to the algebraic reference cos[θ_k(a) + θ_k(b)],
both in natural and dlog-sorted views. Should give a high correlation by
construction IF the MLP is computing a character-k product at all (no matter
what direction).

Also report the alignment ⟨g_k / ||g_k||, W_U·cos_k / ||W_U·cos_k||⟩ — the
cosine angle between the MLP's write direction and the unembed's read direction.
If this is ~1, the model's output character-k product flows to the right
output channel. If small, the character-k product is being "relabeled" into a
different output channel.

Usage:
    python experiments/003_dmodel_sweep_p113/scripts/analyze_intrinsic_signal.py \
        --run-dir experiments/003_dmodel_sweep_p113/runs/dmodel_24_dmlp_32_seed1 \
        --ks 3,10,20,51,6,30
"""
from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from crypto_interp.interp.bases import (
    multiplicative_fourier_basis,
    discrete_log_table,
)
from crypto_interp.interp.load import load_run


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


def compute_mlp_output_grid(model, ds):
    p = ds.p
    a_grid = torch.arange(1, p)
    b_grid = torch.arange(1, p)
    aa, bb = torch.meshgrid(a_grid, b_grid, indexing="ij")
    eq = torch.full_like(aa, ds.eq_token)
    inputs = torch.stack([aa, bb, eq], dim=-1).reshape(-1, 3)
    cache = {}
    model.remove_all_hooks()
    model.cache_all(cache)
    with torch.no_grad():
        _ = model(inputs)
    model.remove_all_hooks()
    mlp_out = cache["blocks.0.hook_mlp_out"][:, -1, :].double()
    return mlp_out.reshape(p - 1, p - 1, -1)


def reference_cos(k, p=113):
    _, dlog = discrete_log_table(p)
    j_a = np.array([dlog[a] for a in range(1, p)])
    theta_a = 2 * np.pi * k * j_a / (p - 1)
    return np.cos(theta_a[:, None] + theta_a[None, :])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--ks", required=True)
    args = ap.parse_args()
    run_dir = Path(args.run_dir).resolve()
    ck = sorted(run_dir.glob("checkpoint_*.pt"))
    print(f"Loading {ck[-1]}")
    model, ds, _ = load_run(ck[-1])
    model.eval()
    basis, names, char_idx = build_basis_indexed()
    p = ds.p
    _, dlog = discrete_log_table(p)
    order_idx = np.argsort([dlog[a] for a in range(1, p)])

    f_ab = compute_mlp_output_grid(model, ds)        # (p-1, p-1, d_model)
    W_U = model.unembed.W_U.detach()[:, :p].double()  # (d_model, p)

    for k_str in args.ks.split(","):
        k = int(k_str)
        ref = reference_cos(k, p)
        ref_t = torch.tensor(ref, dtype=torch.float64)
        # g_k[d] = Σ_{a,b} ref[a,b] · f(a,b)[d]
        g_k = torch.einsum("ab,abd->d", ref_t, f_ab)
        norm = g_k.norm()
        if norm < 1e-10:
            print(f"  k={k}: ||g_k|| ≈ 0, skipping")
            continue
        g_unit = g_k / norm
        # Project MLP output at each (a, b) onto g_unit.
        sig = torch.einsum("abd,d->ab", f_ab, g_unit).cpu().numpy()
        # Reference correlation.
        a = sig.flatten() - sig.mean()
        b = ref.flatten() - ref.mean()
        corr = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
        # Alignment between intrinsic write direction and W_U read direction.
        # Build basis row for cos_k.
        cos_idx = next(i for i, nm in enumerate(names)
                       if re.match(rf"mul cos {k}\b", nm))
        cos_k_basis = basis[cos_idx].double()
        W_U_k = W_U @ cos_k_basis                   # (d_model,)
        if W_U_k.norm() > 1e-10:
            align = float((g_unit @ (W_U_k / W_U_k.norm())).item())
        else:
            align = float("nan")

        # Plot 4 panels.
        sig_sorted = sig[order_idx][:, order_idx]
        ref_sorted = ref[order_idx][:, order_idx]
        fig, axes = plt.subplots(2, 2, figsize=(11, 10))
        for ax, data, title in [
            (axes[0, 0], sig, f"intrinsic char-{k} signal, natural"),
            (axes[0, 1], sig_sorted, "intrinsic, dlog-sorted"),
            (axes[1, 0], ref, f"reference cos[θ_{k}(a)+θ_{k}(b)], natural"),
            (axes[1, 1], ref_sorted, "reference, dlog-sorted"),
        ]:
            vmax = max(abs(data.max()), abs(data.min()))
            ax.imshow(data, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                      aspect="equal", origin="upper")
            ax.set_title(title, fontsize=9)
            ax.set_xlabel("b" + (" (dlog)" if "sorted" in title else ""))
            ax.set_ylabel("a" + (" (dlog)" if "sorted" in title else ""))
        fig.suptitle(
            f"Intrinsic character-{k} signal (order {order_of(k)}): "
            f"corr w/ reference = {corr:+.3f}; "
            f"alignment with W_U·cos_{k} = {align:+.3f}",
            fontsize=11,
        )
        fig.tight_layout()
        out = run_dir / f"intrinsic_signal_k{k}.png"
        fig.savefig(out, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"  k={k:>3} (o={order_of(k)}): corr={corr:+.3f}  "
              f"alignment={align:+.3f}  saved {out.name}")


if __name__ == "__main__":
    main()
