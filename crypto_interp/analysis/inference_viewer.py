"""Visualize one forward pass of the learned mod-mul algorithm, stage by stage.

For a single input (a, b), render the computation in the multiplicative
character basis, where each stage of the algorithm is geometric:

  Row 1 — clocks: every token's embedding projected onto each essential
      character's 2D plane in W_E; the inputs a, b highlighted as hands at
      angles θ_k(a), θ_k(b).  Plus the attention pattern at the '=' position
      (a courier, not a calculator).
  Row 2 — phased arrays: the MLP neurons of each character's cluster placed
      at their preferred phase φ_i around a circle, bar length = actual
      activation on this forward pass.  The bump of active neurons points
      at θ_k(a) + θ_k(b) = θ_k(a·b).  Helper characters have no neurons of
      their own (they ride a primary's cluster) and are annotated as such.
  Row 3 — interference: the actual logits of this forward pass decomposed
      per character over candidates c in dlog order — each character's wave
      is individually ambiguous (coset-periodic); their sum spikes at the
      answer.  Model logits overlaid in gray.

Usage:
    python -m crypto_interp.analysis.inference_viewer \\
        --run-dir experiments/003_dmodel_sweep_p113/runs/dmodel_24_dmlp_20_wd2_seed1 \\
        --a 7 --b 12
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from crypto_interp.interp import Session
from crypto_interp.interp.bases import discrete_log_table


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def char_plane(S: Session, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Orthonormal 2D frame (e1, e2) of character k's plane in embedding space."""
    W_E = S.model.embed.W_E.detach().double()[:, : S.ds.p]  # (d_model, p)
    c = (W_E @ S.basis[S.ci.cos[k]]).numpy()
    e1 = c / (np.linalg.norm(c) + 1e-12)
    if k in S.ci.sin:
        s = (W_E @ S.basis[S.ci.sin[k]]).numpy()
    else:  # Nyquist character: plane degenerates to a line
        s = np.zeros_like(c)
    s = s - (s @ e1) * e1
    n = np.linalg.norm(s)
    e2 = s / n if n > 1e-9 else np.zeros_like(s)
    return e1, e2


def token_coords(S: Session, e1: np.ndarray, e2: np.ndarray) -> np.ndarray:
    """(p-1, 2) coordinates of tokens 1..p-1 in the (e1, e2) frame."""
    W_E = S.model.embed.W_E.detach().double()[:, 1 : S.ds.p].numpy()  # skip 0
    return np.stack([e1 @ W_E, e2 @ W_E], axis=1)


def neuron_phases(S: Session, k: int, cluster: np.ndarray,
                  e1: np.ndarray, e2: np.ndarray) -> np.ndarray:
    """Preferred phase φ_i of each cluster neuron, from its input weights."""
    W_in = S.model.blocks[0].mlp.W_in.detach().double().numpy()  # (d_mlp, d_model)
    return np.array([np.arctan2(W_in[i] @ e2, W_in[i] @ e1) for i in cluster])


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def make_figure(S: Session, a: int, b: int, K: list[int] | None = None):
    p = S.ds.p
    n = p - 1
    g, dlog = discrete_log_table(p)
    if a % p == 0 or b % p == 0:
        raise ValueError("a and b must be nonzero mod p (dlog undefined at 0)")
    if K is None:
        K = sorted(S.essential()["K"])
    helpers = {h: m for (h, m, _mult, _e) in S.helpers(K)}
    _, dom = S.per_neuron_dominant_char()

    theta = lambda k, x: 2 * np.pi * k * dlog[x % p] / n  # noqa: E731
    x_a, x_b = dlog[a % p], dlog[b % p]
    ans = (a * b) % p
    x_ans = (x_a + x_b) % n

    # Forward pass on the single input.
    eq = S.ds.eq_token
    inp = torch.tensor([[a % p, b % p, eq]], dtype=torch.long)
    logits, cache = S.run_with_cache(inp)
    L = logits[0, -1, :p].detach().double().numpy()
    attn = cache["blocks.0.attn.hook_attn"][0, :, -1, :].detach().numpy()  # (heads, 3)
    mlp_post = cache["mlp_post"][0, -1, :].detach().double().numpy()       # (d_mlp,)

    ncol = len(K) + 1
    fig = plt.figure(figsize=(3.1 * ncol, 9.6))
    gs = fig.add_gridspec(3, ncol, height_ratios=[1, 1, 1.1])

    # ---- Row 1: clocks + attention --------------------------------------
    for j, k in enumerate(K):
        ax = fig.add_subplot(gs[0, j])
        e1, e2 = char_plane(S, k)
        pts = token_coords(S, e1, e2)
        ax.scatter(pts[:, 0], pts[:, 1], s=6, c="lightgray", zorder=1)
        for tok, color, lab in ((a % p, "C0", "a"), (b % p, "C1", "b"),
                                (ans, "C2", "a·b")):
            x, y = pts[tok - 1]
            ax.annotate("", xy=(x, y), xytext=(0, 0),
                        arrowprops=dict(arrowstyle="->", color=color, lw=2))
            ax.annotate(f"{lab}={tok}", xy=(x, y), fontsize=8, color=color)
        role = f"helper of χ_{helpers[k]}" if k in helpers else "primary"
        ax.set_title(f"χ_{k} clock ({role})", fontsize=9)
        ax.set_aspect("equal")
        ax.axis("off")

    ax = fig.add_subplot(gs[0, ncol - 1])
    im = ax.imshow(attn, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(3), [f"a={a%p}", f"b={b%p}", "="])
    ax.set_yticks(range(attn.shape[0]), [f"head {h}" for h in range(attn.shape[0])])
    ax.set_title("attention from '='", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046)

    # ---- Row 2: cluster phased arrays ------------------------------------
    for j, k in enumerate(K):
        ax = fig.add_subplot(gs[1, j], projection="polar")
        cluster = np.where(dom == k)[0]
        target = (theta(k, a) + theta(k, b)) % (2 * np.pi)
        if len(cluster) == 0:
            ax.set_yticks([]); ax.set_xticks([])
            note = (f"no neurons:\nrides χ_{helpers[k]}'s cluster"
                    if k in helpers else "no neurons")
            ax.annotate(note, xy=(0.5, 0.5), xycoords="axes fraction",
                        ha="center", fontsize=8)
            ax.set_title(f"χ_{k} cluster (0 neurons)", fontsize=9)
            continue
        phis = neuron_phases(S, k, cluster, *char_plane(S, k))
        acts = mlp_post[cluster]
        amax = max(acts.max(), 1e-9)
        ax.bar(phis, acts, width=2 * np.pi / max(len(phis), 8) * 0.8,
               color="C0", alpha=0.75)
        # The bump should point at ±target (real basis can't fix the sign).
        for sgn, ls in ((1, "-"), (-1, "--")):
            ax.plot([sgn * target] * 2, [0, amax], color="C3",
                    linestyle=ls, lw=1.5)
        ax.set_title(f"χ_{k} cluster ({len(cluster)} neurons)\n"
                     f"red: ±θ_{k}(a)+θ_{k}(b)", fontsize=8)
        ax.set_yticks([]); ax.tick_params(labelsize=6)

    ax = fig.add_subplot(gs[1, ncol - 1])
    ax.axis("off")
    ax.annotate(
        f"p = {p}, g = {g}\n"
        f"dlog(a)={x_a}, dlog(b)={x_b}\n"
        f"dlog(ab)={x_ans}\n"
        f"answer a·b mod p = {ans}\n\n"
        f"K = {K}\n"
        f"helpers: {helpers or 'none'}",
        xy=(0.05, 0.5), xycoords="axes fraction", fontsize=9, va="center")

    # ---- Row 3: interference ---------------------------------------------
    # Decompose THIS forward pass's logits per character, in dlog order.
    order = np.argsort([dlog[c] for c in range(1, p)])  # tokens sorted by dlog
    toks_by_dlog = np.arange(1, p)[order]
    xs = np.arange(n)

    ax = fig.add_subplot(gs[2, :])
    basis_np = S.basis.numpy()
    total = np.zeros(n)
    offset = 0.0
    span = 2.2 * np.abs(L - L.mean()).max() / max(len(K), 1)
    for k in K:
        wave = basis_np[S.ci.cos[k], :p] * (basis_np[S.ci.cos[k], :p] @ L)
        if k in S.ci.sin:
            wave = wave + basis_np[S.ci.sin[k], :p] * (basis_np[S.ci.sin[k], :p] @ L)
        w = wave[toks_by_dlog]
        total += w
        ax.plot(xs, w + offset, lw=1.0, label=f"χ_{k}")
        offset += span
    ax.plot(xs, total + offset, color="black", lw=1.4, label="Σ over K")
    Lc = L[toks_by_dlog]
    ax.plot(xs, (Lc - Lc.mean()) + offset, color="gray", lw=0.8, alpha=0.6,
            label="actual logits")
    ax.axvline(x_ans, color="C2", lw=1.2, linestyle=":")
    ax.annotate(f"dlog(ab) = {x_ans}", xy=(x_ans, ax.get_ylim()[1]),
                fontsize=8, color="C2", ha="center", va="bottom")
    ax.set_xlabel("candidate c, ordered by dlog(c)")
    ax.set_yticks([])
    ax.legend(fontsize=8, ncol=len(K) + 2, loc="lower left")
    ax.set_title("per-character logit waves (each coset-ambiguous) → "
                 "constructive interference at c = a·b", fontsize=10)

    fig.suptitle(f"{a} × {b} ≡ {ans}  (mod {p})", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--a", type=int, required=True)
    ap.add_argument("--b", type=int, required=True)
    ap.add_argument("--ks", type=int, nargs="*", default=None,
                    help="Character set override (default: essential K).")
    ap.add_argument("--out-file", default=None)
    args = ap.parse_args()

    S = Session.from_run(args.run_dir)
    fig = make_figure(S, args.a, args.b, K=args.ks)

    out = args.out_file
    if out is None:
        run = Path(args.run_dir)
        exp = run.parent.parent.name
        out = Path("outputs/inference_viewer") / exp / f"{run.name}_a{args.a}_b{args.b}.png"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
