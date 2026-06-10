"""Analyze the path-decomposed gradient.

For each step, we have four characterizations of the W_E gradient projected
onto the multiplicative-Fourier character basis:
  full     = A + B + C + D    (all paths)
  no_attn  = A + B            (skip + MLP, no attn contribution)
  no_mlp   = A + C            (skip + attn, no MLP contribution)
  bare     = A                (skip-only path)

For each, compute:
  recall@|K|(t)  = |top-|K|-by-char-energy ∩ K| / |K|

Plot recall vs step for each path-characterization, and per seed.

Also derive isolated contributions:
  mlp_contrib  = no_attn - bare = B (gradient component from MLP-skip path alone)
  attn_contrib = no_mlp  - bare = C
  cross_contrib = full - bare - mlp_contrib - attn_contrib  ≈ D

These are computed in tensor space first, then projected.  (Aborted note:
the data we saved is energy-per-char, not the full tensor, so we can only
compare the four characterizations' alignment trajectories — not subtract
them in energy. That's OK; the four overlapping characterizations carry the
same information.)

Produces:
  figures/basis_dynamics/grad_decomp_summary.png
  figures/basis_dynamics/grad_decomp_seed<NN>.png
"""
from __future__ import annotations

import pickle
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
IN_DIR = ROOT / "data" / "grad_decomp"
TRAJ_FILE = ROOT / "data" / "basis_dynamics" / "trajectories.pkl"
FIG_DIR = ROOT / "figures" / "basis_dynamics"


def topk_recall(X_t, K_set, k=None):
    if k is None:
        k = len(K_set)
    top = set((np.argsort(X_t)[-k:] + 1).tolist())
    return len(top & K_set) / max(1, len(K_set))


def first_step_threshold(traj, thresh):
    above = np.where(traj >= thresh)[0]
    return int(above[0]) if len(above) else None


def main():
    with open(TRAJ_FILE, "rb") as f:
        trajectories = pickle.load(f)
    fg_files = sorted(IN_DIR.glob("seed*_grad_decomp.pt"))
    print(f"Found {len(fg_files)} files")

    all_traj = {"full": [], "no_attn": [], "no_mlp": [], "bare": []}
    rows = []
    for fg in fg_files:
        seed = int(re.match(r"seed(\d+)", fg.name).group(1))
        d = torch.load(fg, weights_only=False)
        traj = trajectories.get(seed)
        if traj is None:
            continue
        final_ce = traj["char_energy"][-1]
        Kset = set(int(k + 1) for k, e in enumerate(final_ce)
                   if e >= 0.05 * final_ce.max())

        T = d["char_G_full"].shape[0]
        rec = {}
        for key in ("full", "no_attn", "no_mlp", "bare"):
            X = d[f"char_G_{key}"].numpy()
            rec[key] = np.array([topk_recall(X[t], Kset) for t in range(T)])
            all_traj[key].append(rec[key])

        rows.append(dict(
            seed=seed, K_size=len(Kset),
            full50=first_step_threshold(rec["full"], 0.5),
            no_attn50=first_step_threshold(rec["no_attn"], 0.5),
            no_mlp50=first_step_threshold(rec["no_mlp"], 0.5),
            bare50=first_step_threshold(rec["bare"], 0.5),
            full80=first_step_threshold(rec["full"], 0.8),
            no_attn80=first_step_threshold(rec["no_attn"], 0.8),
            no_mlp80=first_step_threshold(rec["no_mlp"], 0.8),
            bare80=first_step_threshold(rec["bare"], 0.8),
            final_full=float(rec["full"][-1]),
            final_no_attn=float(rec["no_attn"][-1]),
            final_no_mlp=float(rec["no_mlp"][-1]),
            final_bare=float(rec["bare"][-1]),
        ))

        # Per-seed plot.
        fig, ax = plt.subplots(figsize=(10, 4.8))
        steps = np.arange(T)
        for key, label, color in [
            ("full",    "full (A+B+C+D)",  "#1f77b4"),
            ("no_attn", "no attn (A+B)",   "#2ca02c"),
            ("no_mlp",  "no MLP  (A+C)",   "#d62728"),
            ("bare",    "bare    (A)",     "#9467bd"),
        ]:
            ax.plot(steps, rec[key], label=label, color=color, lw=1.4, alpha=0.9)
        ax.axhline(0.5, color="black", ls=":", lw=0.5, alpha=0.5)
        ax.axhline(0.8, color="black", ls=":", lw=0.5, alpha=0.5)
        ax.set_xlabel("step")
        ax.set_ylabel("top-|K| recall")
        ax.set_title(f"seed {seed}: W_E-gradient K-alignment by path decomposition")
        ax.legend(fontsize=9, loc="lower right")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.02)
        fig.tight_layout()
        out = FIG_DIR / f"grad_decomp_seed{seed:02d}.png"
        fig.savefig(out, dpi=130, bbox_inches="tight")
        plt.close(fig)

    # Aggregate.
    fig, ax = plt.subplots(figsize=(11, 5.5))
    T = len(all_traj["full"][0])
    steps = np.arange(T)
    for key, label, color in [
        ("full",    "full (A+B+C+D)",  "#1f77b4"),
        ("no_attn", "no attn (A+B)",   "#2ca02c"),
        ("no_mlp",  "no MLP  (A+C)",   "#d62728"),
        ("bare",    "bare    (A)",     "#9467bd"),
    ]:
        stacked = np.stack(all_traj[key])
        mean = stacked.mean(axis=0)
        lo = np.percentile(stacked, 25, axis=0)
        hi = np.percentile(stacked, 75, axis=0)
        ax.plot(steps, mean, label=label, color=color, lw=2.0)
        ax.fill_between(steps, lo, hi, color=color, alpha=0.18)
    ax.axhline(0.5, color="black", ls=":", lw=0.5, alpha=0.5)
    ax.axhline(0.8, color="black", ls=":", lw=0.5, alpha=0.5)
    ax.set_xlabel("step")
    ax.set_ylabel("top-|K| recall")
    ax.set_title(f"Mean across {len(all_traj['full'])} seeds. Shaded = IQR.\n"
                 "Path decomposition of W_E gradient K-alignment over [0, 200)")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.02)
    fig.tight_layout()
    out = FIG_DIR / "grad_decomp_summary.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")

    # Summary table.
    print()
    print(f"{'seed':>4} {'|K|':>3}  "
          f"{'full50':>7} {'noA50':>6} {'noM50':>6} {'bare50':>7}  "
          f"{'full80':>7} {'noA80':>6} {'noM80':>6} {'bare80':>7}  "
          f"{'rec.full':>8} {'rec.noA':>7} {'rec.noM':>7} {'rec.bare':>8}")
    def f(x): return f"{x:>6}" if x is not None else "     -"
    def f7(x): return f"{x:>7}" if x is not None else "      -"
    for r in rows:
        print(f"{r['seed']:>4} {r['K_size']:>3}  "
              f"{f7(r['full50'])} {f(r['no_attn50'])} {f(r['no_mlp50'])} {f7(r['bare50'])}  "
              f"{f7(r['full80'])} {f(r['no_attn80'])} {f(r['no_mlp80'])} {f7(r['bare80'])}  "
              f"{r['final_full']:>8.2f} {r['final_no_attn']:>7.2f} "
              f"{r['final_no_mlp']:>7.2f} {r['final_bare']:>8.2f}")

    # Medians.
    for thresh in (50, 80):
        print()
        for key in ("full", "no_attn", "no_mlp", "bare"):
            col = f"{key}{thresh}"
            vals = [r[col] for r in rows if r[col] is not None]
            mn = int(np.median(vals)) if vals else None
            print(f"  median step to {key} recall >= 0.{thresh//10}: "
                  f"{mn}  ({len(vals)}/{len(rows)})")


if __name__ == "__main__":
    main()
