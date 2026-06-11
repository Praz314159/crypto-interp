"""Per-token embedding geometry vs per-token errors, across populations.

For every run, two views of every input token x:

  Errors — error rate of x in the A and B positions (from the logits grid),
      and x's minimum margin over all pairs containing it (a subclinical
      defect measure that exists even when a run makes no errors).

  Geometry — for each essential character k, project W_E onto the χ_k
      plane and compare every token's measured angle to the ideal clock map
      θ_k(x) = σ·2πk·dlog(x)/(p−1) + φ_k (orientation σ and global phase
      φ_k fitted on all tokens). Report each token's worst angle residual
      across k — in "notches" (multiples of the token spacing 2π/(p−1)) —
      and its worst relative radius deviation.

The question this answers: are the tokens that carry a noisy seed's errors
geometric outliers *inside* the character planes (mis-tuned clock hands),
and are healthy seeds free of such outliers?

Usage:
    python -m crypto_interp.analysis.embedding_defect \\
        --out-dir outputs/embedding_defect/all
"""
from __future__ import annotations

import argparse
import csv
import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from crypto_interp.interp import Session
from crypto_interp.interp.bases import discrete_log_table
from crypto_interp.analysis.error_localization import analyze_run


PRIME_RUNS = {
    113: "experiments/003_dmodel_sweep_p113/runs/dmodel_24_dmlp_20_wd2_seed*",
    127: "experiments/004_p127/runs/dmodel_24_dmlp_32_seed[0-9]*",
    181: "experiments/005_p181/runs/dmodel_24_dmlp_32_seed[0-9]*",
}

TAU = 2 * np.pi


def wrap(a: np.ndarray) -> np.ndarray:
    """Wrap angles to (-pi, pi]."""
    return (a + np.pi) % TAU - np.pi


def clock_residuals(S: Session, K: list[int]) -> dict:
    """Per-token deviation from the run's own realized clock geometry.

    Fit the embedding as W_E[:, x] ≈ Σ_k C_k cos θ_k(x) + S_k sin θ_k(x)
    by least squares over tokens (θ_k(x) = 2πk·dlog(x)/n). This is the
    run's *own* ellipse on each character plane, so a perfect clock scores
    zero regardless of eccentricity. Then measure, per token and per k:

      angle_resid — in-plane angle between the token's actual and ideal
          positions on the (C_k, S_k) plane, radians
      amp_dev     — relative in-plane amplitude deviation
      leak        — out-of-K residual fraction ||W_E - fit|| / ||W_E||
          projected off every plane (the previously falsified quantity,
          kept for comparison)
    """
    p = S.ds.p
    n = p - 1
    _, dlog_map = discrete_log_table(p)
    dlog = np.array([dlog_map[x] for x in range(1, p)])  # token x = 1..p-1

    W = S.model.embed.W_E.detach().double().numpy()[:, 1:p]  # (d_model, n)
    theta = TAU * np.outer(K, dlog) / n                       # (|K|, n)
    Phi = np.concatenate([np.cos(theta), np.sin(theta)], axis=0)  # (2|K|, n)
    B, *_ = np.linalg.lstsq(Phi.T, W.T, rcond=None)           # (2|K|, d_model)
    ideal = B.T @ Phi                                          # (d_model, n)

    # Per-plane deviation distance (for attributing a defect to a clock).
    plane_dev = np.zeros((len(K), n))
    for j, k in enumerate(K):
        C, Sv = B[j], B[len(K) + j]                # the realized plane of χ_k
        e1 = C / (np.linalg.norm(C) + 1e-12)
        s = Sv - (Sv @ e1) * e1
        e2 = s / (np.linalg.norm(s) + 1e-12)
        P = np.stack([e1, e2])                     # (2, d_model)
        act = P @ W
        idl = P @ (np.outer(C, np.cos(theta[j])) + np.outer(Sv, np.sin(theta[j])))
        plane_dev[j] = np.linalg.norm(act - idl, axis=0)

    # Total in-subspace deviation, signal-normalized: how far the token sits
    # from its ideal position *within* the span of the K planes. Plane
    # weighting is automatic (strong planes contribute proportionally).
    Q, _ = np.linalg.qr(B.T)                       # (d_model, 2|K|) orthonormal
    resid = W - ideal
    in_dev = (np.linalg.norm(Q.T @ resid, axis=0)
              / (np.linalg.norm(ideal, axis=0) + 1e-12))
    # Out-of-K residual fraction (the previously falsified leakage measure).
    leak = (np.linalg.norm(resid - Q @ (Q.T @ resid), axis=0)
            / (np.linalg.norm(W, axis=0) + 1e-12))
    return {"in_dev": in_dev, "plane_dev": plane_dev, "leak": leak}


def analyze_population(pattern: str, p_expect: int) -> list[dict]:
    rows = []
    for d in sorted(glob.glob(pattern)):
        run = Path(d)
        if not run.is_dir():
            continue
        try:
            S = Session.from_run(str(run))
            err = analyze_run(run)
            K = sorted(int(k) for k in S.essential()["K"])
            geo = clock_residuals(S, K)
        except Exception as e:
            print(f"skip {run.name}: {e}")
            continue
        p = S.ds.p
        # per-token error rates and min margin
        margin = err["margin"]
        min_margin = np.minimum(margin.min(axis=1), margin.min(axis=0))
        worst_idx = geo["plane_dev"].argmax(axis=0)
        regime = ("grokked" if err["n_wrong"] == 0 else
                  "noisy" if err["accuracy"] > 0.9 else "failed")
        for x in range(1, p):
            i = x - 1
            rows.append({
                "p": p, "run": run.name, "regime": regime,
                "n_wrong_run": err["n_wrong"], "K": " ".join(map(str, K)),
                "token": x,
                "err_a": float(err["err_by_a"][i]),
                "err_b": float(err["err_by_b"][i]),
                "min_margin": float(min_margin[i]),
                "in_dev": float(geo["in_dev"][i]),
                "worst_k": int(K[worst_idx[i]]),
                "leak": float(geo["leak"][i]),
            })
        err_rate = np.maximum(err["err_by_a"], err["err_by_b"])
        med_dev = float(np.median(geo["in_dev"]))
        # rank correlation between in-subspace deviation and error rate
        def ranks(v):
            r = np.empty_like(v); r[np.argsort(v)] = np.arange(len(v)); return r
        rho = float(np.corrcoef(ranks(geo["in_dev"]), ranks(err_rate))[0, 1]) \
            if err_rate.std() > 0 else float("nan")
        bad = [r for r in rows if r["run"] == run.name
               and max(r["err_a"], r["err_b"]) > 0.01]
        msg = (f"  defect tokens: "
               + ", ".join(f"{r['token']} (err {max(r['err_a'], r['err_b']):.1%}, "
                           f"dev {r['in_dev'] / med_dev:.1f}× median @ χ_{r['worst_k']})"
                           for r in sorted(bad, key=lambda r: -max(r["err_a"], r["err_b"]))[:6])
               if bad else "  no errors")
        print(f"[p={p}] {run.name}: {regime}, wrong={err['n_wrong']}, "
              f"median in_dev {med_dev:.4f}, spearman(dev, err) = {rho:.2f}\n{msg}")
    return rows


def plot_scatter(rows: list[dict], out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    noisy = [r for r in rows if r["regime"] == "noisy"]
    grok = [r for r in rows if r["regime"] == "grokked"]

    ax = axes[0]
    for sel, color, lab, alpha, s in [
            (grok, "#9ca3af", "tokens of grokked runs", 0.25, 8),
            ([r for r in noisy if max(r["err_a"], r["err_b"]) <= 0.01],
             "#2563eb", "noisy runs — clean tokens", 0.35, 10),
            ([r for r in noisy if max(r["err_a"], r["err_b"]) > 0.01],
             "#dc2626", "noisy runs — defect tokens", 0.9, 26)]:
        if not sel:
            continue
        ax.scatter([r["in_dev"] for r in sel],
                   [max(r["err_a"], r["err_b"]) for r in sel],
                   c=color, alpha=alpha, s=s, label=lab, edgecolors="none")
    ax.set_xlabel("in-subspace deviation from fitted clock geometry")
    ax.set_ylabel("token error rate (max of A, B)")
    ax.set_yscale("symlog", linthresh=1e-3)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    ax.set_title("Is the defect a bent clock hand?")

    ax = axes[1]
    bins = np.linspace(0, max(r["in_dev"] for r in rows), 60)
    for sel, color, lab in [
            (grok, "#9ca3af", "grokked runs"),
            ([r for r in noisy if max(r["err_a"], r["err_b"]) <= 0.01],
             "#2563eb", "noisy — clean tokens"),
            ([r for r in noisy if max(r["err_a"], r["err_b"]) > 0.01],
             "#dc2626", "noisy — defect tokens")]:
        if sel:
            ax.hist([r["in_dev"] for r in sel], bins=bins,
                    density=True, histtype="step", lw=1.8, color=color,
                    label=lab)
    ax.set_xlabel("in-subspace deviation from fitted clock geometry")
    ax.set_ylabel("density")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    ax.set_title("Residual distributions")
    fig.tight_layout()
    out = out_dir / "residual_vs_error.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"\nWrote {out}")


# ---------------------------------------------------------------------------
# Focused probes (per-run): plane zoom and causal geometric repair
# ---------------------------------------------------------------------------

def _setup_run(run_dir: str):
    """Shared scaffold: session, errors, K, geometry fit, defect tokens."""
    import torch
    from collections import Counter
    from crypto_interp.interp.grids import compute_logits_grid

    S = Session.from_run(run_dir)
    p = S.ds.p
    n = p - 1
    _, dl = discrete_log_table(p)
    dlog = np.array([dl[x] for x in range(1, p)])
    K = sorted(int(k) for k in S.essential()["K"])
    W = S.model.embed.W_E.detach().double().numpy()[:, 1:p]
    theta = TAU * np.outer(K, dlog) / n
    Phi = np.concatenate([np.cos(theta), np.sin(theta)], 0)
    B, *_ = np.linalg.lstsq(Phi.T, W.T, rcond=None)

    def grid_wrong():
        lg = compute_logits_grid(S.model, S.ds).detach().cpu().numpy()[..., :p]
        a = np.arange(1, p)[:, None]
        b = np.arange(1, p)[None, :]
        wrong = lg.argmax(-1) != (a * b) % p
        return int(wrong.sum()), np.maximum(wrong.mean(1), wrong.mean(0))

    base_wrong, err = grid_wrong()
    defect = [x for x in range(1, p) if err[x - 1] > 0.01]
    # per-plane deviation z-scores and the modal implicated character
    pd = np.zeros((len(K), n))
    planes = []
    for j, k in enumerate(K):
        C, Sv = B[j], B[len(K) + j]
        Q, _ = np.linalg.qr(np.stack([C, Sv], 1))
        act = Q.T @ W
        idl = Q.T @ (np.outer(C, np.cos(theta[j])) + np.outer(Sv, np.sin(theta[j])))
        pd[j] = np.linalg.norm(act - idl, axis=0)
        planes.append((C, Sv, Q))
    z = pd / (np.median(pd, 1, keepdims=True) + 1e-12)
    kstar = (K[Counter(int(z[:, x - 1].argmax()) for x in defect).most_common(1)[0][0]]
             if defect else None)
    return dict(S=S, p=p, n=n, dlog_map=dl, K=K, B=B, theta=theta,
                planes=planes, z=z, kstar=kstar, defect=defect,
                base_wrong=base_wrong, grid_wrong=grid_wrong, torch=torch)


def zoom(run_dirs: list[str]) -> None:
    """Per-plane z-scores of defect tokens on the implicated clock, and the
    circular clustering R of their angles there (R→1: one arc)."""
    for rd in run_dirs:
        c = _setup_run(rd)
        if not c["defect"]:
            print(f"{Path(rd).name}: no errors")
            continue
        j = c["K"].index(c["kstar"])
        zdef = sorted((float(c["z"][j, x - 1]) for x in c["defect"]), reverse=True)
        ang = [TAU * c["kstar"] * c["dlog_map"][x] / c["n"] for x in c["defect"]]
        R = abs(np.mean(np.exp(1j * np.array(ang))))
        m = len(c["defect"])
        print(f"{Path(rd).name} (p={c['p']}): k*={c['kstar']}, {m} defect tokens, "
              f"wrong={c['base_wrong']}")
        print(f"  plane-z on k*: {['%.1f' % v for v in zdef]}")
        print(f"  angle clustering R={R:.2f} (null≈{0.886 / np.sqrt(m):.2f})")


def repair(run_dirs: list[str], seed: int = 0) -> None:
    """Causal test: replace tokens' in-plane coordinates on the implicated
    clock with their fitted ideal positions; recount errors. Controls: the
    same operation on random clean tokens, and on the entire clock."""
    rng = np.random.default_rng(seed)
    for rd in run_dirs:
        c = _setup_run(rd)
        if not c["defect"]:
            print(f"{Path(rd).name}: no errors")
            continue
        S, torch = c["S"], c["torch"]
        j = c["K"].index(c["kstar"])
        C, Sv, Q = c["planes"][j]

        def patched(tokens):
            WE = S.model.embed.W_E.detach().double().numpy().copy()
            for x in tokens:
                w = WE[:, x]
                ideal = (C * np.cos(c["theta"][j, x - 1])
                         + Sv * np.sin(c["theta"][j, x - 1]))
                WE[:, x] = w - Q @ (Q.T @ w) + Q @ (Q.T @ ideal)
            return torch.tensor(WE, dtype=S.model.embed.W_E.dtype)

        orig = S.model.embed.W_E.data.clone()
        out = {}
        clean = [x for x in range(1, c["p"]) if x not in c["defect"]]
        variants = {
            "repair defect tokens": c["defect"],
            "patch random clean": list(rng.choice(clean, size=len(c["defect"]),
                                                  replace=False)),
            "straighten whole clock": list(range(1, c["p"])),
        }
        for lab, toks in variants.items():
            S.model.embed.W_E.data = patched(toks)
            out[lab], _ = c["grid_wrong"]()
            S.model.embed.W_E.data = orig.clone()
        print(f"{Path(rd).name}: k*=χ_{c['kstar']}, {len(c['defect'])} defect "
              f"tokens, baseline wrong={c['base_wrong']}")
        for lab, w in out.items():
            print(f"    {lab:>24s}: wrong {c['base_wrong']} -> {w}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["census", "zoom", "repair"],
                    default="census")
    ap.add_argument("--run-dirs", nargs="*", default=None,
                    help="Run dirs for zoom/repair modes.")
    ap.add_argument("--out-dir", default="outputs/embedding_defect/all")
    args = ap.parse_args()

    if args.mode == "zoom":
        zoom(args.run_dirs or [])
        return
    if args.mode == "repair":
        repair(args.run_dirs or [])
        return

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for p, pattern in PRIME_RUNS.items():
        rows.extend(analyze_population(pattern, p))

    csv_path = out_dir / "census.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {csv_path} ({len(rows)} token rows)")

    plot_scatter(rows, out_dir)


if __name__ == "__main__":
    main()
