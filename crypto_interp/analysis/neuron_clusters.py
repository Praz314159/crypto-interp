"""Identify the cluster of MLP neurons specialized to each character k in K,
reconstruct the cluster's contribution to the residual stream as a function
of (a, b), and compare against the algebraic reference cos(θ_k(a)+θ_k(b)).

Four panels per character: cluster signal (natural + dlog-sorted) and
reference (natural + dlog-sorted).

Migrated to the v1 harness: ``per_neuron_dominant_char``, ``cluster_signal``
and ``reference_cos_signal`` now live in :mod:`crypto_interp.interp.neurons`
(so analyses depend on interp, not the other way around). This module is the
CLI wrapper that drives them via :class:`crypto_interp.interp.Session`.

Usage:
    python -m crypto_interp.analysis.neuron_clusters \\
        --run-dir experiments/003_dmodel_sweep_p113/runs/dmodel_24_dmlp_32_seed1
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from crypto_interp.interp import (
    Session,
    correlate,
    discrete_log_table,
    order_of,
)
# Re-export from interp for back-compat with anything still importing here:
from crypto_interp.interp.neurons import (  # noqa: F401
    per_neuron_dominant_char,
    cluster_signal,
    reference_cos_signal,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=str, required=True)
    ap.add_argument("--out-prefix", type=str, default="neuron_clusters")
    args = ap.parse_args()
    run_dir = Path(args.run_dir).resolve()

    S = Session.from_run(run_dir)
    print(f"Loading {run_dir.name}")
    p = S.ds.p

    # K via the 5%-energy threshold (matches the legacy behavior here).
    char_E_WE = S.char_energy()
    K = sorted(k for k in S.ci.freqs if char_E_WE[k - 1] >= 0.05 * char_E_WE.max())
    print(f"K = {K} (orders {[order_of(k, p) for k in K]}) — primitive root g={S.ci.g}")

    _, dom = S.per_neuron_dominant_char()
    _, dlog = discrete_log_table(p)
    order_idx = np.argsort([dlog[a] for a in range(1, p)])

    for k in K:
        cluster = np.where(dom == k)[0]
        if len(cluster) == 0:
            print(f"  k={k}: no neurons in cluster (skip)")
            continue
        print(f"  k={k} (order {order_of(k, p)}): {len(cluster)} neurons in cluster")

        sig = S.cluster_signal(k)
        ref = S.reference_signal(k)
        sig_sorted = sig[order_idx][:, order_idx]
        ref_sorted = ref[order_idx][:, order_idx]

        fig, axes = plt.subplots(2, 2, figsize=(11, 10))
        for ax, data, title in [
            (axes[0, 0], sig, f"cluster signal (k={k}), natural index"),
            (axes[0, 1], sig_sorted, "cluster signal, dlog-sorted"),
            (axes[1, 0], ref, "reference cos(θ_k(a)+θ_k(b)), natural"),
            (axes[1, 1], ref_sorted, "reference, dlog-sorted"),
        ]:
            vmax = max(abs(data.max()), abs(data.min()))
            ax.imshow(data, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="equal", origin="upper")
            ax.set_title(title, fontsize=9)
            ax.set_xlabel("b" + (" (dlog)" if "sorted" in title else ""))
            ax.set_ylabel("a" + (" (dlog)" if "sorted" in title else ""))
        corr = correlate(sig, ref)
        fig.suptitle(f"Neuron cluster for character k={k} (order {order_of(k, p)}), "
                     f"|cluster|={len(cluster)}; correlation w/ reference = {corr:+.3f}",
                     fontsize=11)
        fig.tight_layout()
        out = run_dir / f"{args.out_prefix}_k{k}.png"
        fig.savefig(out, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"    saved {out}  corr={corr:+.3f}")


if __name__ == "__main__":
    main()
