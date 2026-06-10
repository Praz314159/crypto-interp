"""Grokking-dynamics detectors: cliff, bifurcation, commitment, and a
grokked/failed status summary. Prime-parametric (operate on trajectories).
"""
from __future__ import annotations

import numpy as np


def cliff_step(test_losses, thresh: float = 0.1) -> int | None:
    """First step where test loss drops below ``thresh`` (the grokking cliff)."""
    arr = np.asarray(test_losses, dtype=np.float64)
    idx = np.where(arr < thresh)[0]
    return int(idx[0]) if len(idx) else None


def bifurcation_step(char_E_traj, K_mask, ratio: float = 1.5) -> int | None:
    """First step where the K / non-K mean-energy ratio exceeds ``ratio`` times
    its initial value. ``char_E_traj`` is (T, n_chars); ``K_mask`` is a boolean
    (n_chars,) selecting the key characters."""
    E = np.asarray(char_E_traj, dtype=np.float64)
    K_mask = np.asarray(K_mask, dtype=bool)
    Km = E[:, K_mask].mean(axis=1)
    nKm = E[:, ~K_mask].mean(axis=1)
    r = Km / np.where(nKm > 0, nKm, 1e-30)
    r0 = r[0] if r[0] > 0 else 1e-30
    idx = np.where(r > ratio * r0)[0]
    return int(idx[0]) if len(idx) else None


def commit_step(char_E_traj, final_K, mode: str = "subset") -> int | None:
    """First step after which the top-|K| characters (by energy) stably contain
    (``mode="subset"``) or exactly equal (``mode="exact"``) ``final_K`` for the
    remainder of training. ``final_K`` is a list of 1-based character ids."""
    E = np.asarray(char_E_traj, dtype=np.float64)
    kK = len(final_K)
    Kset = set(int(x) for x in final_K)
    T = E.shape[0]
    cond = np.zeros(T, dtype=bool)
    for t in range(T):
        top = set((np.argsort(E[t])[::-1][:kK] + 1).tolist())
        cond[t] = Kset.issubset(top) if mode == "subset" else (top == Kset)
    for t in range(T):
        if cond[t:].all():
            return t
    return None


def grokking_status(train_losses, test_losses, *,
                    mem_thresh: float = 0.1, grok_thresh: float = 0.1) -> dict:
    """Summarize a run: when it memorized (train loss < ``mem_thresh``), when it
    grokked (test loss < ``grok_thresh``), whether it grokked at all, and the
    final test loss. ``cliff_step is None`` => the run did not grok."""
    tr = np.asarray(train_losses, dtype=np.float64)
    te = np.asarray(test_losses, dtype=np.float64)
    mem = np.where(tr < mem_thresh)[0]
    cliff = cliff_step(te, grok_thresh)
    return dict(
        memorized_step=int(mem[0]) if len(mem) else None,
        cliff_step=cliff,
        grokked=cliff is not None,
        final_test_loss=float(te[-1]),
    )
