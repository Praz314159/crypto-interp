"""Harmonic-helper detection via the Delta_k ablation pipeline.

For each key character k: ablate it from W_E, take the change in the logit grid,
reduce to a 1-D function of the discrete-log difference, and read its dominant
Fourier frequency. A *primary* destroys its own frequency (dominant == k); a
*helper* destroys a different (lower) frequency (dominant != k) -- the Sylow-2
doubling signature. ``find_primary_helper_pairs`` is the per-prime test of the
harmonic-helper finding. Prime-parametric.
"""
from __future__ import annotations

import numpy as np

from .ablate import ablate_character
from .bases import CharIndex, fold_frequency
from .grids import compute_logits_grid
from .reductions import fourier_spectrum_1d, reduce_to_diff


def helper_multiplier(helper: int, primary: int, p: int, max_mult: int = 8) -> int | None:
    """Smallest j>=2 such that chi_primary^j (= chi_{j*primary}) folds to the
    helper's real-basis index. j=2 is the Sylow-2 doubling (squared feature),
    j=3 a Sylow-3 rider, etc. Returns None if no small power matches.

    Fold-aware: compares ``fold_frequency(j*primary)`` to ``fold_frequency(helper)``
    so a doubling whose raw index 2*primary exceeds (p-1)/2 (and folds back) is
    still recognized as a doubling.
    """
    h = fold_frequency(helper, p)
    for j in range(2, max_mult + 1):
        if fold_frequency(j * primary, p) == h:
            return j
    return None


def delta_k(model, ds, ci: CharIndex, basis, k: int, *, logits_full=None) -> np.ndarray:
    """Exact contribution of character k to the logits:
    ``logit_full - logit_ablated``, shape (p-1, p-1, vocab) as a numpy array.
    Pass ``logits_full`` (a torch grid) to avoid recomputing it across calls."""
    from .interventions import ablate_char_w
    full = compute_logits_grid(model, ds) if logits_full is None else logits_full
    with ablate_char_w(model, k, basis, ci):
        ablated = compute_logits_grid(model, ds)
    return (full - ablated).cpu().numpy()


def delta_k_spectrum(model, ds, ci: CharIndex, basis, k: int, *, logits_full=None):
    """Return ``(f_diff, energy, dominant_freq)`` for character k: the 1-D
    Dlog reduction of Delta_k, its Fourier energy spectrum, and the dominant
    frequency m (1-based)."""
    grid = delta_k(model, ds, ci, basis, k, logits_full=logits_full)
    f_diff = reduce_to_diff(grid, ds.p, value_axis="full")
    energy = fourier_spectrum_1d(f_diff)
    dominant = int(energy.argmax() + 1)
    return f_diff, energy, dominant


def find_primary_helper_pairs(model, ds, ci: CharIndex, basis, K, max_mult: int = 8):
    """For each k in K, classify as primary (dominant freq == k) or helper
    (dominant != k). Returns ``(helper_k, primary_m, mult, energy)`` for each
    helper, where ``mult`` is the fold-aware power relation (2 = Sylow-2 doubling,
    3 = Sylow-3, None = no small power) from :func:`helper_multiplier`."""
    logits_full = compute_logits_grid(model, ds)
    pairs = []
    for k in K:
        _, energy, dominant = delta_k_spectrum(model, ds, ci, basis, k, logits_full=logits_full)
        if dominant != k:
            mult = helper_multiplier(k, dominant, ds.p, max_mult=max_mult)
            pairs.append((k, dominant, mult, float(energy.max())))
    return pairs
