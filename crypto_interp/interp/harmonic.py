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
from .bases import CharIndex
from .grids import compute_logits_grid
from .reductions import fourier_spectrum_1d, reduce_to_diff


def delta_k(model, ds, ci: CharIndex, basis, k: int, *, logits_full=None) -> np.ndarray:
    """Exact contribution of character k to the logits:
    ``logit_full - logit_ablated``, shape (p-1, p-1, vocab) as a numpy array.
    Pass ``logits_full`` (a torch grid) to avoid recomputing it across calls."""
    p = ds.p
    full = compute_logits_grid(model, ds) if logits_full is None else logits_full
    W_over = model.embed.W_E.detach().clone()
    W_ab = ablate_character(model.embed.W_E.detach()[:, :p], basis, ci, k)
    W_over[:, :p] = W_ab.to(W_over.dtype)
    ablated = compute_logits_grid(model, ds, W_E_override=W_over)
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


def find_primary_helper_pairs(model, ds, ci: CharIndex, basis, K):
    """For each k in K, classify as primary (dominant freq == k) or helper
    (dominant != k). Returns the list of ``(helper_k, primary_m, energy)`` for
    the helpers -- i.e. the detected harmonic-helper pairs."""
    logits_full = compute_logits_grid(model, ds)
    pairs = []
    for k in K:
        _, energy, dominant = delta_k_spectrum(model, ds, ci, basis, k, logits_full=logits_full)
        if dominant != k:
            pairs.append((k, dominant, float(energy.max())))
    return pairs
