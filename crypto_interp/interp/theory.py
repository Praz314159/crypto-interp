"""Approximate-CRT theory baseline.

Given a learned support ``K ⊂ {0, ..., p-2}`` (essential characters) and per-
character amplitudes ``α_k`` (extracted from the model or assumed uniform), the
approximate-CRT algorithm predicts that the logit assigned to candidate ``c``
under inputs ``(a, b)`` depends only on the dlog offset

    d = (log_g(ab) − log_g c) mod (p−1)

via the kernel

    κ(d) = Σ_{k ∈ K} α_k · cos(2π k d / (p−1)).

Because the correct answer is at ``d = 0`` (the kernel peak), the predicted
cross-entropy on the multiplicative grid reduces to the closed-form scalar

    L_theory = log Σ_d exp κ(d) − κ(0).

This module provides the verbs to:
  • extract κ_obs(d) from a trained model's logits grid (the model's actual
    "symmetric reduction"),
  • decompose κ_obs into per-character amplitudes via cosine-DFT,
  • re-build a kernel from a chosen subset K (the truncated theory),
  • compute the closed-form CE for any kernel.

The gap between L_theory(K, α_obs) and L_empirical is the part of the model's
behavior that **isn't** approximate-CRT with support K — a quantitative
identification metric for the learned algorithm.
"""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import torch

# --------------------------------------------------------------------------
# Kernel <-> amplitudes
# --------------------------------------------------------------------------


def kernel_from_amplitudes(
    alpha: np.ndarray,
    p: int,
    K: Optional[Iterable[int]] = None,
) -> np.ndarray:
    """Reconstruct κ(d) = Σ_k α_k cos(2π k d / (p−1)) for d ∈ {0, …, p−2}.

    ``alpha`` is indexed by k ∈ {0, …, ⌊(p−1)/2⌋}. If ``K`` is given, only
    amplitudes at indices in ``K`` are used (others zeroed).
    """
    n = p - 1
    n_half = len(alpha)
    d = np.arange(n)
    if K is None:
        ks = range(n_half)
    else:
        ks = [k for k in K if 0 <= k < n_half]
    kappa = np.zeros(n, dtype=np.float64)
    for k in ks:
        kappa += float(alpha[k]) * np.cos(2 * np.pi * k * d / n)
    return kappa


def kernel_loss(kappa: np.ndarray) -> float:
    """Closed-form CE: log Σ_d exp(κ(d)) − κ(0). Uses logsumexp for stability."""
    m = float(np.max(kappa))
    return float(m + np.log(np.sum(np.exp(kappa - m))) - kappa[0])


def amplitudes_from_kernel(kappa: np.ndarray, p: int) -> np.ndarray:
    """Cosine-DFT of κ, assuming κ is real and even in d.

    Returns ``alpha`` of length ⌊(p−1)/2⌋ + 1, where ``alpha[k]`` is the
    cosine amplitude such that

        κ(d) ≈ Σ_{k=0}^{⌊(p−1)/2⌋} alpha[k] · cos(2π k d / (p−1)).

    For real signals the rfft is conjugate-symmetric; for even-in-d signals
    the imaginary parts vanish and only cosine amplitudes survive.
    """
    n = p - 1
    F = np.fft.rfft(kappa)
    n_half = len(F)
    alpha = np.zeros(n_half, dtype=np.float64)
    alpha[0] = F[0].real / n
    if n % 2 == 0:
        # Even n: k=0 and k=n/2 are "self-conjugate" — no factor of 2.
        alpha[-1] = F[-1].real / n
        if n_half > 2:
            alpha[1:-1] = 2.0 * F[1:-1].real / n
    else:
        alpha[1:] = 2.0 * F[1:].real / n
    return alpha


# --------------------------------------------------------------------------
# Observed kernel from a model's logits grid
# --------------------------------------------------------------------------


def observed_kernel(
    logits_grid: torch.Tensor | np.ndarray,
    p: int,
    dlog: dict[int, int],
) -> np.ndarray:
    """Extract κ_obs(d) = mean over (a, b, c) of logit_c(a, b) at fixed offset d.

    ``logits_grid`` is shape ``(p−1, p−1, vocab)`` with first two dims indexed
    by ``a, b ∈ {1, …, p−1}`` and the last dim spanning vocabulary including
    ``c = 0`` at position 0. Only nonzero ``c`` are used.

    ``dlog`` maps ``a ∈ {1, …, p−1} ↦ log_g(a) ∈ {0, …, p−2}``.

    The output κ_obs is the symmetric reduction of the model's logits: if the
    model implements approximate-CRT, every ``(a, b)`` produces the same
    logit-vs-d profile up to noise, and this is that profile.
    """
    n = p - 1
    L = np.asarray(logits_grid)
    if L.shape[0] != n or L.shape[1] != n or L.shape[2] < p:
        raise ValueError(f"unexpected logits_grid shape {L.shape}, expected ({n}, {n}, ≥{p})")
    # Restrict to answer-class logits and drop c=0 column.
    L_pos = L[:, :, 1:p].astype(np.float64)

    dlog_arr = np.array([dlog[x] for x in range(1, p)], dtype=np.int64)
    # Broadcast (p-1, p-1, p-1) of offsets.
    d_arr = (dlog_arr[:, None, None] + dlog_arr[None, :, None] - dlog_arr[None, None, :]) % n

    kappa_sum = np.zeros(n, dtype=np.float64)
    counts = np.zeros(n, dtype=np.int64)
    np.add.at(kappa_sum, d_arr.ravel(), L_pos.ravel())
    np.add.at(counts, d_arr.ravel(), 1)
    return kappa_sum / np.maximum(counts, 1)


def empirical_test_loss_from_logits(
    logits_grid: torch.Tensor | np.ndarray,
    p: int,
) -> float:
    """Average CE over the full nonzero (a, b) grid (no train/test split here).

    Logits are restricted to the answer classes ``c ∈ {0, …, p−1}`` (dropping
    any ``=`` token in the vocab) so the softmax-normalization matches the
    model's actual class space.
    """
    L = np.asarray(logits_grid)[..., :p].astype(np.float64)
    # Targets: c = (a * b) % p for a, b ∈ {1, …, p−1}.
    a = np.arange(1, p)[:, None]
    b = np.arange(1, p)[None, :]
    target = (a * b) % p
    m = L.max(axis=-1, keepdims=True)
    logsumexp = m.squeeze(-1) + np.log(np.sum(np.exp(L - m), axis=-1))
    target_logit = np.take_along_axis(L, target[..., None], axis=-1).squeeze(-1)
    return float((logsumexp - target_logit).mean())


# --------------------------------------------------------------------------
# High-level convenience: theory baseline for a Session
# --------------------------------------------------------------------------


def theory_baseline(
    session,
    K: Optional[Iterable[int]] = None,
    use_observed_amplitudes: bool = True,
) -> dict:
    """One-call theory baseline for a Session.

    Returns a dict with:
        kappa_obs      — observed kernel from model logits
        alpha          — per-character cosine amplitudes from κ_obs
        K              — support used (defaults to ``session.essential()["K"]``)
        L_empirical    — model's CE averaged over the full nonzero grid
        L_symmetric    — CE of the symmetric-reduced logits (κ_obs treated as logits)
        L_theory_K     — CE of the K-truncated theory kernel (the headline number)
        kappa_K        — the truncated kernel (for plotting alongside κ_obs)

    Three useful diagnostics from the dict:
      • ``L_symmetric ≈ L_empirical``  ⇒ model's logits are essentially a function
        of the dlog offset alone (approximate-CRT symmetry holds).
      • ``L_theory_K ≈ L_symmetric``  ⇒ the symmetric kernel is already supported
        on K (truncation is lossless).
      • ``L_theory_K ≪ L_symmetric``  ⇒ K omits significant amplitude; the
        essential-character heuristic is missing something.
    """
    from .bases import discrete_log_table

    p = session.ds.p
    if K is None:
        K = session.essential()["K"]
    K = sorted(int(k) for k in K)

    _g, dlog = discrete_log_table(p)
    logits = session.logits_grid.detach().cpu()
    # The Session's logits_grid has shape (p-1, p-1, vocab). Match observed_kernel's
    # expectations.
    kappa_obs = observed_kernel(logits, p, dlog)
    alpha = amplitudes_from_kernel(kappa_obs, p)

    # Truncate to K (folding k and (p-1)-k together, since cos is even).
    n = p - 1
    K_folded = {min(int(k), n - int(k)) for k in K}
    kappa_K = kernel_from_amplitudes(alpha, p, K=K_folded)

    return {
        "K": K,
        "p": p,
        "alpha": alpha,
        "kappa_obs": kappa_obs,
        "kappa_K": kappa_K,
        "L_empirical": empirical_test_loss_from_logits(logits, p),
        "L_symmetric": kernel_loss(kappa_obs),
        "L_theory_K": kernel_loss(kappa_K),
    }
