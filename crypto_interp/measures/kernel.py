"""Kernel/noise decomposition of a model of a cyclic group operation.

In the coordinates of the task's cyclic group of order n (the discrete log for
multiplication mod p, where n = p − 1; the residue itself for addition mod n),
if the model implements approximate CRT on the characters K, the logit it
assigns to candidate ``c`` for inputs ``(a, b)`` depends only on the offset
``d = (x(a) + x(b) − x(c)) mod n``, through a kernel

    κ(d) = Σ_{k ∈ K} α_k · cos(2π k d / n).

The correct answer sits at the kernel peak ``d = 0``, so a pure kernel
predicts the cross-entropy ``L(κ) = log Σ_d exp κ(d) − κ(0)``.

The measure :func:`kernel_noise` splits a model's loss into two parts:

- ``L_sym``: the loss of κ_obs, the mean logit at each offset (the symmetric,
  algorithmic part of the model);
- ``L_emp``: the model's actual loss on the units grid. The gap is the
  asymmetric "noise" the kernel doesn't explain.

Both are also computed on the held-out pairs alone (``L_sym_test``,
``L_emp_test``), where memorization cannot raise the correct answer's logit.
The regime uses those, so a run whose kernel only looks clean because of its
training pairs is not called algorithmic.

It also classifies the outcome:

- **CRT-fail** when the (non-empty) characters in K cannot separate all of the
  group, i.e. ``g = gcd(K ∪ {n}) > 1``. A kernel supported on K then has period
  n/g, ties κ(0) at g offsets, and so has loss at least ``log g`` (the
  algebraic floor). The floor bounds ``L_K``; ``L_sym`` can fall below it,
  because κ_obs also carries small amplitudes on characters outside K;
- otherwise **grokked** if ``L_emp_test < 1e-2``, **algorithmic-but-noisy** if
  only ``L_sym_test < 1e-2``, and **intermediate** if neither.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import torch

from .characters import support
from .core import measure

#: Loss below which a model (L_emp) or its kernel (L_sym) counts as solved.
SOLVED = 1e-2

REGIMES = ("grokked", "algorithmic-but-noisy", "CRT-fail", "intermediate")


# --------------------------------------------------------------------------
# Kernel <-> amplitudes
# --------------------------------------------------------------------------


def kernel_from_amplitudes(
    alpha: np.ndarray, *, n: int, K: Iterable[int] | None = None
) -> np.ndarray:
    """κ(d) = Σ_k α_k cos(2π k d / n) for d ∈ Z/n, n the group order.

    ``alpha`` is indexed by k ∈ {0, …, ⌊n/2⌋}. If ``K`` is given, only those
    amplitudes are used.
    """
    d = np.arange(n)
    ks = range(len(alpha)) if K is None else [k for k in K if 0 <= k < len(alpha)]
    kappa = np.zeros(n, dtype=np.float64)
    for k in ks:
        kappa += float(alpha[k]) * np.cos(2 * np.pi * k * d / n)
    return kappa


def amplitudes_from_kernel(kappa: np.ndarray) -> np.ndarray:
    """Cosine amplitudes ``alpha[k]``, k ∈ {0, …, ⌊n/2⌋}, of a real even kernel on Z/n."""
    n = len(kappa)
    F = np.fft.rfft(kappa)
    alpha = np.zeros(len(F), dtype=np.float64)
    alpha[0] = F[0].real / n
    if n % 2 == 0:
        # k = 0 and k = n/2 are their own conjugates: no factor of 2.
        alpha[-1] = F[-1].real / n
        alpha[1:-1] = 2.0 * F[1:-1].real / n
    else:
        alpha[1:] = 2.0 * F[1:].real / n
    return alpha


def kernel_loss(kappa: np.ndarray) -> float:
    """Closed-form cross-entropy of a pure kernel: log Σ_d exp κ(d) − κ(0)."""
    m = float(np.max(kappa))
    return float(m + np.log(np.sum(np.exp(kappa - m))) - kappa[0])


def kernel_margin(kappa: np.ndarray) -> float:
    """How far the correct answer's logit (d = 0) clears the best wrong offset."""
    return float(kappa[0] - kappa[1:].max())


# --------------------------------------------------------------------------
# From a model's logits on the group grid
# --------------------------------------------------------------------------


def observed_kernel(logits_grid, structure, pairs: np.ndarray | None = None) -> np.ndarray:
    """κ_obs(d): the mean logit over all group tokens (a, b, c) at offset
    d = x(a) + x(b) − x(c) mod n (for mul, the dlog offset).

    ``logits_grid`` is the group grid, shape (n, n, ≥ V) for V value tokens, with
    the vocabulary on the last axis; only the group tokens' columns are used.
    ``pairs``, an (n, n) boolean mask, restricts the mean to those (a, b).
    """
    n, V = structure.n, structure.n_values
    L = np.asarray(logits_grid)
    if L.shape[:2] != (n, n) or L.shape[2] < V:
        raise ValueError(f"unexpected logits_grid shape {L.shape}, expected ({n}, {n}, ≥{V})")
    d = structure.offsets()
    w = L[:, :, structure.token_array].astype(np.float64)
    if pairs is not None:
        d, w = d[pairs], w[pairs]
    total = np.bincount(d.ravel(), weights=w.ravel(), minlength=n)
    return total / np.bincount(d.ravel(), minlength=n)


def empirical_loss(logits_grid, structure, pairs: np.ndarray | None = None) -> float:
    """Mean cross-entropy over the group grid (or the (a, b) in ``pairs``),
    softmax over the answer classes, the V value tokens."""
    L = np.asarray(logits_grid)[..., : structure.n_values].astype(np.float64)
    target = structure.product()
    m = L.max(axis=-1, keepdims=True)
    lse = m.squeeze(-1) + np.log(np.sum(np.exp(L - m), axis=-1))
    ce = lse - np.take_along_axis(L, target[..., None], axis=-1).squeeze(-1)
    return float((ce if pairs is None else ce[pairs]).mean())


# --------------------------------------------------------------------------
# CRT coverage and regimes
# --------------------------------------------------------------------------


def crt_gcd(K: Iterable[int], *, n: int) -> int:
    """gcd(K ∪ {n}): 1 iff the characters in K jointly separate the group of order n.

    Equivalently n / lcm of the characters' orders, the size of the subgroup
    that every χ_k, k ∈ K, maps to 1.
    """
    return math.gcd(n, *(int(k) for k in K))


def crt_floor(K: Iterable[int], *, n: int) -> float:
    """log gcd(K ∪ {n}): a lower bound on the loss of any kernel supported on K (L_K)."""
    return math.log(crt_gcd(K, n=n))


def classify_regime(K: Iterable[int], *, n: int, L_emp: float, L_sym: float) -> str:
    """The regime from K and a pair of losses (:func:`kernel_noise` passes the held-out ones)."""
    K = list(K)
    if K and crt_gcd(K, n=n) > 1:  # an empty K (nothing learned yet) is not a CRT failure
        return "CRT-fail"
    if L_emp < SOLVED:
        return "grokked"
    if L_sym < SOLVED:
        return "algorithmic-but-noisy"
    return "intermediate"


# --------------------------------------------------------------------------
# The measure
# --------------------------------------------------------------------------


@dataclass
class KernelNoise:
    K: tuple[int, ...]  # support used (default: the energy support)
    L_emp: float  # model loss on the group grid
    L_sym: float  # loss of the observed kernel κ_obs
    L_emp_test: float  # L_emp over the held-out pairs only
    L_sym_test: float  # L_sym with κ_obs from the held-out pairs only
    L_K: float  # loss of κ_obs truncated to K
    kernel_margin: float  # κ_obs(0) − max_{d≠0} κ_obs(d)
    crt_gcd: int  # gcd(K ∪ {n}); > 1 means CRT failure
    crt_floor: float  # log crt_gcd, lower bound on L_K
    regime: str  # one of REGIMES, from the held-out losses
    kappa_obs: np.ndarray  # κ_obs(d), d = 0..n−1
    kappa_K: np.ndarray  # κ_obs truncated to K
    alpha: np.ndarray  # cosine amplitudes of κ_obs


@measure("kernel_noise", version=2)
def kernel_noise(S, K: Iterable[int] | None = None) -> KernelNoise:
    """Kernel/noise decomposition and regime of the model in Session ``S``.

    ``K`` defaults to the energy support, :func:`.characters.energy_support`.
    """
    from ..interp.grids import group_mask

    s = S.ds.structure
    n = s.n
    K = support(S, K)
    logits = S.logits_grid.detach().cpu()
    if isinstance(logits, torch.Tensor):
        logits = logits.numpy()

    kappa_obs = observed_kernel(logits, s)
    alpha = amplitudes_from_kernel(kappa_obs)
    kappa_K = kernel_from_amplitudes(alpha, n=n, K={s.fold(k) for k in K})  # cos is even
    L_emp = empirical_loss(logits, s)
    L_sym = kernel_loss(kappa_obs)
    held_out = group_mask(S.ds, S.ds.test_mask).numpy()
    L_emp_test = empirical_loss(logits, s, held_out)
    L_sym_test = kernel_loss(observed_kernel(logits, s, held_out))
    return KernelNoise(
        K=K,
        L_emp=L_emp,
        L_sym=L_sym,
        L_emp_test=L_emp_test,
        L_sym_test=L_sym_test,
        L_K=kernel_loss(kappa_K),
        kernel_margin=kernel_margin(kappa_obs),
        crt_gcd=crt_gcd(K, n=n),
        crt_floor=crt_floor(K, n=n),
        regime=classify_regime(K, n=n, L_emp=L_emp_test, L_sym=L_sym_test),
        kappa_obs=kappa_obs,
        kappa_K=kappa_K,
        alpha=alpha,
    )
