"""Kernel/noise decomposition: the math, the regime rule, and the reference model."""

import math

import numpy as np
import pytest

from crypto_interp.interp import discrete_log_table
from crypto_interp.measures.kernel import (
    amplitudes_from_kernel,
    classify_regime,
    crt_floor,
    crt_gcd,
    empirical_loss,
    kernel_from_amplitudes,
    kernel_loss,
    kernel_noise,
    observed_kernel,
)

P = 13


def test_amplitudes_round_trip():
    alpha = np.zeros((P - 1) // 2 + 1)
    alpha[[0, 2, 6]] = [0.5, 3.0, -1.0]
    kappa = kernel_from_amplitudes(alpha, P)
    assert np.allclose(amplitudes_from_kernel(kappa, P), alpha)


def test_pure_kernel_logits_recover_the_kernel_and_its_loss():
    """Logits that depend only on the dlog offset: κ_obs is that kernel, and the
    empirical loss equals the closed-form kernel loss."""
    _, dlog = discrete_log_table(P)
    alpha = np.zeros((P - 1) // 2 + 1)
    alpha[[1, 4]] = [4.0, 2.0]
    kappa = kernel_from_amplitudes(alpha, P)
    lg = np.array([dlog[x] for x in range(1, P)])
    d = (lg[:, None, None] + lg[None, :, None] - lg[None, None, :]) % (P - 1)
    logits = np.full((P - 1, P - 1, P + 1), -50.0)
    logits[:, :, 1:P] = kappa[d]

    assert np.allclose(observed_kernel(logits, P, dlog), kappa)
    assert empirical_loss(logits, P) == pytest.approx(kernel_loss(kappa), rel=1e-9)


def test_crt_rule():
    # p = 113: p-1 = 112 = 2^4 * 7. K = {2, 4} (seed 9) misses the odd part of the 2-group.
    assert crt_gcd([2, 4], 113) == 2 and crt_floor([2, 4], 113) == pytest.approx(math.log(2))
    assert crt_gcd([3, 10, 20, 51], 113) == 1
    assert classify_regime([2, 4], 113, L_emp=1.4, L_sym=1.3) == "CRT-fail"
    assert classify_regime([3, 10], 113, L_emp=1e-5, L_sym=1e-6) == "grokked"
    assert classify_regime([3, 10], 113, L_emp=0.05, L_sym=1e-4) == "algorithmic-but-noisy"
    assert classify_regime([3, 10], 113, L_emp=0.5, L_sym=0.4) == "intermediate"
    assert classify_regime([], 113, L_emp=4.7, L_sym=4.7) == "intermediate"


def test_crt_floor_bounds_any_kernel_supported_on_K():
    """With g = gcd(K ∪ {p−1}) > 1 the kernel has period (p−1)/g, so it ties κ(0) at
    g offsets and its loss is at least log g, whatever the amplitudes."""
    rng = np.random.default_rng(0)
    for K, p in [([2, 4], 113), ([22, 34, 36, 56], 113), ([39, 78], 181), ([6, 9], 127)]:
        for _ in range(20):
            alpha = np.zeros((p - 1) // 2 + 1)
            alpha[K] = rng.normal(scale=10.0, size=len(K))
            assert kernel_loss(kernel_from_amplitudes(alpha, p)) >= crt_floor(K, p) - 1e-9


def test_reference_model_is_grokked(S):
    r = kernel_noise(S)
    assert r.K == (3, 10, 20, 51)
    assert r.regime == "grokked" and r.crt_gcd == 1 and r.crt_floor == 0
    assert r.L_emp < 1e-3 and r.L_sym < r.L_emp  # the kernel alone is better than the model
    assert r.kernel_margin > 0 and r.kappa_obs.shape == (112,)
