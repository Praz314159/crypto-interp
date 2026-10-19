"""The noise_modes measure: a pure kernel has no noise; a class bias is single_c."""

import numpy as np
import pytest

from crypto_interp.interp import discrete_log_table
from crypto_interp.measures import noise_modes
from crypto_interp.measures.kernel import kernel_from_amplitudes
from crypto_interp.measures.noise import noise_modes_of

P = 13


def _kernel_logits(p=P, bias=None):
    """Units-grid logits that are a pure dlog kernel, plus an optional per-class bias."""
    _, dlog = discrete_log_table(p)
    alpha = np.zeros((p - 1) // 2 + 1)
    alpha[[1, 3]] = [5.0, 2.0]
    kappa = kernel_from_amplitudes(alpha, p)
    lg = np.array([dlog[x] for x in range(1, p)])
    d = (lg[:, None, None] + lg[None, :, None] - lg[None, None, :]) % (p - 1)
    logits = np.full((p - 1, p - 1, p + 1), -30.0)
    logits[:, :, 1:p] = kappa[d] + (0 if bias is None else bias[None, None, :])
    return logits


def test_noise_of_a_pure_kernel_is_zero_and_a_class_bias_is_single_c():
    clean = noise_modes_of(_kernel_logits(), P)
    assert clean.total_power < 1e-18
    biased = noise_modes_of(_kernel_logits(bias=np.linspace(-1, 1, P - 1)), P)
    assert biased.single_c == pytest.approx(biased.total_power, rel=1e-9)
    assert biased.top_classes[0] == "single_c"


def test_reference_model(S):
    n = noise_modes(S)
    assert n.algorithm < 1e-9 * n.total_power  # the kernel was subtracted
    assert len(n.top_classes) == len(n.top_modes) == 12
