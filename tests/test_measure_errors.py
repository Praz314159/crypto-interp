"""The errors measure: planted errors and the reference model."""

import numpy as np
import pytest

from crypto_interp.data import CyclicStructure
from crypto_interp.interp import discrete_log_table
from crypto_interp.measures import errors, kernel_noise
from crypto_interp.measures.errors import error_profile
from crypto_interp.measures.kernel import kernel_from_amplitudes

P = 13
M = CyclicStructure.multiplicative


def _kernel_logits(p=P, bias=None):
    """Units-grid logits that are a pure dlog kernel, plus an optional per-class bias."""
    _, dlog = discrete_log_table(p)
    alpha = np.zeros((p - 1) // 2 + 1)
    alpha[[1, 3]] = [5.0, 2.0]
    kappa = kernel_from_amplitudes(alpha, n=p - 1)
    lg = np.array([dlog[x] for x in range(1, p)])
    d = (lg[:, None, None] + lg[None, :, None] - lg[None, None, :]) % (p - 1)
    logits = np.full((p - 1, p - 1, p + 1), -30.0)
    logits[:, :, 1:p] = kappa[d] + (0 if bias is None else bias[None, None, :])
    return logits


def test_error_profile_finds_planted_errors():
    logits = _kernel_logits()
    a, b = 3, 5  # make (3, 5) wrong: push a different class above the target
    logits[a - 1, b - 1, (a * b + 1) % P] = 100.0
    e = error_profile(logits, M(P))
    assert e.n_wrong == 1 and e.n_total == (P - 1) ** 2
    assert (
        e.margin[a - 1, b - 1] < 0
        and (np.delete(e.margin.ravel(), (a - 1) * (P - 1) + b - 1) > 0).all()
    )
    assert e.err_by_a[a - 1] == pytest.approx(1 / (P - 1))
    assert e.defect_tokens == (a, b)  # one error in 12 is above the 1% defect threshold


def test_reference_model(S):
    e = errors(S)
    assert e.n_total == 112**2 and e.accuracy == pytest.approx(1 - e.n_wrong / e.n_total)
    assert e.kernel_margin == pytest.approx(kernel_noise(S).kernel_margin)
