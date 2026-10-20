"""The clock measures: a perfect clock has no deviation; a displaced token is found."""

import numpy as np

from crypto_interp.data import CyclicStructure
from crypto_interp.interp import discrete_log_table
from crypto_interp.measures import clock_defect, clock_repair, clock_residuals, errors
from crypto_interp.measures.clock import fit_clock

M = CyclicStructure.multiplicative


def test_clock_fit_flags_a_displaced_token():
    p, K = 13, (2, 5)
    _, dlog = discrete_log_table(p)
    rng = np.random.default_rng(0)
    C, S = rng.normal(size=(2, 8)), rng.normal(size=(2, 8))
    theta = 2 * np.pi * np.outer(K, [dlog[x] for x in range(1, p)]) / (p - 1)
    W = np.zeros((8, p + 1))
    W[:, 1:p] = C.T @ np.cos(theta) + S.T @ np.sin(theta)
    assert fit_clock(W, K, M(p)).plane_deviation().max() < 1e-9
    W[:, 4] += 0.5 * C[1]  # token 4 off the χ_5 clock
    dev = fit_clock(W, K, M(p)).plane_deviation()
    assert dev[1].argmax() == 3 and dev[1, 3] > dev[0].max()


def test_reference_model(S):
    c = clock_residuals(S)
    assert c.K == (3, 10, 20, 51) and c.in_dev.shape == (112,) and set(c.worst_k) <= set(c.K)
    e, d, r = errors(S), clock_defect(S), clock_repair(S)
    assert d.defect_tokens == e.defect_tokens and r.wrong_baseline == e.n_wrong
