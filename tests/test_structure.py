"""CyclicStructure: the group a task computes, for addition mod n and multiplication mod p."""

import math

import numpy as np
import pytest
import torch

from crypto_interp.data import CyclicStructure, TaskSpec, group_order


def _labels(ds):
    return dict(zip(map(tuple, ds.inputs[:, :2].tolist()), ds.labels.tolist(), strict=True))


@pytest.mark.parametrize(("task", "p"), [("mul", 13), ("mul_units", 13), ("add", 12), ("add", 15)])
def test_product_grid_is_the_task_label(task, p):
    ds = TaskSpec(task, p).build()
    s = ds.structure
    assert s.n == group_order(task, p) and len(s.tokens) == s.n
    labels, prod = _labels(ds), s.product()
    for i, a in enumerate(s.tokens):
        for j, b in enumerate(s.tokens):
            assert prod[i, j] == labels[(a, b)]


def test_multiplicative_coordinates_are_discrete_logs():
    s = CyclicStructure.multiplicative(113)
    assert (s.n, s.n_values, s.generator, s.op) == (112, 113, 3, "×")
    assert s.tokens == tuple(range(1, 113))
    assert all(pow(3, x, 113) == t for t, x in zip(s.tokens, s.coords, strict=True))
    assert list(s.token_array[s.coord_order]) == [pow(3, x, 113) for x in range(112)]


@pytest.mark.parametrize("s", [CyclicStructure.additive(12), CyclicStructure.multiplicative(13)])
def test_char_basis_is_orthonormal_and_indexed(s):
    basis, ci = s.char_basis()
    assert basis.shape == (s.n_values, s.n_values)
    assert torch.allclose(basis @ basis.T, torch.eye(s.n_values, dtype=torch.float64))
    assert ci.freqs == list(s.freqs) and ci.n == s.n
    for k in ci.freqs:  # each character's cos row is a positive multiple of cos θ_k
        row = basis[ci.cos[k]].numpy()
        expected = np.zeros(s.n_values)
        expected[list(s.tokens)] = np.cos(s.theta(k))
        assert np.allclose(row / np.linalg.norm(row), expected / np.linalg.norm(expected))
    outside = [t for t in range(s.n_values) if t not in s.tokens]
    assert [ci.names[i] for i in range(len(outside))] == [f"delta_{t}" for t in outside]


def test_order_and_fold():
    s = CyclicStructure.additive(12)
    assert [s.order(k) for k in range(1, 7)] == [12, 6, 4, 3, 12, 2]
    assert s.fold(9) == 3 and s.fold(-1) == 1 and s.fold(18) == 6
    m = CyclicStructure.multiplicative(113)
    assert m.order(56) == 2 and m.order(16) == 7 and m.order(3) == 112
    assert all(m.order(k) == 112 // math.gcd(k, 112) for k in m.freqs)
