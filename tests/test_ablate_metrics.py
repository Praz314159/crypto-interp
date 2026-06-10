import numpy as np
import torch

from crypto_interp.interp import (
    ablate_character,
    char_energy,
    char_index,
    per_frequency_energy_from_embedding,
    project_back,
)


def test_ablate_character_zeros_energy_and_matches_project_back():
    p = 113
    basis, ci = char_index(p)
    torch.manual_seed(0)
    W = torch.randn(24, p, dtype=torch.float64)
    k = 10

    W_ab = ablate_character(W, basis, ci, k)
    ce = char_energy(W_ab, basis, ci)
    ce0 = char_energy(W, basis, ci)

    assert ce[k - 1] < 1e-9                       # character k removed
    for kk in ci.freqs:                            # others untouched
        if kk != k:
            assert abs(ce[kk - 1] - ce0[kk - 1]) < 1e-6

    # equivalent to project_back with the complementary keep-mask
    mask = torch.ones(p, dtype=torch.bool)
    for r in ci.by_char[k]:
        mask[r] = False
    W_pb = project_back(W, basis, mask)
    assert torch.allclose(W_ab, W_pb.double(), atol=1e-9)


def test_char_energy_matches_per_frequency_list():
    p = 113
    basis, ci = char_index(p)
    torch.manual_seed(1)
    W = torch.randn(24, p, dtype=torch.float64)
    ce = char_energy(W, basis, ci)
    ref = np.array(per_frequency_energy_from_embedding(W, basis, p))
    assert np.allclose(ce[: len(ref)], ref, atol=1e-9)
