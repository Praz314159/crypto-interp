"""Character measures on the reference model (K = {3, 10, 20, 51}; χ_20 doubles χ_10)."""

import math

from crypto_interp.measures import (
    char_ablation,
    energy_support,
    fourier2d,
    helpers,
    kernel_noise,
    neuron_clusters,
)


def test_energy_support_is_the_default_K(S):
    sup = energy_support(S)
    assert sup.K == (3, 10, 20, 51) and sup.orders == (112, 56, 28, 112)
    assert kernel_noise(S).K == sup.K
    assert sup.energy.shape == sup.char_order.shape == (56,)


def test_ablation_classes_are_a_separate_question(S):
    """Ablation's 'essential' is not K: the documented note-06 picture."""
    a = char_ablation(S)
    assert {3, 10, 51} <= set(a.essential)
    assert {6, 20} <= set(a.essential) | set(a.load_bearing)
    assert a.control_k not in energy_support(S).K


def test_helpers_find_the_doubling(S):
    h = helpers(S)
    assert h.primaries == (3, 10, 51)
    assert (h.helpers, h.helper_of, h.helper_mult) == ((20,), (10,), (2,))
    assert h.spectrum.shape == (4, 56) and h.delta_ab.shape == (4, 112, 112)


def test_neuron_clusters(S):
    nc = neuron_clusters(S)
    assert nc.sizes == (11, 13, 0, 8)
    assert nc.corr[0] > 0.97 and nc.corr[1] > 0.95 and math.isnan(nc.corr[2])


def test_mlp_output_is_on_the_character_diagonal(S):
    f = fourier2d(S)
    assert f.energy.shape == (56, 56) and f.diagonal_fraction > 0.9
