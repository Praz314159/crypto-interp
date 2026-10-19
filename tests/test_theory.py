import pytest

from crypto_interp.theory.relu_harmonics import (
    RELU_COS_EXACT,
    product_fit_fvu,
    relu_cos_coefficients,
)


def test_relu_cos_second_harmonic():
    c = relu_cos_coefficients()
    for m, exact in RELU_COS_EXACT.items():
        assert c["relu"][m] == pytest.approx(exact, abs=1e-6)
    assert c["abs"][2] == pytest.approx(2 * RELU_COS_EXACT[2], abs=1e-6)  # 4/(3π)


def test_helper_features_lower_the_fit_error_at_small_width():
    fvu = product_fit_fvu(n=28, k=3, widths=[2], n_seeds=2, steps=1500)
    assert fvu["helper"][0] < fvu["primary"][0]
