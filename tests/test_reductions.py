import numpy as np
import pytest

from crypto_interp.interp import discrete_log_table, fourier_spectrum_1d, reduce_to_diff


def _synthetic_grid(p, k):
    """delta[a-1, b-1, c] = cos(2 pi k * (dlog(c) - dlog(ab)) / n) over full vocab."""
    n = p - 1
    _, dlog = discrete_log_table(p)
    dlv = np.array([dlog[v] for v in range(1, p)])
    a = np.arange(1, p)
    ab = (a[:, None] * a[None, :]) % p
    j_ab = dlv[ab - 1]
    full = np.zeros((n, n, p))
    for cval in range(1, p):
        full[:, :, cval] = np.cos(2 * np.pi * k * ((dlv[cval - 1] - j_ab) % n) / n)
    return full


@pytest.mark.parametrize("p,k", [(113, 10), (127, 7)])
def test_reduce_to_diff_modes_agree_and_recover(p, k):
    full = _synthetic_grid(p, k)
    vals = full[:, :, 1:p]  # value-restricted columns c=1..p-1
    f_full = reduce_to_diff(full, p, value_axis="full")
    f_vals = reduce_to_diff(vals, p, value_axis="values")
    # the off-by-one regression: both conventions must agree exactly
    assert np.allclose(f_full, f_vals, atol=1e-9)
    # and recover a clean cosine at frequency k
    dominant = int(fourier_spectrum_1d(f_full).argmax() + 1)
    assert dominant == k
    ref = np.cos(2 * np.pi * k * np.arange(p - 1) / (p - 1))
    x = f_full - f_full.mean()
    y = ref - ref.mean()
    corr = float(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y)))
    assert corr > 0.999


def test_reduce_to_diff_shape_guards():
    with pytest.raises(ValueError):
        reduce_to_diff(np.zeros((112, 112, 50)), 113, value_axis="full")   # wrong vocab width
    with pytest.raises(ValueError):
        reduce_to_diff(np.zeros((112, 112, 113)), 113, value_axis="values")  # not p-1 columns
    with pytest.raises(ValueError):
        reduce_to_diff(np.zeros((112, 112)), 113)                          # not 3-D
