import pytest

from crypto_interp.interp import char_index, order_of


@pytest.mark.parametrize("p", [113, 127, 181])
def test_char_index_layout(p):
    basis, ci = char_index(p)
    n = p - 1
    n_freq = (n - 1) // 2 + (1 if n % 2 == 0 else 0)
    assert basis.shape == (p, p)
    assert len(ci.freqs) == n_freq
    assert ci.p == p
    for k in ci.freqs:
        assert ci.by_char[k], f"empty rows for k={k}"
        if k in ci.cos:
            assert ci.cos[k] == 2 * k
        if k in ci.sin:
            assert ci.sin[k] == 2 * k + 1
    # the Nyquist character k=n/2 is cos-only for even n
    if n % 2 == 0:
        assert max(ci.freqs) not in ci.sin


def test_order_of():
    assert order_of(56, 113) == 2     # Legendre / quadratic-residue character
    assert order_of(16, 113) == 7     # pure Sylow-7
    assert order_of(1, 113) == 112
    assert order_of(3, 113) == 112
    assert order_of(7, 127) == 18     # 126 // gcd(7,126)=126//7
