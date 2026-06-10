"""Fold-aware character bookkeeping: the real cos/sin basis identifies frequency
f with (p-1)-f, so products in the complex character group must be folded back
before relating them to real-basis indices."""
from crypto_interp.interp import fold_frequency, helper_multiplier


def test_fold_frequency():
    # p=127, n=126: frequencies fold f <-> 126-f into 0..63
    assert fold_frequency(104, 127) == 22     # 2*52 = 104 folds to 22
    assert fold_frequency(30, 127) == 30      # already <= 63
    assert fold_frequency(63, 127) == 63      # Nyquist folds to itself
    assert fold_frequency(0, 127) == 0
    assert fold_frequency(66, 127) == 60      # 3*22 = 66 folds to 60


def test_helper_multiplier_fold_aware():
    # p=127 (the seed-3 cases): 30=2*15 (no fold), 22=fold(2*52) (folded doubling)
    assert helper_multiplier(30, 15, 127) == 2
    assert helper_multiplier(22, 52, 127) == 2      # would be missed without folding (2*52=104)
    # p=127 seed-1: 60 = fold(3*22) is a Sylow-3 rider, not a doubling
    assert helper_multiplier(60, 22, 127) == 3

    # p=113 (the original reference helpers): plain doublings
    assert helper_multiplier(44, 22, 113) == 2
    assert helper_multiplier(6, 3, 113) == 2
    assert helper_multiplier(20, 10, 113) == 2

    # an unrelated pair returns None
    assert helper_multiplier(7, 22, 127) is None
