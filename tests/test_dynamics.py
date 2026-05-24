import numpy as np

from crypto_interp.interp import bifurcation_step, cliff_step, commit_step, grokking_status


def test_cliff_step():
    assert cliff_step([5, 4, 3, 0.05, 0.01], 0.1) == 3
    assert cliff_step([5, 4, 3], 0.1) is None


def test_grokking_status():
    s = grokking_status([5, 0.5, 0.01, 0.001], [5, 5, 5, 0.02], mem_thresh=0.1, grok_thresh=0.1)
    assert s["memorized_step"] == 2
    assert s["cliff_step"] == 3
    assert s["grokked"] is True

    s2 = grokking_status([5, 0.01], [5, 4], grok_thresh=0.1)
    assert s2["grokked"] is False
    assert s2["cliff_step"] is None


def test_bifurcation_and_commit():
    T, nch = 10, 4
    E = np.ones((T, nch)) * 0.1
    for t in range(T):
        E[t, 0] = 0.2 + 0.2 * t   # K characters start above and grow vs non-K
        E[t, 1] = 0.2 + 0.2 * t
    K_mask = np.array([True, True, False, False])

    bs = bifurcation_step(E, K_mask, ratio=1.5)
    assert bs is not None and bs > 0

    cs = commit_step(E, [1, 2], mode="subset")   # 1-based char ids of the two growers
    assert cs == 0
