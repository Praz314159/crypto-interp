"""1-D reductions of (a, b, c) signal grids over the discrete-log difference.

The Delta_k pipeline reduces a logit/contribution grid to a function of
``Dlog = (dlog(c) - dlog(a*b)) mod (p-1)``. Two earlier per-script copies
disagreed on the value-token column convention (full-vocab column == value c,
vs value-restricted column == c-1) -- the recurring off-by-one. Here it lives
in ONE function with an explicit ``value_axis`` flag; both conventions, fed
correctly-shaped inputs, produce identical results.
"""
from __future__ import annotations

import numpy as np

from .bases import discrete_log_table


def reduce_to_ab(delta_abc, p: int) -> np.ndarray:
    """Average a (p-1, p-1, C) grid over (a, b) pairs with the same product ab.
    Returns (p-1, C) indexed by ``ab-1`` on axis 0."""
    arr = np.asarray(delta_abc, dtype=np.float64)
    n = p - 1
    a = np.arange(1, p)
    ab = (a[:, None] * a[None, :]) % p              # (n, n), in 1..p-1
    out = np.zeros((n, arr.shape[-1]))
    cnt = np.zeros(n)
    np.add.at(out, ab.ravel() - 1, arr.reshape(-1, arr.shape[-1]))
    np.add.at(cnt, ab.ravel() - 1, 1.0)
    return out / np.where(cnt > 0, cnt, 1.0)[:, None]


def reduce_to_diff(delta_abc, p: int, *, value_axis: str = "full") -> np.ndarray:
    """Reduce a (p-1, p-1, C) grid over (a, b, c) to f(Dlog), length p-1.

    ``value_axis``:
      - ``"full"``   : last axis is the full vocab (C == p or p+1); column index
                       equals the value c, so columns 1..p-1 are used (c=0 dropped).
      - ``"values"`` : last axis is already restricted to c=1..p-1 (C == p-1);
                       column j corresponds to value j+1.
    Both modes return the SAME result on correctly-shaped inputs.
    """
    arr = np.asarray(delta_abc, dtype=np.float64)
    if arr.ndim != 3:
        raise ValueError(f"expected a 3-D (p-1, p-1, C) grid, got shape {arr.shape}")
    n = p - 1
    C = arr.shape[-1]

    _, dlog = discrete_log_table(p)
    dlog_of_value = np.array([dlog[v] for v in range(1, p)])   # index v-1 -> dlog(v)

    if value_axis == "full":
        if C not in (p, p + 1):
            raise ValueError(f"value_axis='full' expects vocab columns (p or p+1), got C={C}")
        col_idx = np.arange(1, p)            # columns 1..p-1 are values 1..p-1
    elif value_axis == "values":
        if C != n:
            raise ValueError(f"value_axis='values' expects p-1 columns, got C={C}")
        col_idx = np.arange(n)               # column j -> value j+1
    else:
        raise ValueError(f"value_axis must be 'full' or 'values', got {value_axis!r}")

    sub = arr[:, :, col_idx]                  # (n, n, n), last axis = c=1..p-1 in order
    j_c = dlog_of_value                       # (n,) dlog of value c=1..p-1

    a = np.arange(1, p)
    ab = (a[:, None] * a[None, :]) % p        # (n, n)
    j_ab = dlog_of_value[ab - 1]              # (n, n)

    delta = (j_c[None, None, :] - j_ab[:, :, None]) % n        # (n, n, n)
    out = np.zeros(n)
    cnt = np.zeros(n)
    np.add.at(out, delta.ravel(), sub.ravel())
    np.add.at(cnt, delta.ravel(), 1.0)
    return out / np.where(cnt > 0, cnt, 1.0)


def fourier_spectrum_1d(f_diff) -> np.ndarray:
    """Per-frequency energy (cos^2 + sin^2) of a 1-D Dlog signal.
    Returns array of length ``N//2`` indexed by ``m-1`` (frequency m = 1..N//2)."""
    f = np.asarray(f_diff, dtype=np.float64)
    N = len(f)
    dl = np.arange(N)
    ms = np.arange(1, N // 2 + 1)
    ang = 2 * np.pi * np.outer(ms, dl) / N            # (n_m, N)
    cos_proj = (f[None, :] * np.cos(ang)).sum(axis=1)
    sin_proj = (f[None, :] * np.sin(ang)).sum(axis=1)
    return cos_proj ** 2 + sin_proj ** 2
