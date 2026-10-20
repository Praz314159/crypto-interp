"""1-D reductions of (a, b, c) signal grids over the group-coordinate difference.

The Delta_k pipeline reduces a logit/contribution grid on the group grid to a
function of ``D = (x(c) − x(a) − x(b)) mod n``, the coordinate distance of the
candidate c from the answer a∘b (for mul, the discrete-log difference
``dlog(c) − dlog(ab)``). Two earlier per-script copies disagreed on the
value-token column convention (full-vocab column == token c, vs group-restricted
column == the c-th group token) -- the recurring off-by-one. Here it lives in
ONE function with an explicit ``value_axis`` flag; both conventions, fed
correctly-shaped inputs, produce identical results.
"""

from __future__ import annotations

import numpy as np


def reduce_to_ab(delta_abc, structure) -> np.ndarray:
    """Average an (n, n, C) group-grid array over (a, b) pairs with the same product.
    Returns (n, C), indexed on axis 0 by the product's position among the group tokens."""
    arr = np.asarray(delta_abc, dtype=np.float64)
    n = structure.n
    position = np.empty(structure.n_values, dtype=np.int64)
    position[structure.token_array] = np.arange(n)
    ab = position[structure.product()]  # (n, n)
    out = np.zeros((n, arr.shape[-1]))
    cnt = np.zeros(n)
    np.add.at(out, ab.ravel(), arr.reshape(-1, arr.shape[-1]))
    np.add.at(cnt, ab.ravel(), 1.0)
    return out / np.where(cnt > 0, cnt, 1.0)[:, None]


def reduce_to_diff(delta_abc, structure, *, value_axis: str = "full") -> np.ndarray:
    """Reduce an (n, n, C) group-grid array over (a, b, c) to f(D), length n.

    ``value_axis``:
      - ``"full"``   : last axis is the full vocab (C == V or V+1 for V value
                       tokens); column index equals the token c, and the group
                       tokens' columns are used.
      - ``"values"`` : last axis is already restricted to the group tokens
                       (C == n); column j is the j-th group token.
    Both modes return the SAME result on correctly-shaped inputs.
    """
    arr = np.asarray(delta_abc, dtype=np.float64)
    if arr.ndim != 3:
        raise ValueError(f"expected a 3-D (n, n, C) grid, got shape {arr.shape}")
    n, V = structure.n, structure.n_values
    C = arr.shape[-1]

    if value_axis == "full":
        if C not in (V, V + 1):
            raise ValueError(f"value_axis='full' expects vocab columns (V or V+1), got C={C}")
        col_idx = structure.token_array
    elif value_axis == "values":
        if C != n:
            raise ValueError(f"value_axis='values' expects n columns, got C={C}")
        col_idx = np.arange(n)
    else:
        raise ValueError(f"value_axis must be 'full' or 'values', got {value_axis!r}")

    sub = arr[:, :, col_idx]  # (n, n, n), last axis = the group tokens in order
    x = structure.x
    x_ab = (x[:, None] + x[None, :]) % n  # (n, n): coordinate of the answer
    delta = (x[None, None, :] - x_ab[:, :, None]) % n  # (n, n, n)
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
    ang = 2 * np.pi * np.outer(ms, dl) / N  # (n_m, N)
    cos_proj = (f[None, :] * np.cos(ang)).sum(axis=1)
    sin_proj = (f[None, :] * np.sin(ang)).sum(axis=1)
    return cos_proj**2 + sin_proj**2
