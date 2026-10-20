"""Where a model's errors are: per-example margins and per-token error rates.

For every (a, b) on the group grid::

    margin = L(a, b, ab) − max_{c ≠ ab} L(a, b, c)     (> 0 ⇔ correct)

aggregated per input token (in the A and in the B position) and per target
class. An error pattern that is the same in the A and B positions points at
the token's embedding rather than at position-dependent circuitry.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .core import measure
from .kernel import kernel_margin, observed_kernel

#: Error rate above which an input token counts as a defect token.
DEFECT_ERROR_RATE = 0.01


@dataclass
class ErrorProfile:
    n_wrong: int  # wrong (a, b) pairs on the group grid
    n_total: int
    accuracy: float
    margin_min: float
    margin_p1: float  # 1st percentile of the margin
    margin_median: float
    kernel_margin: float  # κ_obs(0) − max_{d≠0} κ_obs(d): the kernel's own headroom
    defect_tokens: tuple[int, ...]  # tokens x with error rate > DEFECT_ERROR_RATE as A or B
    margin: np.ndarray  # (n, n), indexed by the positions of a and b among the group tokens
    target_logit: np.ndarray  # (n, n)
    max_alt_logit: np.ndarray  # (n, n)
    wrong: np.ndarray  # (n, n) bool
    err_by_a: np.ndarray  # (n,) error rate of each group token in position A
    err_by_b: np.ndarray  # (n,)
    err_by_target: np.ndarray  # (V,) error rate by target class c


def error_profile(logits_grid: np.ndarray, structure) -> ErrorProfile:
    """The error profile of a group-grid logits array (n, n, ≥ V)."""
    V = structure.n_values
    L = np.asarray(logits_grid)[..., :V].astype(np.float64)
    target = structure.product()
    target_logit = np.take_along_axis(L, target[..., None], axis=-1).squeeze(-1)
    alt = L.copy()
    np.put_along_axis(alt, target[..., None], -np.inf, axis=-1)
    max_alt = alt.max(axis=-1)
    margin = target_logit - max_alt
    wrong = L.argmax(axis=-1) != target

    err_by_a, err_by_b = wrong.mean(axis=1), wrong.mean(axis=0)
    per_target = np.bincount(target.ravel(), weights=wrong.ravel(), minlength=V)
    err_by_target = per_target / np.maximum(np.bincount(target.ravel(), minlength=V), 1)
    defect = structure.token_array[np.maximum(err_by_a, err_by_b) > DEFECT_ERROR_RATE]

    return ErrorProfile(
        n_wrong=int(wrong.sum()),
        n_total=int(wrong.size),
        accuracy=float(1 - wrong.mean()),
        margin_min=float(margin.min()),
        margin_p1=float(np.percentile(margin, 1)),
        margin_median=float(np.median(margin)),
        kernel_margin=kernel_margin(observed_kernel(logits_grid, structure)),
        defect_tokens=tuple(int(x) for x in defect),
        margin=margin,
        target_logit=target_logit,
        max_alt_logit=max_alt,
        wrong=wrong,
        err_by_a=err_by_a,
        err_by_b=err_by_b,
        err_by_target=err_by_target,
    )


@measure("errors", version=1)
def errors(S) -> ErrorProfile:
    """Margins and per-token error rates of the model in Session ``S``."""
    return error_profile(S.logits_grid.detach().cpu().numpy(), S.ds.structure)


def outlier_tokens(err: np.ndarray, sigma: float = 3.0) -> list[tuple[int, float]]:
    """Tokens (1-indexed) whose error rate exceeds mean + ``sigma`` std, worst first."""
    cut = err.mean() + sigma * err.std()
    out = [(i + 1, float(e)) for i, e in enumerate(err) if e > cut and e > 0]
    return sorted(out, key=lambda t: -t[1])
