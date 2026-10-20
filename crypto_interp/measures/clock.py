"""Embedding clock geometry: how far each token sits from its ideal clock position.

For each character χ_k in K, the embedding of token x should lie on a
clock: ``W_E[:, x] ≈ Σ_k C_k cos θ_k(x) + S_k sin θ_k(x)`` with
``θ_k(x) = 2πk·x(t)/n``, x the token's coordinate in the task's group (its
discrete log for mul mod p, where n = p − 1). :func:`fit_clock` fits the C_k, S_k by least
squares over the tokens. That is the run's own clock, so a perfect clock scores
zero whatever its eccentricity. Three measures build on it:

- :func:`clock_residuals`: per-token deviation from the fitted clock (the census);
- :func:`clock_defect`: which clock the defect tokens are most displaced on,
  and, separately, how tightly they cluster in angle on each clock;
- :func:`clock_repair`: a causal test. Move the defect tokens onto their ideal
  clock positions and recount errors, with two controls: the same move for as
  many random clean tokens, and for every token on that clock.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np
import torch

from .core import measure
from .errors import error_profile

TAU = 2 * np.pi


@dataclass
class ClockFit:
    K: tuple[int, ...]
    W: np.ndarray  # (d_model, n) embeddings of the group tokens, in token order
    theta: np.ndarray  # (|K|, n) ideal angles θ_k(x)
    C: np.ndarray  # (|K|, d_model) cosine directions
    S: np.ndarray  # (|K|, d_model) sine directions

    @property
    def ideal(self) -> np.ndarray:
        return self.C.T @ np.cos(self.theta) + self.S.T @ np.sin(self.theta)

    def plane(self, j: int) -> np.ndarray:
        """(d_model, 2) orthonormal basis of the j-th clock's plane."""
        Q, _ = np.linalg.qr(np.stack([self.C[j], self.S[j]], axis=1))
        return Q

    def plane_deviation(self) -> np.ndarray:
        """(|K|, n): the norm of each token's residual from the full fitted clock,
        projected into each clock plane. Zero for every token of a perfect clock.

        (The residual is taken against all clocks at once. Comparing a token with
        one clock's ideal alone would count the other clocks' projections into
        that plane as deviation wherever the planes are not orthogonal.)
        """
        resid = self.W - self.ideal
        return np.stack(
            [np.linalg.norm(self.plane(j).T @ resid, axis=0) for j in range(len(self.K))]
        )


def fit_clock(W_E: np.ndarray, K, structure) -> ClockFit:
    """Least-squares clock fit of the group tokens' columns of ``W_E`` on characters ``K``."""
    K = tuple(sorted(int(k) for k in K))
    W = np.asarray(W_E, dtype=np.float64)[:, structure.token_array]
    theta = TAU * np.outer(K, structure.x) / structure.n
    Phi = np.concatenate([np.cos(theta), np.sin(theta)], axis=0)  # (2|K|, n)
    coef, *_ = np.linalg.lstsq(Phi.T, W.T, rcond=None)  # (2|K|, d_model)
    return ClockFit(K, W, theta, coef[: len(K)], coef[len(K) :])


def _session_fit(S, K):
    from .characters import support

    K = support(S, K)
    return fit_clock(S.model.embed.W_E.detach().double().cpu().numpy(), K, S.ds.structure)


# --------------------------------------------------------------------------
# Census
# --------------------------------------------------------------------------


@dataclass
class ClockResiduals:
    K: tuple[int, ...]
    median_in_dev: float
    in_dev: np.ndarray  # (n,) in-subspace deviation from the fitted clock / |ideal|
    plane_dev: np.ndarray  # (|K|, n) deviation within each clock plane
    worst_k: np.ndarray  # (n,) the character whose plane each token deviates most in
    leak: np.ndarray  # (n,) fraction of the embedding outside every clock plane


@measure("clock_residuals", version=1)
def clock_residuals(S, K=None) -> ClockResiduals:
    """Per-token deviation of the embedding from its fitted clock geometry.
    ``K`` defaults to the energy support (see :func:`.characters.energy_support`)."""
    fit = _session_fit(S, K)
    ideal = fit.ideal
    Q, _ = np.linalg.qr(np.concatenate([fit.C, fit.S]).T)  # span of all clock planes
    resid = fit.W - ideal
    in_dev = np.linalg.norm(Q.T @ resid, axis=0) / (np.linalg.norm(ideal, axis=0) + 1e-12)
    outside = resid - Q @ (Q.T @ resid)
    leak = np.linalg.norm(outside, axis=0) / (np.linalg.norm(fit.W, axis=0) + 1e-12)
    plane_dev = fit.plane_deviation()
    return ClockResiduals(
        K=fit.K,
        median_in_dev=float(np.median(in_dev)),
        in_dev=in_dev,
        plane_dev=plane_dev,
        worst_k=np.array(fit.K)[plane_dev.argmax(axis=0)],
        leak=leak,
    )


# --------------------------------------------------------------------------
# Which clock, and which arc
# --------------------------------------------------------------------------


@dataclass
class ClockDefect:
    K: tuple[int, ...]
    defect_tokens: tuple[int, ...]  # tokens with error rate > DEFECT_ERROR_RATE
    kstar: int | None  # the clock most defect tokens deviate most on
    plane_z_kstar: tuple[float, ...]  # defect tokens' deviation on k*, in units of its median
    arc_R: float | None  # arc clustering of the defect tokens on k* (see arc_R_by_k)
    arc_R_by_k: tuple[float, ...]  # |mean e^{iθ_k(x)}| over defect tokens, per k in K:
    #   1 = all at one clock position, ≈ arc_R_null = spread uniformly
    arc_R_null: float | None  # ≈ expected R for that many uniformly spread tokens


def _implicated(S, K):
    fit = _session_fit(S, K)
    err = error_profile(S.logits_grid.detach().cpu().numpy(), S.ds.structure)
    pd = fit.plane_deviation()
    z = pd / (np.median(pd, axis=1, keepdims=True) + 1e-12)
    defect = list(err.defect_tokens)
    kstar = None
    if defect:
        pos = S.ds.structure.position
        j = Counter(int(z[:, pos[x]].argmax()) for x in defect).most_common(1)[0][0]
        kstar = fit.K[j]
    return fit, err, z, defect, kstar


@measure("clock_defect", version=1)
def clock_defect(S, K=None) -> ClockDefect:
    """Locate the defect: the implicated clock k*, and how tightly the defect
    tokens cluster in angle on it."""
    fit, _, z, defect, kstar = _implicated(S, K)
    if kstar is None:
        return ClockDefect(fit.K, (), None, (), None, (), None)
    pos = S.ds.structure.position
    idx = pos[np.array(defect)]
    R = np.abs(np.mean(np.exp(1j * fit.theta[:, idx]), axis=1))  # per clock
    j = fit.K.index(kstar)
    return ClockDefect(
        K=fit.K,
        defect_tokens=tuple(defect),
        kstar=kstar,
        plane_z_kstar=tuple(sorted((float(z[j, pos[x]]) for x in defect), reverse=True)),
        arc_R=float(R[j]),
        arc_R_by_k=tuple(float(r) for r in R),
        arc_R_null=float(0.886 / np.sqrt(len(defect))),
    )


# --------------------------------------------------------------------------
# Causal test
# --------------------------------------------------------------------------


@dataclass
class ClockRepair:
    kstar: int | None
    n_defect: int
    wrong_baseline: int
    wrong_repair_defects: int | None  # defect tokens moved to their ideal k* positions
    wrong_patch_random_clean: int | None  # control: as many random clean tokens moved
    #   (all clean tokens if there are fewer clean than defect tokens)
    wrong_straighten_clock: int | None  # control: every token moved onto the ideal k* clock


@measure("clock_repair", version=1)
def clock_repair(S, K=None, seed: int = 0) -> ClockRepair:
    """Move tokens' coordinates in the k* plane to their fitted ideal positions and
    recount wrong (a, b) pairs, for the defect tokens and for two controls."""
    from ..interp.grids import compute_logits_grid

    fit, err, _, defect, kstar = _implicated(S, K)
    if kstar is None:
        return ClockRepair(None, 0, err.n_wrong, None, None, None)
    s = S.ds.structure
    j = fit.K.index(kstar)
    Q = fit.plane(j)
    W_E = S.model.embed.W_E.detach()

    def wrong_after_moving(tokens) -> int:
        W = W_E.double().cpu().numpy().copy()
        for x in tokens:
            th = fit.theta[j, s.position[x]]
            ideal = fit.C[j] * np.cos(th) + fit.S[j] * np.sin(th)
            W[:, x] += Q @ (Q.T @ (ideal - W[:, x]))
        patched = torch.tensor(W, dtype=W_E.dtype, device=W_E.device)
        lg = compute_logits_grid(S.model, S.ds, W_E_override=patched).cpu().numpy()
        return error_profile(lg, s).n_wrong

    rng = np.random.default_rng(seed)
    clean = [x for x in s.tokens if x not in defect]
    n_control = min(len(defect), len(clean))  # a failed run can have more defect than clean
    return ClockRepair(
        kstar=kstar,
        n_defect=len(defect),
        wrong_baseline=err.n_wrong,
        wrong_repair_defects=wrong_after_moving(defect),
        wrong_patch_random_clean=wrong_after_moving(
            rng.choice(clean, size=n_control, replace=False)
        ),
        wrong_straighten_clock=wrong_after_moving(s.tokens),
    )


# --------------------------------------------------------------------------
# Token table
# --------------------------------------------------------------------------


def token_table(target):
    """One row per (run, token): error rates, minimum margin, clock deviation and
    leak, with the run's kernel_noise regime. Built from cached measures."""
    import pandas as pd

    from .core import _runs
    from .errors import errors
    from .kernel import kernel_noise

    rows = []
    for run in _runs(target):
        tokens = run.task.structure().tokens
        ckpt = run.latest()
        e, c, kn = errors.at(ckpt), clock_residuals.at(ckpt), kernel_noise.at(ckpt)
        min_margin = np.minimum(e.margin.min(axis=1), e.margin.min(axis=0))
        for i in range(len(e.err_by_a)):
            rows.append(
                {
                    "run": run.id,
                    "p": run.task.p,
                    "n": len(tokens),
                    "seed": run.task.seed,
                    "regime": kn.regime,
                    "n_wrong_run": e.n_wrong,
                    "token": tokens[i],
                    "err_a": float(e.err_by_a[i]),
                    "err_b": float(e.err_by_b[i]),
                    "min_margin": float(min_margin[i]),
                    "in_dev": float(c.in_dev[i]),
                    "worst_k": int(c.worst_k[i]),
                    "leak": float(c.leak[i]),
                }
            )
    return pd.DataFrame(rows)
