"""The asymmetric noise term, split into Fourier mode classes.

In discrete-log coordinates (x_a, x_b, x_c) the logits decompose as

    L(x_a, x_b, x_c) = κ_obs(x_a + x_b − x_c) + ε(x_a, x_b, x_c),

where κ_obs is the symmetric kernel (the algorithm; see :mod:`.kernel`) and ε
the asymmetric residual, zero-mean at every offset. ε's 3-D Fourier modes
(k_a, k_b, k_c) on (Z/(p−1))³ fall into classes with distinct mechanistic
readings:

==============  ==========================================  ===========================
class           modes                                       reading
==============  ==========================================  ===========================
single_a        k_b = k_c = 0, k_a ≠ 0                      output depends on a alone
single_b        k_a = k_c = 0, k_b ≠ 0                      output depends on b alone
single_c        k_a = k_b = 0, k_c ≠ 0                      class-prior bias
pair_ac         k_b = 0; k_a, k_c ≠ 0                       a–c shortcut
pair_bc         k_a = 0; k_b, k_c ≠ 0                       b–c shortcut
pair_ab         k_c = 0; k_a ≠ k_b, both ≠ 0                input interaction, no readout
off_resonance   k_a = k_b, k_c ≠ −k_a                       right input product, read out
                                                            at the wrong output character
generic         everything else
algorithm       k_a = k_b, k_c = −k_a                       the kernel; zero in ε
==============  ==========================================  ===========================

Powers are Σ|Ê|² over each class, with Ê = FFT(ε)/(p−1).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .core import measure
from .kernel import empirical_loss, kernel_loss, observed_kernel

NOISE_CLASSES = (
    "single_a",
    "single_b",
    "single_c",
    "pair_ac",
    "pair_bc",
    "pair_ab",
    "off_resonance",
    "generic",
)


def logits_in_coords(logits_grid: np.ndarray, structure) -> np.ndarray:
    """``L[x_a, x_b, x_c]`` = logit of c for inputs (a, b), indexed by group
    coordinates (discrete logs for mul)."""
    by_x = structure.coord_order  # by_x[x] = position of the group token with coordinate x
    L = np.asarray(logits_grid)[..., structure.token_array].astype(np.float64)
    return L[np.ix_(by_x, by_x, by_x)]


def mode_classes(n: int) -> dict[str, np.ndarray]:
    """Boolean (n, n, n) masks over Fourier modes (k_a, k_b, k_c), one per class."""
    ka = np.arange(n)[:, None, None]
    kb = np.arange(n)[None, :, None]
    kc = np.arange(n)[None, None, :]
    algorithm = (ka == kb) & ((ka + kc) % n == 0)
    masks = {
        "algorithm": algorithm,
        "single_a": (kb == 0) & (kc == 0) & (ka != 0),
        "single_b": (ka == 0) & (kc == 0) & (kb != 0),
        "single_c": (ka == 0) & (kb == 0) & (kc != 0),
        "pair_ac": (kb == 0) & (ka != 0) & (kc != 0),
        "pair_bc": (ka == 0) & (kb != 0) & (kc != 0),
        "pair_ab": (kc == 0) & (ka != 0) & (kb != 0) & (ka != kb),
        "off_resonance": (ka == kb) & ~algorithm,
    }
    masks["generic"] = ~np.logical_or.reduce(list(masks.values()))
    return masks


@dataclass
class NoiseModes:
    L_emp: float
    L_sym: float
    total_power: float  # Σ|Ê|² over all modes of ε
    noise_at_target_var: float  # Var over (a, b) of ε(a, b, ab): the noise the CE sees
    noise_at_target_mean: float  # ≈ 0 by construction
    kernel_3d_power: float  # n² Σ_d κ_obs(d)²
    snr_3d: float  # kernel_3d_power / total_power
    single_a: float
    single_b: float
    single_c: float
    pair_ac: float
    pair_bc: float
    pair_ab: float
    off_resonance: float
    generic: float
    algorithm: float  # ≈ 0: the kernel was subtracted
    top_classes: tuple[str, ...]  # classes of the strongest modes
    top_modes: np.ndarray  # (n_top, 4): k_a, k_b, k_c, power


def noise_modes_of(logits_grid: np.ndarray, structure, n_top: int = 12) -> NoiseModes:
    n = structure.n
    kappa = observed_kernel(logits_grid, structure)
    L = logits_in_coords(logits_grid, structure)
    x = np.arange(n)
    d = (x[:, None, None] + x[None, :, None] - x[None, None, :]) % n
    eps = L - kappa[d]

    power = np.abs(np.fft.fftn(eps) / n) ** 2
    masks = mode_classes(n)
    by_class = {name: float(power[m].sum()) for name, m in masks.items()}
    at_target = eps[x[:, None], x[None, :], (x[:, None] + x[None, :]) % n]

    # Conjugate modes have equal power; a stable sort breaks ties by mode index.
    order = np.argsort(-power.ravel(), kind="stable")[:n_top]
    top = np.array([[*np.unravel_index(i, power.shape), power.ravel()[i]] for i in order])
    top_classes = tuple(
        next(name for name, m in masks.items() if m[tuple(int(v) for v in row[:3])]) for row in top
    )
    total = float(power.sum())
    kernel_3d = n * n * float(np.sum(kappa**2))
    return NoiseModes(
        L_emp=empirical_loss(logits_grid, structure),
        L_sym=kernel_loss(kappa),
        total_power=total,
        noise_at_target_var=float(at_target.var()),
        noise_at_target_mean=float(at_target.mean()),
        kernel_3d_power=kernel_3d,
        snr_3d=kernel_3d / max(total, 1e-300),
        top_classes=top_classes,
        top_modes=top,
        **by_class,
    )


@measure("noise_modes", version=1)
def noise_modes(S, n_top: int = 12) -> NoiseModes:
    """Fourier mode classes of the asymmetric residual of the model in Session ``S``."""
    return noise_modes_of(S.logits_grid.detach().cpu().numpy(), S.ds.structure, n_top=n_top)
