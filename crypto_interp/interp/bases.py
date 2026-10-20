"""Character bases for the cyclic groups the tasks live on.

The bases themselves are built by :class:`crypto_interp.data.CyclicStructure`
(``ds.structure.char_basis()``): the Fourier basis of Z/n pulled back to the
group's tokens through their coordinates. For multiplication mod p that is the
multiplicative Fourier basis of (Z/p)*, through the discrete log; for addition
mod n it is the ordinary Fourier basis.

This module keeps the prime-specific entry points (:func:`char_index`,
:func:`multiplicative_fourier_basis`, :func:`fold_frequency`) and the
projection helpers.
"""

import torch

from ..data.structure import (
    CharIndex,
    CyclicStructure,
    additive_fourier_basis,
    discrete_log_table,
    primitive_root,
)

__all__ = [
    "CharIndex",
    "additive_fourier_basis",
    "char_index",
    "discrete_log_table",
    "fold_frequency",
    "multiplicative_fourier_basis",
    "primitive_root",
    "project_1d",
    "project_2d",
]


def multiplicative_fourier_basis(
    p: int, device: str = "cpu"
) -> tuple[torch.Tensor, list[str], int]:
    """``(basis, names, g)``: the (p, p) character basis of (Z/p)* with a delta_0 row first.

    See :meth:`CyclicStructure.char_basis`.
    """
    s = CyclicStructure.multiplicative(p)
    basis, ci = s.char_basis(device=device)
    return basis, ci.names, s.generator


# ---------- Projection ----------


def project_1d(tensor: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """Project a (p, ...) tensor onto a basis of shape (p, p), along dim 0.

    Returns a (p, ...) tensor whose row k is the basis-k coefficient.
    """
    # basis: (p_basis, p) ; tensor: (p, ...) → coef: (p_basis, ...)
    return torch.einsum("kp,p...->k...", basis.to(tensor.dtype), tensor)


def project_2d(
    tensor_pp: torch.Tensor,
    basis_x: torch.Tensor,
    basis_y: torch.Tensor | None = None,
) -> torch.Tensor:
    """Project a (p, p, ...) tensor onto a 2D basis. Default: same basis on both axes."""
    if basis_y is None:
        basis_y = basis_x
    return torch.einsum(
        "kp,lq,pq...->kl...",
        basis_x.to(tensor_pp.dtype),
        basis_y.to(tensor_pp.dtype),
        tensor_pp,
    )


def char_index(p: int, device: str = "cpu") -> tuple[torch.Tensor, CharIndex]:
    """``(basis, CharIndex)`` for (Z/p)*: ``CyclicStructure.multiplicative(p).char_basis()``."""
    return CyclicStructure.multiplicative(p).char_basis(device=device)


def fold_frequency(f: int, p: int) -> int:
    """Real-basis index of complex frequency ``f`` of (Z/p)*; see :meth:`CyclicStructure.fold`."""
    n = p - 1
    f %= n
    return min(f, n - f)
