"""Cryptographer's reference algorithms for each task.

Used to (a) generate ground-truth labels for datasets and (b) compare against
the algorithm the model learns. Each module is a faithful implementation of
the standard textbook algorithm(s) for one task.
"""

from .sqrt import (
    canonical_root,
    cipolla,
    is_quadratic_residue,
    legendre_symbol,
    tonelli_shanks,
)

__all__ = [
    "legendre_symbol",
    "is_quadratic_residue",
    "canonical_root",
    "tonelli_shanks",
    "cipolla",
]
