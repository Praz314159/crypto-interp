"""The cyclic group a task is defined on, and its characters.

The group tasks compute a group operation on tokens:

- ``add`` mod n: the group ℤ/n; token t is the element t.
- ``mul`` and ``mul_units`` mod p: the group (ℤ/p)*, cyclic of order p − 1.
  Token t ≠ 0 is the element x(t) = dlog_g(t) of ℤ/(p − 1), for g the smallest
  primitive root; token 0 is not in the group.

Everything the analyses need follows from three things, held by
:class:`CyclicStructure`: the group order n, the tokens that are group
elements, and each one's coordinate x ∈ ℤ/n. The label of (a, b) is the token
with coordinate x(a) + x(b). The characters are χ_k(t) = exp(2πi k x(t) / n),
in real form the angle θ_k(t) = 2πk·x(t)/n, and χ_k has order n / gcd(k, n).

Multiplication mod p is therefore addition mod p − 1 with the tokens renamed
by the discrete log, and an analysis written against the structure applies to
both.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from functools import cached_property

import numpy as np
import torch

# ---------- discrete logarithms in (Z/p)* ----------


def _prime_factors(n: int) -> list[int]:
    out, d = [], 2
    while d * d <= n:
        if n % d == 0:
            out.append(d)
            while n % d == 0:
                n //= d
        d += 1
    if n > 1:
        out.append(n)
    return out


def primitive_root(p: int) -> int:
    """The smallest primitive root g of (Z/p)*."""
    factors = _prime_factors(p - 1)
    for g in range(2, p):
        if all(pow(g, (p - 1) // q, p) != 1 for q in factors):
            return g
    raise RuntimeError(f"No primitive root found for {p}")


def discrete_log_table(p: int) -> tuple[int, dict[int, int]]:
    """``(g, dlog)``: g a primitive root and dlog[a] = k with g^k ≡ a mod p, for a in 1..p-1."""
    g = primitive_root(p)
    dlog = {}
    val = 1
    for k in range(p - 1):
        dlog[val] = k
        val = (val * g) % p
    return g, dlog


# ---------- the Fourier basis of Z/n ----------


def additive_fourier_basis(n: int, device: str = "cpu") -> tuple[torch.Tensor, list[str]]:
    """``(basis, names)``: an (n, n) orthonormal basis of functions on Z/n.

    Row 0 is the constant. For k = 1, ..., (n-1)//2 we include the pair
    (cos_k, sin_k). For even n we add one extra cos at k = n/2 (the sin at
    that frequency is identically zero).
    """
    basis = []
    names = []
    x = torch.arange(n, device=device, dtype=torch.float64)
    basis.append(torch.ones(n, device=device, dtype=torch.float64) / (n**0.5))
    names.append("const")
    for k in range(1, (n - 1) // 2 + 1):
        c = torch.cos(2 * torch.pi * x * k / n)
        s = torch.sin(2 * torch.pi * x * k / n)
        basis.append(c / c.norm())
        basis.append(s / s.norm())
        names.append(f"cos {k}")
        names.append(f"sin {k}")
    if n % 2 == 0:
        c = torch.cos(2 * torch.pi * x * (n // 2) / n)  # alternating ±1
        basis.append(c / c.norm())
        names.append(f"cos {n // 2}")
    return torch.stack(basis, dim=0), names


@dataclass(frozen=True)
class CharIndex:
    """Map from character k to its rows in a structure's character basis.

    ``cos[k]`` and ``sin[k]`` are the rows of the real pair for χ_k (the
    Nyquist character k = n/2 has a cos row but no sin row). ``n`` is the
    group order; ``generator`` is the primitive root for multiplicative groups.
    """

    n: int
    generator: int | None
    names: list[str]
    by_char: dict[int, list[int]]  # k -> [basis row indices] (cos and sin)
    cos: dict[int, int]  # k -> cos row index
    sin: dict[int, int]  # k -> sin row index (absent for k = n/2)
    freqs: list[int]  # sorted character frequencies present


# ---------- the structure ----------


@dataclass(frozen=True)
class CyclicStructure:
    """A cyclic group of order ``n`` realised on value tokens 0..n_values−1.

    ``tokens`` are the value tokens that are group elements, ascending, and
    ``coords`` their coordinates in Z/n. Value tokens outside the group (token
    0 for the multiplicative tasks) are never inputs to the group grid.
    """

    n: int
    n_values: int
    tokens: tuple[int, ...]
    coords: tuple[int, ...]
    op: str  # "+" or "×", for labels
    generator: int | None = None

    @classmethod
    def additive(cls, n: int) -> CyclicStructure:
        """Z/n on tokens 0..n−1: token t is the element t."""
        return cls(n=n, n_values=n, tokens=tuple(range(n)), coords=tuple(range(n)), op="+")

    @classmethod
    def multiplicative(cls, p: int) -> CyclicStructure:
        """(Z/p)* on tokens 1..p−1 of the values 0..p−1: token t is the element dlog(t)."""
        g, dlog = discrete_log_table(p)
        return cls(
            n=p - 1,
            n_values=p,
            tokens=tuple(range(1, p)),
            coords=tuple(dlog[t] for t in range(1, p)),
            op="×",
            generator=g,
        )

    # ---- coordinates ----

    @cached_property
    def x(self) -> np.ndarray:
        """(n,) coordinate of each group token, in token order."""
        return np.array(self.coords, dtype=np.int64)

    @cached_property
    def token_array(self) -> np.ndarray:
        """(n,) the group tokens, ascending."""
        return np.array(self.tokens, dtype=np.int64)

    @cached_property
    def position(self) -> np.ndarray:
        """(n_values,) each value token's index among the group tokens; −1 outside the group."""
        out = np.full(self.n_values, -1, dtype=np.int64)
        out[self.token_array] = np.arange(self.n)
        return out

    @cached_property
    def token_at(self) -> np.ndarray:
        """(n,) ``token_at[x]`` is the group token with coordinate x."""
        out = np.empty(self.n, dtype=np.int64)
        out[self.x] = self.token_array
        return out

    @property
    def coord_name(self) -> str:
        """What the coordinate is called in labels: "dlog" for (Z/p)*, "x" for Z/n."""
        return "dlog" if self.op == "×" else "x"

    @cached_property
    def coord_order(self) -> np.ndarray:
        """(n,) indices that sort the group tokens by coordinate (dlog order for mul)."""
        return np.argsort(self.x, kind="stable")

    def product(self) -> np.ndarray:
        """(n, n) the token of a·b for the i-th and j-th group tokens: the label grid."""
        return self.token_at[(self.x[:, None] + self.x[None, :]) % self.n]

    def offsets(self) -> np.ndarray:
        """(n, n, n) ``(x(a) + x(b) − x(c)) mod n`` over group tokens a, b, c."""
        x = self.x
        return (x[:, None, None] + x[None, :, None] - x[None, None, :]) % self.n

    # ---- characters ----

    @property
    def freqs(self) -> range:
        """The real characters' frequencies, 1..⌊n/2⌋ (χ_k and χ_{n−k} are one real pair)."""
        return range(1, self.n // 2 + 1)

    def theta(self, k: int) -> np.ndarray:
        """(n,) the angle θ_k of each group token, in token order."""
        return 2 * np.pi * k * self.x / self.n

    def order(self, k: int) -> int:
        """The order of χ_k: n / gcd(k, n)."""
        return self.n // math.gcd(int(k), self.n)

    def fold(self, f: int) -> int:
        """Real-basis index of complex frequency f: ``min(f, n − f)`` after reducing mod n.

        The cos/sin basis cannot tell f from n − f, so use this when relating
        products in the character group (where indices add and can exceed n/2)
        back to the indices the analyses report.
        """
        f %= self.n
        return min(f, self.n - f)

    def char_basis(self, device: str = "cpu") -> tuple[torch.Tensor, CharIndex]:
        """``(basis, CharIndex)``: an orthonormal basis of R^n_values by characters.

        The Fourier basis of Z/n pulled back to the group tokens through their
        coordinates, plus one delta row for each value token outside the group,
        first. Double precision.
        """
        add_basis, add_names = additive_fourier_basis(self.n, device=device)
        pulled = torch.zeros(self.n, self.n_values, device=device, dtype=torch.float64)
        for t, x in zip(self.tokens, self.coords, strict=True):
            pulled[:, t] = add_basis[:, x]
        norms = pulled.norm(dim=1, keepdim=True)
        pulled = pulled / torch.where(norms > 0, norms, torch.ones_like(norms))
        outside = [t for t in range(self.n_values) if t not in set(self.tokens)]
        deltas = torch.zeros(len(outside), self.n_values, device=device, dtype=torch.float64)
        for i, t in enumerate(outside):
            deltas[i, t] = 1.0
        basis = torch.cat([deltas, pulled], dim=0)
        names = [f"delta_{t}" for t in outside] + [f"char {nm}" for nm in add_names]

        by_char: dict[int, list[int]] = {}
        cos: dict[int, int] = {}
        sin: dict[int, int] = {}
        for i, nm in enumerate(names):
            m = re.match(r"char (cos|sin) (\d+)", nm)
            if m:
                k = int(m.group(2))
                by_char.setdefault(k, []).append(i)
                (cos if m.group(1) == "cos" else sin)[k] = i
        ci = CharIndex(
            n=self.n,
            generator=self.generator,
            names=names,
            by_char=by_char,
            cos=cos,
            sin=sin,
            freqs=sorted(by_char),
        )
        return basis.double(), ci
