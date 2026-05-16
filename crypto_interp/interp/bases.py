"""Bases for analyzing activations on Z/p × Z/p input grids.

Two bases of interest for modular-arithmetic tasks:

  - ADDITIVE Fourier basis on Z/p: characters of (Z/p, +). Used by Nanda for
    modular addition. Basis vectors are:
        const = 1/sqrt(p) * [1, 1, ..., 1]
        cos_k = sqrt(2/p) * [cos(2πk·x/p) for x in 0..p-1]    for k=1..(p-1)/2
        sin_k = sqrt(2/p) * [sin(2πk·x/p) for x in 0..p-1]    for k=1..(p-1)/2

  - MULTIPLICATIVE Fourier basis on (Z/p)*: characters of (Z/p)*, ≅ Z/(p-1).
    Computed by taking the discrete log w.r.t. a primitive root g, then doing
    additive Fourier analysis on Z/(p-1). Indexed over a..p-1; we leave the
    a=0 row separate because 0 is not in (Z/p)*.

The two bases give very different sparsity patterns. For modular addition,
activations are sparse in the additive basis. For modular multiplication, we
predict sparsity in the multiplicative basis.
"""

import torch


# ---------- Additive Fourier basis on Z/p ----------

def additive_fourier_basis(n: int, device: str = "cpu") -> tuple[torch.Tensor, list[str]]:
    """Returns (basis, names) where basis is an (n, n) orthonormal matrix
    on R^n indexed over Z/n.

    Row 0 is the constant. For k = 1, ..., (n-1)//2 we include the pair
    (cos_k, sin_k). For even n we add one extra cos at k = n/2 (the sin at
    that frequency is identically zero).
    """
    basis = []
    names = []
    x = torch.arange(n, device=device, dtype=torch.float64)
    basis.append(torch.ones(n, device=device, dtype=torch.float64) / (n ** 0.5))
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


# ---------- Multiplicative Fourier basis on (Z/p)* ----------

def primitive_root(p: int) -> int:
    """Find the smallest primitive root g of (Z/p)*."""
    factors = _factorize(p - 1)
    for g in range(2, p):
        if all(pow(g, (p - 1) // q, p) != 1 for q in factors):
            return g
    raise RuntimeError(f"No primitive root found for {p}")


def _factorize(n: int) -> list[int]:
    out = []
    d = 2
    while d * d <= n:
        if n % d == 0:
            out.append(d)
            while n % d == 0:
                n //= d
        d += 1
    if n > 1:
        out.append(n)
    return out


def discrete_log_table(p: int) -> tuple[int, dict[int, int]]:
    """Return (g, dlog) where g is a primitive root and dlog[a] = k s.t. g^k ≡ a mod p,
    for a in 1..p-1. (a=0 is undefined.)
    """
    g = primitive_root(p)
    dlog = {}
    val = 1
    for k in range(p - 1):
        dlog[val] = k
        val = (val * g) % p
    return g, dlog


def multiplicative_fourier_basis(p: int, device: str = "cpu") -> tuple[torch.Tensor, list[str], int]:
    """Build a basis for functions on Z/p indexed via the multiplicative group.

    The basis we return is for vectors of length p, but each basis vector is
    only nonzero on indices a ∈ (Z/p)* = {1, ..., p-1} and zero on a=0. We
    explicitly include a "delta_0" row to handle the a=0 case.

    Layout: row 0 is delta_0 (1 at index 0, 0 elsewhere). Rows 1..p-1 are the
    additive Fourier basis on Z/(p-1) pulled back through the discrete log:
    basis_k(a) = (additive_basis_k(dlog(a))) for a != 0, and 0 at a = 0.

    The returned basis is orthonormal w.r.t. the standard inner product on R^p.
    """
    g, dlog = discrete_log_table(p)
    n = p - 1  # size of (Z/p)*

    # First build additive basis on Z/n.
    add_basis_n, add_names_n = additive_fourier_basis(n, device=device)  # (n, n)

    # Pull each additive basis vector back to a vector in R^p indexed by a.
    # We want b_k(a) = add_basis_n[k, dlog(a)] for a != 0, and 0 for a = 0.
    # Construct an indexing tensor of length p: idx[a] = dlog(a) for a != 0,
    # and we'll place 0 at index a=0.
    pulled = torch.zeros(n, p, device=device, dtype=torch.float64)
    for a in range(1, p):
        pulled[:, a] = add_basis_n[:, dlog[a]]
    # pulled[k, a] is now the multiplicative-character k evaluated at a, for a in 1..p-1.

    # Each pulled[k] vector has L2 norm ≤ 1; we need to renormalize on R^p (since
    # add_basis_n was orthonormal on R^n, but pulled[k] now has p-1 nonzeros).
    norms = pulled.norm(dim=1, keepdim=True)
    norms = torch.where(norms > 0, norms, torch.ones_like(norms))
    pulled = pulled / norms

    # Add delta_0 as the final row.
    delta0 = torch.zeros(p, device=device, dtype=torch.float64)
    delta0[0] = 1.0

    basis = torch.cat([delta0[None, :], pulled], dim=0)  # (n+1, p) = (p, p)
    names = ["delta_0"] + [f"mul {nm}" for nm in add_names_n]
    return basis, names, g


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
        basis_x.to(tensor_pp.dtype), basis_y.to(tensor_pp.dtype), tensor_pp,
    )


if __name__ == "__main__":
    p = 113
    add_basis, add_names = additive_fourier_basis(p)
    mul_basis, mul_names, g = multiplicative_fourier_basis(p)

    print(f"Additive basis shape: {add_basis.shape}")
    add_orth = add_basis @ add_basis.T
    print(f"  orthonormality error: {(add_orth - torch.eye(p, dtype=torch.float64)).abs().max().item():.3e}")

    print(f"\nMultiplicative basis (p={p}, primitive root g={g}):")
    print(f"  shape: {mul_basis.shape}")
    mul_orth = mul_basis @ mul_basis.T
    print(f"  orthonormality error: {(mul_orth - torch.eye(p, dtype=torch.float64)).abs().max().item():.3e}")
    print(f"  first 5 names: {mul_names[:5]}")
