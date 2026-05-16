"""Reference implementations of square-root mod p algorithms.

Two classical cryptographer's algorithms:
  - Tonelli-Shanks: iterative, exploits the 2-adic structure of (Z/p)*.
  - Cipolla: algebraic, computes in the quadratic extension F_p[T]/(T^2 - a).

Both return the canonical root in [0, (p-1)/2] when one exists.
"""

import random
from typing import Optional


def legendre_symbol(a: int, p: int) -> int:
    a %= p
    if a == 0:
        return 0
    return 1 if pow(a, (p - 1) // 2, p) == 1 else -1


def is_quadratic_residue(a: int, p: int) -> bool:
    return legendre_symbol(a, p) == 1 or a % p == 0


def canonical_root(x: int, p: int) -> int:
    x %= p
    return min(x, p - x)


def tonelli_shanks(n: int, p: int) -> Optional[int]:
    n %= p
    if n == 0:
        return 0
    if legendre_symbol(n, p) != 1:
        return None
    if p % 4 == 3:
        return canonical_root(pow(n, (p + 1) // 4, p), p)

    # Factor p-1 = 2^s * q with q odd.
    s, q = 0, p - 1
    while q % 2 == 0:
        s += 1
        q //= 2

    # Find a non-residue z.
    z = 2
    while legendre_symbol(z, p) != -1:
        z += 1

    m = s
    c = pow(z, q, p)
    t = pow(n, q, p)
    r = pow(n, (q + 1) // 2, p)

    while True:
        if t == 1:
            return canonical_root(r, p)
        # Find least i, 0 < i < m, such that t^(2^i) = 1.
        i, tmp = 0, t
        while tmp != 1:
            tmp = (tmp * tmp) % p
            i += 1
        b = pow(c, 1 << (m - i - 1), p)
        m = i
        c = (b * b) % p
        t = (t * c) % p
        r = (r * b) % p


def cipolla(n: int, p: int, rng: Optional[random.Random] = None) -> Optional[int]:
    n %= p
    if n == 0:
        return 0
    if legendre_symbol(n, p) != 1:
        return None

    rng = rng or random.Random(0)

    # Find a such that a^2 - n is a non-residue.
    while True:
        a = rng.randrange(p)
        w = (a * a - n) % p
        if legendre_symbol(w, p) == -1:
            break

    # Compute (a + sqrt(w))^((p+1)/2) in F_p[T]/(T^2 - w).
    # Element is (x, y) representing x + y*T.
    def mul(u, v):
        x1, y1 = u
        x2, y2 = v
        return ((x1 * x2 + y1 * y2 * w) % p, (x1 * y2 + y1 * x2) % p)

    result = (1, 0)
    base = (a, 1)
    e = (p + 1) // 2
    while e:
        if e & 1:
            result = mul(result, base)
        base = mul(base, base)
        e >>= 1

    x, y = result
    assert y == 0, "Cipolla result should land in F_p"
    return canonical_root(x, p)


if __name__ == "__main__":
    # Sanity check: both algorithms should agree on all QRs mod 113.
    p = 113
    rng = random.Random(0)
    for n in range(p):
        ts = tonelli_shanks(n, p)
        cp = cipolla(n, p, rng)
        if ts is None:
            assert cp is None, f"Disagreement on n={n}"
            continue
        assert (ts * ts) % p == n, f"Tonelli-Shanks wrong for n={n}: got {ts}"
        assert (cp * cp) % p == n, f"Cipolla wrong for n={n}: got {cp}"
        assert ts == cp, f"Algorithms disagree for n={n}: TS={ts}, CP={cp}"
    print(f"OK: Tonelli-Shanks and Cipolla agree on all {p} elements of Z/{p}.")
