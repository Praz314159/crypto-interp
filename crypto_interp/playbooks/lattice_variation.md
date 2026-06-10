# Lattice variation — varying p−1's prime decomposition

**The first experimental program** the harness enables. We've only studied
a narrow slice of lattice structures so far (p = 113, 127, 181). To learn
which algebraic structure the model *actually exploits*, train at primes
deliberately chosen so that p−1 has different prime factorizations, then
apply the standard playbooks (basis discovery, helper detection, cost
atom, mechanism verification) at each.

## The two predictions being tested

1. **CRT minimality.** By CRT, `Z/(p−1) ≅ ∏ Z/p_i^{e_i}`. The prediction:
   for every grokked seed at every prime, K contains at least one character
   whose order is divisible by each prime-power factor `p_i^{e_i}` of p−1.
   (Otherwise the model is missing a basis direction needed to separate
   the function.) This is the **classical-algebra** prediction.
2. **Doubling economy is prime-invariant.** The ×2 helper multipliers
   appear at every prime regardless of the lattice structure, because
   they're a ReLU 2nd-harmonic effect (`a₂ ≠ 0`), not a group-theoretic
   one. This is the **mech-interp** prediction.

Plus an open question: **does lattice complexity drive backbone size**?
Smooth primes (p = 211, four distinct prime factors) might exhibit a
different number of primaries or different order distributions than
2-factor primes.

## Candidate prime set

| Prime | p − 1 factorization | n distinct factors | v₂ | Lattice type |
|---|---|---|---|---|
| 47 | 2·23 | 2 | 1 | safe prime |
| 113 | 2⁴·7 | 2 | 4 | (already studied) |
| 127 | 2·3²·7 | 3 | 1 | (already studied) |
| 181 | 2²·3²·5 | 3 | 2 | (already studied) |
| 193 | 2⁶·3 | 2 | 6 | deep Sylow-2 |
| 211 | 2·3·5·7 | 4 | 1 | smooth |
| 17 *(opt.)* | 2⁴ | 1 | 4 | Fermat prime |

## Verbs

The recipe is "the standard playbooks, applied at each prime." Concretely:

```python
import math
from crypto_interp.interp import Session

def prime_factorize(n):
    out, d = [], 2
    while d * d <= n:
        e = 0
        while n % d == 0:
            n //= d; e += 1
        if e: out.append((d, e))
        d += 1
    if n > 1: out.append((n, 1))
    return out

def crt_minimality_check(S):
    """For each grokked seed, check whether K covers each prime-power factor of p-1."""
    K = S.essential()["K"]
    factors = prime_factorize(S.ds.p - 1)
    orders = [S.order(k) for k in K]
    covers = {f"{q}^{e}": any(o % (q**e) == 0 for o in orders) for q, e in factors}
    return {"K": K, "orders": orders, "factors": factors, "covers": covers,
            "all_covered": all(covers.values())}

# After training: per-prime population
results = {}
for p in [47, 113, 127, 181, 193, 211]:
    seed_results = []
    for d in sorted(glob.glob(f"experiments/00X_p{p}/runs/*")):     # adjust path
        try: S = Session.from_run(d)
        except Exception: continue
        if S.evaluate()[1] > 1e-3: continue
        chk = crt_minimality_check(S)
        helpers = S.helpers(chk["K"])
        seed_results.append({
            "K": chk["K"], "orders": chk["orders"],
            "all_covered": chk["all_covered"],
            "helper_mults": [m for (_h, _m, m, _e) in helpers if m],
        })
    results[p] = seed_results
```

## Reading the results

- **CRT minimality** is supported iff `all_covered` is True for every
  grokked seed at every prime. A single violation falsifies it.
- **Prime-invariant doubling** is supported iff `helper_mults` is `[2, 2,
  ...]` at every prime. A `3` would be the open Sylow-3 signal we'd want
  to chase.
- **Lattice complexity** is exploratory: compare backbone size
  distributions across `n_distinct_factors`. No prior prediction.

## What this experiment will inform

The v2 abstraction. Whether `Task` / `AlgebraicDomain` becomes a rich
dataclass with subgroup-lattice metadata, a simple string registry, or
something in between — the answer is whichever shape the data above
indicates is actually used by the analyses.
