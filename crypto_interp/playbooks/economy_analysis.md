# Economy analysis — capacity-constrained simplification

**Strategy 7.** Under tight capacity, *how does the network economize* the
representation? This is our distinctive contribution — the structural unit
of the economy isn't classical algebra; it's the **mech-interp doubling-pair
cluster** we discovered.

## The mech-interp anchor (what's first-class here)

A *doubling pair* `{χ_k, χ_{2k}}` is the unit of allocation. ReLU's second
Fourier harmonic — coefficient `a₂ = 2/3π ≈ 0.21`, while `a₃ = 0` exactly —
generates `χ_{2k}` from `χ_k` for free at the output. So one neuron cluster
serves both characters: χ_k as the fundamental, χ_{2k} as the ReLU-square
byproduct.

That is **not a classical algebra object**. It's the building block of how a
trained, budget-limited net allocates neurons across characters. Three
measured laws on top of it:

1. **Doubling-only economy.** Helpers always have multiplier ×2, never ×3
   (matches `a₂ ≠ 0 / a₃ = 0`). 9/9 helpers across 8 grokked seeds.
2. **Zero output-neuron cost.** Helpers' dedicated neuron count is 0; their
   energy rides on the primary's cluster. Confirmed across budgets.
3. **Cost atom.** Cost per primary ≈ `d_mlp / #primaries`, not ∝ order.
   Order-2 (Legendre) gets a ~6.6× discount; other orders are flat.

## Verbs

- `S.essential(threshold=0.05)` — K and per-character essentialness.
- `S.helpers(K)` — `[(helper_k, primary_m, mult, energy), ...]` via
  `find_primary_helper_pairs`. The `mult` is the multiplier in `χ_helper`
  = ×mult of `χ_primary`.
- `S.cluster_sizes(K)` — `{k → n_neurons_dominant_in_χ_k}`. Zero for
  helpers.
- `S.per_neuron_dominant_char()` — full `(char_E, dom)` matrices.

## Recipe — the economy on one model

```python
from crypto_interp.interp import Session

S = Session.from_run("experiments/003_dmodel_sweep_p113/runs/dmodel_24_dmlp_20_wd2_seed1")
K = S.essential()["K"]
pairs = S.helpers(K)               # [(helper, primary, mult, energy), ...]
sizes = S.cluster_sizes(K)

print(f"K = {K}   orders = {[S.order(k) for k in K]}")
print(f"helpers: {[(h, m, mult) for (h, m, mult, _) in pairs]}")
print(f"cluster sizes: {sizes}    (helpers should be 0)")

helpers = {h for (h, _m, _mt, _e) in pairs}
prims = [k for k in K if k not in helpers]
print(f"\nbackbone: {len(prims)} primaries; {len(helpers)} free-rider helpers")
```

Expected on seed 1: K = `[8, 10, 33, 46, 52]`; helpers `(8, 52, 2)` and
`(46, 33, 2)` — both ×2. Sizes: `{8:0, 10:8, 33:6, 46:0, 52:6}`. **3-primary
backbone + 2 doubling helpers, helpers cost 0 output neurons.**

## Recipe — the cost atom across budgets

```python
from crypto_interp.interp import Session, char_index
import glob
from collections import Counter

rows = []
for d in sorted(glob.glob("experiments/003_dmodel_sweep_p113/runs/dmodel_24*")):
    try: S = Session.from_run(d)
    except Exception: continue
    if S.evaluate()[1] > 1e-3: continue       # grokked only
    K = S.essential()["K"]
    helpers = {h for (h, *_) in S.helpers(K)}
    prims = [k for k in K if k not in helpers]
    _, dom = S.per_neuron_dominant_char()
    cnt = Counter(dom.tolist())
    for k in prims:
        rows.append((S.model.cfg.d_mlp, k, S.order(k), int(cnt.get(k, 0))))

# rows: (d_mlp, k, order(k), neurons_per_primary)
# Aggregate → cost-atom plot: x=d_mlp, y=neurons; color/marker by order(k).
# Expect mean(neurons | d_mlp) ≈ d_mlp / #primaries; order-2 ≈ 6.6× cheaper.
```

## Common variations

- **Helper multiplier histogram.** Across many seeds:
  `Counter(mult for (_h, _m, mult, _e) in S.helpers(K) if mult)`. Expect
  all 2 for ReLU; if a 3 appears, that's the open Sylow-3 question.
- **Population zero-neuron check.** For each grokked seed, count
  `sizes[h]` for each helper `h ∈ pairs`. Should be 0 for every seed; any
  exception is a seed-7-style readout inversion worth chasing.
- **Order-2 (Legendre) discount.** Filter `prims` by `S.order(k) == 2`
  and compare cluster size to non-order-2 primaries at the same budget.
  Expect ~6× smaller.
