# Population aggregation — replication across seeds

**Strategy 6.** Which findings are *universal* (the algorithm) vs
*idiosyncratic* (one seed's quirk)? Distinguishes paper-worthy claims from
single-seed accidents.

## Algebraic + mech-interp anchor

Aggregating across replicas at the same task and architecture to test
universality. For our setting, the questions are:

- Is the 3-primary backbone universal, or do some seeds use different
  numbers?
- Is the ×2-only doubling economy universal, or do other multipliers appear?
- Are the helpers zero-output-neuron-cost in every seed?
- Are de-grokking, basin-commitment timings, etc., consistent across seeds?

## Verbs

There's no `Session.from_runs(...)` constructor yet (deliberately — we want
to see the use cases before designing it). Instead, iterate `Session.from_run`
and aggregate Python-side.

The aggregation utilities live in the standard library — `collections.Counter`
for multipliers and frequencies, `numpy` for means / quantiles.

## Recipe — the 21-seed cost-atom population

```python
import glob
from collections import Counter
import numpy as np
from crypto_interp.interp import Session

rows = []          # one per (seed, primary): (seed, k, order, neurons)
helper_mults = []  # all detected helper multipliers
prim_counts = []   # one per seed: number of primaries

for d in sorted(glob.glob("experiments/003_dmodel_sweep_p113/runs/dmodel_24_dmlp_20_wd2_seed*")):
    seed = int(d.split("seed")[-1])
    try: S = Session.from_run(d)
    except Exception: continue
    if S.evaluate()[1] > 1e-3: continue                  # grokked only

    K = S.essential()["K"]
    helpers = {h: (m, mult) for (h, m, mult, _e) in S.helpers(K)}
    prims = [k for k in K if k not in helpers]
    sizes = S.cluster_sizes(K)

    prim_counts.append(len(prims))
    for k in prims:
        rows.append((seed, k, S.order(k), sizes[k]))
    helper_mults.extend(mult for (_, mult) in helpers.values() if mult)

print(f"#primaries distribution: {dict(Counter(prim_counts))}")
print(f"helper multipliers: {dict(Counter(helper_mults))}")
print(f"primary cluster sizes (mean ± std): "
      f"{np.mean([r[3] for r in rows]):.1f} ± {np.std([r[3] for r in rows]):.1f}")
```

Expected on the 21-seed wd=2 d_mlp=20 population: 6/8 groks use 3 primaries,
2/8 use 2; **all 9/9 helpers have multiplier ×2** (the doubling-only economy);
mean primary cluster size ~7 neurons (≈ d_mlp / #primaries).

## Common variations

- **Stability check (the de-grokking population view).** Compare
  `min(test_losses)` to `final(test_losses)` per seed; flag runs where
  min ≪ final ("ever-grok but de-grokked"). Re-classify the population
  by `ever-grok` vs `stable-grok`.
- **Cluster verification across seeds.** For each grokked seed, correlate
  the primary χ_k cluster signal against the algebraic reference; report
  the population distribution of correlations. Universality of the
  character-product mechanism.
- **Cliff time vs basin strength.** Across seeds, compute the K/non-K
  ratio at a fixed early epoch (e.g. 1500) and correlate with cliff time;
  the predictor described in `notes/04` (ρ ≈ -0.92).
