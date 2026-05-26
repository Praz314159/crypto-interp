# Mechanism verification — match the circuit to an algorithm

**Strategy 4.** Does a learned neuron cluster *compute* an identifiable
algebraic operation, or does it just look structured? The move from
"correlated with χ_k" to "implements χ_k(a)·χ_k(b)".

## Algebraic anchor

The **character product formula**: for any character χ of an abelian group,
`χ_k(ab) = χ_k(a) · χ_k(b)`. In real-basis form on `(Z/p)*`, the cluster
output as a function of (a, b) should be `cos(θ_k(a) + θ_k(b))` where
`θ_k(a) = 2πk · dlog(a) / (p−1)`. This is the **reference signal** — a
ground-truth (a, b)-grid we can correlate the learned cluster against.

## Verbs

- `S.per_neuron_dominant_char()` → `(char_E[d_mlp, n_chars], dom[d_mlp])`.
  Assigns each neuron the character it most writes to (via `W_U @ W_out`).
- `S.cluster_signal(k)` → `(p−1, p−1)` array: the residual-stream signal
  contributed by the χ_k cluster, aligned with the unembed's χ_k direction.
- `S.reference_signal(k)` → `(p−1, p−1)` array: the algebraic ground truth
  `cos(θ_k(a) + θ_k(b))`.
- `correlate(sig, ref)` — Pearson correlation of two (a, b)-grids.

## Recipe

```python
from crypto_interp.interp import Session, correlate

S = Session.from_run("experiments/003_dmodel_sweep_p113/runs/dmodel_24_dmlp_20_wd2_seed1")
K = S.essential()["K"]
sizes = S.cluster_sizes(K)

for k in K:
    if sizes[k] == 0:
        print(f"χ{k}: no dedicated cluster (helper)")
        continue
    sig = S.cluster_signal(k); ref = S.reference_signal(k)
    print(f"χ{k}: cluster (n={sizes[k]}) vs reference  corr = {correlate(sig, ref):+.3f}")
```

Expected on seed 1: primaries `χ_10` (8 neurons), `χ_33` (6), `χ_52` (6)
all correlate ≥ 0.94 with the reference; helpers `χ_8` and `χ_46` report
"no dedicated cluster" (the free-rider story — they ride on a primary's
cluster). That's the "circuit IS character product" claim.

## Common variations

- **Inverted readout (seed-7 pattern).** When a *primary* shows 0 neurons,
  check whether it's the base of a doubling pair whose double dominates the
  output readout. Iterate `cluster_sizes` and look for primaries with size 0
  that satisfy `fold(2·k_base) = k_dominant_neighbor`.
- **Reconstruct the full logit contribution.** Sum cluster signals over K
  and compare to `S.logits_grid` (after appropriate scaling/centering) for
  end-to-end mechanism reconstruction.
- **Negative control.** Compare the cluster's correlation against a *random*
  reference (a different k); should be near zero. Confirms the high
  correlation is genuine and not generic structure.
