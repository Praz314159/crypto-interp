# Playbooks

Canonical recipes for the seven mechanistic-interpretability strategies we use
to study learned algorithms in computational number theory. Each playbook is
a short markdown file with a problem statement, the 5–15-line snippet of
library verbs that answers it, the algebraic-structure anchor, and common
variations.

The playbooks are the **shared idioms** between the researcher (working in
a notebook) and Claude (working in a turn). Both surfaces compose the same
verbs from `crypto_interp.interp`.

## The six-layer scaffolding

| Layer | Module | What it gives you |
|---|---|---|
| Model + hook points | `crypto_interp.models.transformer` | Tiny 1-block no-LN transformer with named `HookPoint`s at every internal activation. |
| Cache | `crypto_interp.interp.cache` | `ActivationCache` — short-name aliases (`mlp_post`, `attn_out`, …), `.final()` / `.grid()` accessors. |
| Domain primitives | `crypto_interp.interp.{bases,metrics,ablate,harmonic,dynamics,neurons}` | Characters, Fourier bases, helper detection, dynamics markers, per-neuron analysis. The "verbs" of computational number theory. |
| Interventions | `crypto_interp.interp.{hooks,interventions,patching}` | `run_with_hooks` / `hooks(...)` context manager / `weight_patch` / `ablate_char_w` / `act_patch`. Causal probes. |
| Session | `crypto_interp.interp.session` | Bundles `(model, ds, basis, ci)`; lazy `.cache`; one-line passthroughs. The entry point. |
| Playbooks | `crypto_interp.playbooks` | This directory. Documented recipes. |

## Two layers of structure, both first-class

Every playbook is anchored in two things:

- **Classical algebra (the substrate).** For a given task, the natural objects
  are groups, characters, subgroup lattices — `(Z/p, +)` and additive Fourier
  for `add`; `(Z/p)*` and multiplicative / Dirichlet characters for `mul`.
  This is what determines which basis we project onto, which subgroup chains
  to check.
- **Mech-interp structure (what lives on top).** Capacity-constrained
  representational economy is not classical — we found it. The doubling-pair
  cluster `{χ_k, χ_{2k}}`, the cost atom (neurons ≈ budget/#primaries), free-
  rider helpers via ReLU's `a₂ ≠ 0` — these are the units of the learned
  algorithm. Playbooks make them first-class verbs alongside the algebra.

## The seven playbooks (by strategy)

| # | Playbook | Strategy |
|---|---|---|
| 1 | `basis_discovery.md` | Find the natural basis (Strategy 1). |
| 2 | `mechanism_verification.md` | Match circuit to algorithm (Strategy 4). |
| 3 | `causal_intervention.md` | Weight ablation + activation patching (Strategy 3). |
| 4 | `economy_analysis.md` | Capacity-constrained simplification (Strategy 7). |
| 5 | `grokking_dynamics.md` | Trajectory through training (Strategy 5). |
| 6 | `population_aggregation.md` | Replication across seeds (Strategy 6). |
| 7 | `lattice_variation.md` | Varying p−1's prime decomposition (the first experimental program). |

## The canonical entry pattern

Every playbook starts from this:

```python
from crypto_interp.interp import Session
S = Session.from_run("experiments/003_dmodel_sweep_p113/runs/dmodel_24_dmlp_20_wd2_seed1")
# S.model, S.ds, S.basis, S.ci ready; S.cache and S.logits_grid lazy.
```

From here, every playbook is a few lines of `S.<verb>(…)`.
