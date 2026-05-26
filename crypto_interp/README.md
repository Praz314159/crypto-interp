# `crypto_interp` — library overview

A small mechanistic-interpretability toolkit for algebraic-number-theory
algorithms learned by tiny transformers. Modeled on TransformerLens but
stripped to the minimum for our domain — modular arithmetic, characters,
Sylow structure — on 1-block no-LN models.

## Where to start

**For "how do I run an experiment":** read [`playbooks/README.md`](playbooks/README.md).
The playbooks are the canonical recipes for each mech-interp strategy
(basis discovery, causal intervention, mechanism verification, economy
analysis, dynamics, population aggregation, lattice variation), each in
5–15 lines of library verbs.

**For "how is the library structured":** the six-layer scaffolding model
(model → cache → primitives → interventions → Session → playbooks) is
documented in `playbooks/README.md`. Browse:

- [`crypto_interp/models/transformer.py`](models/transformer.py) — tiny
  1-block no-LN transformer with named `HookPoint`s at every internal
  activation.
- [`crypto_interp/interp/cache.py`](interp/cache.py) — `ActivationCache`
  with short-name aliases (`mlp_post` ↔ `blocks.0.mlp.hook_post`) and
  `.final` / `.grid` / `.decompose_resid` accessors.
- [`crypto_interp/interp/hooks.py`](interp/hooks.py) — `run_with_hooks` /
  `hooks(...)` context manager / intervention factories.
- [`crypto_interp/interp/interventions.py`](interp/interventions.py) —
  weight-space context managers (`weight_patch`, `ablate_char_w`,
  `freeze_param`).
- [`crypto_interp/interp/patching.py`](interp/patching.py) — activation
  patching (`act_patch`, `patch_mlp_out`, `patch_resid_pre`,
  `patch_attn_out`).
- [`crypto_interp/interp/session.py`](interp/session.py) — the analysis
  bundle (`Session.from_run(...)` is the entry point).
- [`crypto_interp/interp/{bases,metrics,ablate,harmonic,dynamics,neurons}.py`](interp/) —
  the domain primitives: character / Fourier bases, helper detection,
  dynamics markers.

## Quickstart — the cost-atom analysis in seven lines

```python
from crypto_interp.interp import Session, correlate

S = Session.from_run("experiments/003_dmodel_sweep_p113/runs/dmodel_24_dmlp_20_wd2_seed1")
K = S.essential()["K"]                                  # essential characters
helpers = [(h, m, mult) for (h, m, mult, _e) in S.helpers(K)]   # ×2 doubling
sizes = S.cluster_sizes(K)                              # 0 for helpers
sig = S.cluster_signal(K[1]); ref = S.reference_signal(K[1])
print(K, helpers, sizes, correlate(sig, ref))
# [8,10,33,46,52]  [(8,52,2),(46,33,2)]  {8:0,10:8,33:6,46:0,52:6}  0.978
```

## Running canonical analyses from the CLI

The handful of analyses with their own CLI entrypoints live in
`crypto_interp/analysis/`. They are thin wrappers around `Session`:

```bash
python -m crypto_interp.analysis.ablation_full --run-dir <run_dir>
python -m crypto_interp.analysis.neuron_clusters --run-dir <run_dir>
```

Most analyses don't need a CLI; they live in a notebook cell or a Claude
turn that composes the verbs directly. The playbooks document those
patterns.

## Adding a new task

The data layer dispatches by string. To add `modular_exp` (say), drop a
`crypto_interp/data/exp.py` with a `build(p, frac_train, seed) → Dataset`
function and register it in `crypto_interp/data/__init__.py::_TASKS`.
See `crypto_interp/data/add.py` for the minimal example.

The first major experimental program the harness enables is the
**lattice-variation sweep** — train at primes deliberately chosen so
`p−1` has different prime factorizations, and apply the standard playbooks
at each. See [`playbooks/lattice_variation.md`](playbooks/lattice_variation.md).

## Status

Harness v1 (cache + hooks + interventions + patching + Session + playbooks)
landed on branch `harness-v1`. The `Task` / `AlgebraicDomain` abstraction
is deliberately deferred to v2 — it will be designed around the data from
the lattice-variation sweep, not stipulated a priori.
