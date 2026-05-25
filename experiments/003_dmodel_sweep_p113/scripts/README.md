# Experiment 003 — scripts

Scripts that **define and run** this experiment (sweeps at p=113, d_model=24).
The **canonical, reusable analyses** no longer live here — they moved to the
library at **`crypto_interp/analysis/`** and run against *any* experiment's runs:

```
python -m crypto_interp.analysis.<name> --run-dir <experiment>/runs/<run>
```

So a new experiment (e.g. p=127) holds only `config.py`, a driver, and `runs/`;
every analysis is invoked from the library — no copying scripts in.

## Canonical analyses (now in `crypto_interp/analysis/`, run via `python -m`)

| Module | Args | Backs |
|---|---|---|
| `ablation_delta` | `--run-dir [--ks]` | note 06 |
| `delta_spectrum` | `--run-dir [--ks]` | note 06 |
| `neuron_clusters` | `--run-dir` | note 05 |
| `fourier2d` | `--run-dir --tag [--out-dir]` | note 06 |
| `ablation_full` | `--run-dir [--out-dir]` | note 06 |
| `vestigial_ablation` | `--run-dir` | note 06 |
| `dmlp_sweep` | `--runs-dir [--out-dir]` | Exp 1 |
| `bifurcation` | `--data-dir [--out-dir]` | note 04 |
| `cliff_vs_commit` | `--runs-dir --data-dir [--out-dir]` | note 04 |
| `check_grokking` | `--runs-dir` | Exp 1 failure detector |
| `verify_helper_mechanism` | `[--out-dir]` | note 06 §7.4 |

Library primitives they build on: `crypto_interp.interp` (`char_index`,
`char_energy`, `compute_logits_grid`, `ablate_character`, `essential_characters`,
`reduce_to_diff`, `cliff_step`/`bifurcation_step`/`grokking_status`,
`delta_k_spectrum`/`find_primary_helper_pairs`). Tests in `tests/`.

## What stays here (experiment-specific)

**Sweep drivers** — define *this* experiment's runs:
`run_sweep.py`, `run_seed_sweep.py`, `run_dmlp_sweep.py`, `run_dmlp_seed_sweep.py`
(the last is the Exp 1 driver: `--d-mlps 24,22,20,18,16`; `--p`/`--experiment` for multi-prime).

**Per-step diagnostics** — train with extra logging to produce `data/`:
`run_fine_grained.py`, `run_fine_grained_exact.py`, `run_freeze_we.py`,
`run_grad_decomp.py`, `run_with_grad.py`. (Still hardcode p=113 in their basis;
migrate to `char_index` lazily when a multi-prime run needs them.)

**Cache / viz**: `build_basis_cache.py` (Pareto/cost, p=113-specific),
`viz_dmlp_grokking.py`, `viz_basis_space.py`, `viz_basis_dynamics.py`,
`viz_basin_commitment.py`, `viz_order_energy_evolution.py`.

**One-off / exploratory** (kept for provenance; promote to `analysis/` if they
become canonical): `analyze_attn_output.py`, `analyze_attn_gating.py`,
`analyze_full_signal.py`, `analyze_intrinsic_signal.py`,
`analyze_orthogonalized_signal.py`, `analyze_kk_decomposition.py`,
`analyze_path_contribution.py`, `analyze_grad_alignment.py`,
`analyze_grad_decomp.py`, `analyze_char_concentration.py`,
`analyze_neuron_packing.py`, `analyze_order_multisets.py`,
`analyze_basis_dynamics.py`, `analyze_WU_WE_orthogonality.py`,
`analyze_bifurcation_dmlp.py`, `analyze_fine_grained.py`.
