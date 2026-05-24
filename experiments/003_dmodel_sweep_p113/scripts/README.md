# Experiment 003 — scripts

Analysis/driver scripts for the d_model / d_mlp / seed sweeps at p=113 and the
harmonic-helper investigation. Reusable primitives and analysis methods now live
in the **`crypto_interp.interp`** library (prime-parametric); scripts here are
thin CLIs that call it and handle plotting/IO.

## Library primitives (use these in new scripts)

- `bases`: `char_index(p) -> (basis, CharIndex)`, `multiplicative_fourier_basis`, `discrete_log_table`.
- `metrics`: `char_energy`, `char_energy_batch`, `order_energy`, `order_of`, `correlate`, `find_cliff`, `topk_recall`.
- `grids`: `compute_logits_grid` (W_E-override safe), `compute_activation_grid(hook)`, `ab_grid_inputs`.
- `ablate`: `ablate_character`, `essential_characters`, `project_back`, `evaluate_loss`.
- `reductions`: `reduce_to_diff(grid, p, value_axis=...)` (the canonical Δlog reduction), `fourier_spectrum_1d`, `reduce_to_ab`.
- `dynamics`: `cliff_step`, `bifurcation_step`, `commit_step`, `grokking_status`.
- `harmonic`: `delta_k`, `delta_k_spectrum`, `find_primary_helper_pairs` (the Δ_k / Sylow-2 helper test).

Tests: `tests/` at repo root (`pip install -e ".[dev]" && pytest`).

## Canonical / migrated (call the library; prime-parametric)

| Script | Purpose | Backs |
|---|---|---|
| `analyze_ablation_delta.py` | per-character Δ_k(a,b,c) contribution + 1D reduction | note 06 |
| `analyze_delta_spectrum.py` | Fourier spectrum of Δ_k (helper detection) | note 06 |
| `analyze_neuron_clusters.py` | per-character neuron-cluster reconstruction | note 05 |
| `analyze_2d_fourier.py` | 2D Fourier diag/off-diag split of MLP output | note 06 |
| `analyze_dmlp_sweep.py` | K / orders / cliff across d_mlp budgets | Exp 1 |
| `analyze_ablation_full.py` | full per-character ablation essentialness | note 06 |
| `analyze_vestigial_ablation.py` | load-bearing vs vestigial classification | note 06 |
| `analyze_bifurcation.py` | K/non-K bifurcation step distribution | note 04 |
| `analyze_cliff_vs_commit.py` | cliff vs bifurcation/ratio (cliff predictor) | note 04 |
| `verify_helper_mechanism.py` | model-free polarization/squaring check | note 06 (§7.4) |
| `check_grokking.py` | scan runs, flag grokked/memorized-only/failed | Exp 1 failure detector |

## Drivers (sweep orchestration)

| Script | Purpose | Notes |
|---|---|---|
| `run_dmlp_seed_sweep.py` | (d_mlp × seed) cross-product | **Exp 1 driver**; `--d-mlps 24,22,20,18,16`; `--p`/`--experiment` for multi-prime |
| `run_dmlp_sweep.py` | d_mlp sweep, single seed | reusable |
| `run_seed_sweep.py` | seed sweep at fixed d_model | reusable |
| `run_sweep.py` | d_model sweep | reusable |

## Per-step diagnostics (not yet migrated; hardcode p=113 in their basis)

`run_fine_grained.py`, `run_fine_grained_exact.py` (config-driven; multi-prime via a new config),
`run_freeze_we.py`, `run_grad_decomp.py`, `run_with_grad.py`. Migrate to `char_index`/`char_energy`
lazily when a multi-prime run needs them.

## Visualization (not yet migrated; p=113-flavored)

`viz_dmlp_grokking.py` (reusable, color tweak for Exp 1), `viz_basis_space.py` (cache-driven),
`viz_basis_dynamics.py`, `viz_basin_commitment.py`, `viz_order_energy_evolution.py`.

## Cache / one-off (active but not migrated)

`build_basis_cache.py` (Pareto/cost enumeration — p=113-specific; parametrize per prime for multi-prime),
and the exploratory analyses: `analyze_attn_output.py`, `analyze_attn_gating.py`,
`analyze_full_signal.py`, `analyze_intrinsic_signal.py`, `analyze_orthogonalized_signal.py`,
`analyze_kk_decomposition.py`, `analyze_path_contribution.py`, `analyze_grad_alignment.py`,
`analyze_grad_decomp.py`, `analyze_char_concentration.py`, `analyze_neuron_packing.py`,
`analyze_order_multisets.py`, `analyze_basis_dynamics.py`, `analyze_WU_WE_orthogonality.py`,
`analyze_bifurcation_dmlp.py`, `analyze_fine_grained.py`. These keep their local helpers until
touched; each later migration is a delete-and-import against the library above.
