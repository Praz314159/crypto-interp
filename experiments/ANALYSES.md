# Analysis catalog

A map of every analysis we have tried across experiments, organized by the
**question** it answers. For each question: the **canonical** entry point (in
`crypto_interp/`, prime-parametric, reusable) and the **ad-hoc** exploration
scripts (in `experiments/<id>/scripts/`, often run-specific).

> **Update 2026-05-26 — harness v1 landed.** The recommended way to write a
> new analysis is to compose verbs from `crypto_interp.interp` (`Session`,
> `act_patch`, `ablate_char_w`, etc.) — usually 5–15 lines, no script
> needed. The canonical recipes are documented in
> [`crypto_interp/playbooks/`](../crypto_interp/playbooks/). Existing
> ad-hoc scripts in `experiments/<id>/scripts/` are kept for reference but
> should be **phased out, not extended**. New analyses → playbooks +
> notebook/Claude turn, not new scripts.

## How the code is layered

- **`crypto_interp/interp/`** — reusable primitives + scaffolding (bases,
  ablation, metrics, dynamics detectors, harmonic-helper detection,
  ActivationCache, hooks, interventions, activation patching, Session).
  Import these; don't re-implement.
- **`crypto_interp/analysis/`** — canonical, runnable CLI analyses (`python
  -m crypto_interp.analysis.<name> --run-dir ...`). Prime-parametric.
  After v1, these are thin wrappers around `Session`.
- **`crypto_interp/playbooks/`** — markdown recipes documenting how to
  compose the library verbs to answer common questions (basis discovery,
  causal intervention, mechanism verification, economy analysis, etc.).
  These are the canonical entry point for new analyses.
- **`experiments/<id>/scripts/`** — ad-hoc / experiment-specific scripts.
  Many predate the v1 harness; treat as historical reference. Do not add
  new scripts here.

Experiment 002 has **no scripts of its own** — it reuses experiment 001's
`--ckpt`-driven scripts. Experiments 004 (p=127) and 005 (p=181) currently have
configs/runs only; they consume the canonical (prime-parametric) analyses.

---

## 1. Embedding basis: which Fourier basis, and how sparse?

*Does W_E factor through the additive or multiplicative Fourier basis? Which
characters K are present, and how concentrated is the energy (participation
ratio)?*

- **Canonical:** `interp/bases.py` (`additive_fourier_basis`,
  `multiplicative_fourier_basis`, `char_index`, `fold_frequency`),
  `interp/metrics.py` (`char_energy`, `char_energy_batch`, `order_energy`,
  `per_frequency_energy_from_embedding`).
- **Ad-hoc:** `001/analyze_embedding.py`, `003/analyze_char_concentration.py`,
  `003/build_basis_cache.py`, `003/viz_basis_space.py`.

## 2. Ablation: which characters are causally load-bearing?

*Remove a character from W_E and measure the test-loss hit. Vestigial vs
load-bearing vs essential; restricted/excluded loss curves; exact per-logit Δ_k.*

- **Canonical:** `interp/ablate.py` (`ablate_character`, `essential_characters`,
  `evaluate_loss`), `analysis/ablation_full.py` (per-character sweep + CSV/PNG),
  `analysis/ablation_delta.py` (exact Δ_k(a,b,c)), `analysis/vestigial_ablation.py`.
- **Ad-hoc:** `001/analyze_ablation.py`, `001/analyze_ablation_curve.py`,
  `003/analyze_full_signal.py`.

## 3. Neuron mechanism: are MLP neurons monosemantic per character?

*Does each neuron specialize to one character (cluster), implementing
cos(θ_k(a)+θ_k(b))? Monosemanticity, cluster sizes, always-firing neurons,
neuron packing.*

- **Canonical:** `analysis/neuron_clusters.py` (`per_neuron_dominant_char`,
  `cluster_signal`, `reference_cos_signal`), `interp/neurons.py`
  (`compute_per_neuron_frequency_energy`, `matched_bigrams`).
- **Ad-hoc:** `001/analyze_neurons.py`, `001/analyze_always_firing.py`,
  `003/analyze_neuron_packing.py`.

## 4. Harmonic-helper / Sylow doubling: the free-rider economy

*Does a character χ_{2k} ride on χ_k's neuron cluster for free (ReLU squaring)?
Primary vs helper classification; the zero-neuron-cost mechanism.*

- **Canonical:** `interp/harmonic.py` (`delta_k_spectrum`,
  `find_primary_helper_pairs`, `helper_multiplier`),
  `analysis/delta_spectrum.py` (which frequency an ablation destroys),
  `analysis/verify_helper_mechanism.py` (ReLU-squaring control + bilinear fit),
  `analysis/fourier2d.py` (2D Fourier of MLP output over (a,b)).
- **Ad-hoc:** `003/analyze_kk_decomposition.py` (product vs quotient at (k,k)),
  `003/analyze_intrinsic_signal.py`, `003/analyze_orthogonalized_signal.py`.

## 5. Grokking dynamics: cliff, bifurcation, basin commitment

*When does the model commit to its final basis relative to the test-loss cliff?
Bifurcation (K starts outgrowing non-K), commitment (top-|K| stably = final K).*

- **Canonical:** `interp/dynamics.py` (`cliff_step`, `bifurcation_step`,
  `commit_step`, `grokking_status`), `analysis/bifurcation.py`,
  `analysis/cliff_vs_commit.py`, `analysis/check_grokking.py`.
- **Ad-hoc:** `003/viz_basin_commitment.py` (per-run 3-panel commitment figure;
  reads `metrics.pt` freq_energies), `003/analyze_basis_dynamics.py`,
  `003/analyze_bifurcation_dmlp.py`, `003/viz_basis_dynamics.py`,
  `003/viz_order_energy_evolution.py`.

## 6. Early-phase / fine-grained trajectories and gradients

*What happens in the first hundreds of steps? When do the gradient / Adam moments
first align with the final K?*

- **Canonical:** none specific (uses `metrics.char_energy_batch` over a stack).
- **Ad-hoc (data-gen):** `003/run_fine_grained.py`,
  `003/run_fine_grained_exact.py`, `003/run_with_grad.py`,
  `003/run_grad_decomp.py`.
- **Ad-hoc (analysis):** `003/analyze_fine_grained.py`,
  `003/analyze_grad_alignment.py`, `003/analyze_grad_decomp.py`.

## 7. Order-class / group structure (Pohlig–Hellman)

*Does the model's character set respect the subgroup/order structure of (Z/p)*?
Distribution of K-order multisets across seeds.*

- **Canonical:** `interp/metrics.py` (`order_of`, `order_energy`).
- **Ad-hoc:** `001/analyze_order_classes.py`, `003/analyze_order_multisets.py`.

## 8. Attention pathway

*Does attention gate on, or carry, the piggybacker characters?*

- **Ad-hoc only:** `003/analyze_attn_gating.py`, `003/analyze_attn_output.py`.

## 9. Unembed (W_U) structure

*Does W_U carry the same superposition structure as W_E / the MLP write directions?*

- **Ad-hoc only:** `001/analyze_unembed.py`,
  `003/analyze_WU_WE_orthogonality.py`.

## 10. Path decomposition

*Which paths through the residual block carry the character-product contribution?*

- **Ad-hoc only:** `003/analyze_path_contribution.py`,
  `003/analyze_grad_decomp.py` (+ `run_grad_decomp.py`).

## 11. Interventions / controls

*Freeze W_E at init — does the model still find the basis? Weight-decay sweeps.*

- **Ad-hoc only:** `003/run_freeze_we.py`, `003/analyze_freeze_we.py`.

## 12. Capacity sweeps (d_model, d_mlp, weight_decay)

*Where are the grokking / algorithmic floors, and how do they depend on the recipe?*

- **Canonical:** `analysis/dmlp_sweep.py`.
- **Ad-hoc (drivers):** `003/run_sweep.py` (d_model), `003/run_dmlp_sweep.py`,
  `003/run_seed_sweep.py`, `003/run_dmlp_seed_sweep.py` (generalized:
  `--weight-decay`, `--batch-size` for parallel batches).
- **Ad-hoc (viz):** `003/viz_dmlp_grokking.py`.

## 13. Cross-seed structural summaries

*Why are some seeds slow (laggards)? Weight-norm trajectories; per-seed structure.*

- **Ad-hoc only:** `001/analyze_laggards.py`,
  `001/analyze_laggards_mechanism.py`, `001/analyze_weight_norms.py`,
  `001/analyze_trajectories.py`, `001/seed_summary.py`.

---

## Worked pipelines (reusable command recipes)

**Full basis + mechanism on a single run** (what we ran for the d_mlp=20 study):

```bash
# 1. Essential characters K + per-character ablation (energy, Δlog10, class)
python -m crypto_interp.analysis.ablation_full --run-dir <RUN>
# 2. Neuron clusters: monosemanticity, cluster sizes, cos-reference correlation,
#    and "no neurons in cluster" => zero-neuron-cost doubling helper
python -m crypto_interp.analysis.neuron_clusters --run-dir <RUN>
# 3. Which frequency each ablation destroys (primary vs helper)
python -m crypto_interp.analysis.delta_spectrum --run-dir <RUN> --ks <k1,k2,...>
# 4. Basin commitment (bifurcation/commit/cliff markers + test-loss overlay)
python experiments/003_dmodel_sweep_p113/scripts/viz_basin_commitment.py \
    --metrics <RUN>/metrics.pt --losses <RUN>/losses.pt --tag <tag>
```

**Helper classification in-process** (primary vs Sylow-2 doubling):

```python
from crypto_interp.interp import (char_index, load_run, latest_checkpoint,
                                  essential_characters, find_primary_helper_pairs)
basis, ci = char_index(113)
model, ds, _ = load_run(latest_checkpoint(RUN))
K = essential_characters(model, ds, ci, basis)["K"]
pairs = find_primary_helper_pairs(model, ds, ci, basis, K)  # (helper, primary, mult, energy)
```

---

## Cleanup candidates (ad-hoc ≈ superseded by canonical)

These predate the `crypto_interp` extraction and largely duplicate canonical
logic; prefer the canonical module, migrate or delete the ad-hoc one:

| Ad-hoc script | Canonical replacement |
|---|---|
| `001/analyze_embedding.py` | `interp/bases.py` + `metrics.char_energy` |
| `001/analyze_ablation.py`, `001/analyze_ablation_curve.py` | `analysis/ablation_full.py`, `interp/ablate.py` |
| `001/analyze_neurons.py` | `analysis/neuron_clusters.py` |
| `001/analyze_order_classes.py`, `003/analyze_order_multisets.py` | `metrics.order_energy` / `order_of` |
| `003/analyze_char_concentration.py` | `neurons.compute_per_neuron_frequency_energy` |
| `003/analyze_basis_dynamics.py`, `003/analyze_bifurcation_dmlp.py` | `interp/dynamics.py` + `analysis/bifurcation.py` + `analysis/cliff_vs_commit.py` |
| `003/analyze_kk_decomposition.py` | `analysis/fourier2d.py` + `interp/harmonic.py` |

`viz_basin_commitment.py` was refactored to delegate to `interp.dynamics` and
`interp.char_energy_batch` (no longer ad-hoc in its internals).

Not superseded (no canonical equivalent yet): the early-phase/gradient data-gen
and analysis (§6), attention (§8), unembed (§9), path decomposition (§10),
interventions (§11), and cross-seed summaries (§13).
