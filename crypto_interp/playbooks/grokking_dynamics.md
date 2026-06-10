# Grokking dynamics — trajectory through training

**Strategy 5.** When during training does basis commitment happen, and how
does it relate to the test-loss cliff? Dynamics is its own strategy with
its own phenomenology that doesn't reduce to the static circuit.

## Algebraic anchor + mech-interp findings

The trajectory of W_E's character spectrum (`char_energy` over time) is the
anchor. Three derived markers (all canonical):

- **Bifurcation** — first step at which the K / non-K mean-energy ratio
  exceeds 1.5× its initial value. The model has *tilted* toward its eventual
  K.
- **Commit** — first step after which the top-|K| characters (by energy)
  stably equal the final K for the rest of training.
- **Cliff** — first step at which test loss drops below 0.1.

Empirical pattern (21-seed population): bifurcation is early (~200–400) and
seed-independent across grokked *and* stuck runs; commit ≈ cliff (basis
finalization ≈ generalization, same event) for the strong groks; commit
precedes cliff for the simplest-basis groks (embedding locks before the
MLP circuit completes wiring).

## Verbs

- `S.dynamics(metrics_path=…)` — returns `{"K", "epochs", "bifurcation",
  "commit", "cliff"}`. Reads `metrics.pt` (the trajectory of
  `freq_energies`) and `losses.pt`.
- Underlying functions if needed directly: `bifurcation_step(char_E_traj,
  K_mask, ratio=1.5)`, `commit_step(char_E_traj, final_K, mode="subset")`,
  `cliff_step(test_losses, thresh=0.1)`.

## Recipe

```python
from pathlib import Path
from crypto_interp.interp import Session

run = Path("experiments/003_dmodel_sweep_p113/runs/dmodel_24_dmlp_20_wd2_seed1")
S = Session.from_run(run)
d = S.dynamics(metrics_path=run / "metrics.pt")
print(f"K = {d['K']}")
print(f"bifurcation step: {d['bifurcation']}")
print(f"commit step:      {d['commit']}")
print(f"cliff step:       {d['cliff']}")
print(f"commit - cliff:   {d['commit'] - d['cliff']}")
```

Expected on seed 1: bifurcation=400, commit=4250, cliff=3948 (commit
≈ cliff, lag +302). Visualize the K-vs-non-K energy ratio over time as a
two-panel plot — the existing `viz_basin_commitment.py` is the canonical
figure.

## De-grokking sub-recipe — train-degenerate drift

A separate phenomenon. After grokking, test CE rises (e.g. 4.6e-6 → 1.7e-3
over 13k epochs) while test *accuracy* stays ~100%. Mechanism (refutation
ladder):

1. **Not** global margin decay (temperature-rescaling fails to recover).
2. **Not** helper erosion (W_E energies stable; helper unchanged).
3. **Not** gauge drift (the function changed, but gauge transforms preserve
   function exactly; we verified by validating the gauge transform).
4. **Is** drift along train-loss-flat directions: weights move ~57% while
   train CE stays ~1.5e-6; a few test examples cross their decision
   boundary at a threshold-crossing.

This is its own appendix-worthy story. The verb is essentially "load the
intermediate checkpoints and plot train CE / test CE / weight drift vs
epoch."

## Common variations

- **Population dynamics.** Compute bifurcation/commit/cliff across seeds;
  see whether grokking timing is predicted by an early-epoch quantity
  (the K-vs-non-K ratio at epoch 1500 is a strong predictor — Spearman
  ρ ≈ -0.92 in `notes/04`).
- **Compare seeds by outcome.** Group bifurcation by {grok, partial,
  stuck}; the universal earliness of bifurcation across outcomes is the
  central result.
