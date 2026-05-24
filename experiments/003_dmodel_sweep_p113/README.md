# Experiment 003 — `d_model` sweep for mul mod 113

## Question

Two distinct quantities, separated after experiment 002:

- **`d_alg`** — minimum `d_model` such that the model finds (even
  partially) the multiplicative-Fourier algorithm. Experiment 002 showed
  `d_alg ≤ 12`: at d_model=12 the model prefers the mul basis and learns
  one strong frequency (k=30) plus scraps, but does not grok.
- **`d_grok`** — minimum `d_model` such that the classical two-phase
  grokking dynamic (memorization plateau → sharp test-loss cliff) occurs.

Hypothesis: `d_grok > d_alg`. The grokking transition requires the model
to enter a memorization basin deep enough that weight decay's erosion of
it migrates the model to the generalizing basin. At d_model=12 the
memorization basin can't be entered (train loss bottoms at ~0.15, not
~10⁻⁷), so the migration mechanism never engages.

## Predicted outcome (from experiment 002)

| d_model | predicted min train loss | predicted grokking? |
|---:|---|---|
| 12 | 0.15 *(observed)* | no |
| 16 | ~0.03 | unlikely |
| 24 | ~10⁻³ | possibly slow |
| 32 | ~10⁻⁴ | yes, delayed |
| 64 | ~10⁻⁶ | yes, near-baseline |
| 128 | ~10⁻⁷ *(observed in 001)* | yes (baseline) |

The threshold between "smooth descent" and "clean two-phase" should fall
somewhere in {24, 32}.

## Status

Not yet run. Scaffolding in place; user will trigger the run sequentially.

## Running

```bash
# Full sweep (sequential, ~20 min × N runs)
python experiments/003_dmodel_sweep_p113/scripts/run_sweep.py

# Subset
python experiments/003_dmodel_sweep_p113/scripts/run_sweep.py --d-models 16,24

# Short pilot (5k epochs each, useful for shaking out plumbing)
python experiments/003_dmodel_sweep_p113/scripts/run_sweep.py --epochs 5000
```

The sweep driver calls `scripts/train.py --override d_model=<N>` once per
value, logging each child to `runs/dmodel_<N>.log` and writing checkpoints
to `runs/dmodel_<N>/`.

## Post-run analysis

For each completed `dmodel_<N>`:

```bash
python experiments/001_mul_p113/scripts/analyze_embedding.py \
    --ckpt experiments/003_dmodel_sweep_p113/runs/dmodel_<N>
```

A useful follow-up script (to be written when results are in):
`scripts/plot_sweep.py` to overlay all the loss curves on one panel and
plot (d_model → min train loss, final test loss) — the two summary
statistics that answer the two questions above.

## Layout

```
config.py                            CONFIG: ExperimentConfig (d_model=128 default,
                                       overridden per-run)
README.md                            this file
runs/                                per-d_model checkpoints, losses.pt, metrics.pt
                                       (gitignored)
runs/dmodel_<N>.log                  per-run subprocess log
datasets/                            cached datasets (gitignored)
figures/                             sweep summary plots (gitignored)
scripts/run_sweep.py                 sequential driver
```
