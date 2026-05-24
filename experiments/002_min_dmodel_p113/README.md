# Experiment 002 — minimum `d_model` for grokking mul mod 113

## Task

Same as experiment 001 (modular multiplication mod 113), but with the residual
stream radically narrowed: `d_model = 12` instead of `128`. Everything else
held at the 001 baseline.

## Hypothesis

Experiment 001 measured the algorithm using $|\mathcal K| = 4$ key
multiplicative-Fourier frequencies. Two readings of the algorithmic
minimum:

| Reasoning                                            | Frequencies | Position overhead | Min `d_model` |
|------------------------------------------------------|------------:|------------------:|--------------:|
| Frequency-counting only                              | $|\mathcal K|=4$ | 0 (`=`-token doubles as pos. signal) | 8  |
| Frequencies + rank-2 PosEmbed                        | 4           | 2                 | 10            |
| Frequencies + rank-3 PosEmbed                        | 4           | 3                 | 11            |
| Frequencies = 5 (one extra) + no pos. overhead       | 5           | 0                 | 10            |

`d_model = 12` is therefore one step *above* the predicted floor — enough
slack to make grokking plausible, tight enough to discriminate which of
the regimes below the model actually inhabits.

## Predicted outcome regimes

The post-training analysis (embedding-basis energy + PosEmbed-rank check)
should land the run in one of:

1. **Groks, $|\mathcal K| = 4$, PosEmbed near-zero.** Algorithmic-floor regime only; shrink further (try `d_model = 10`, then `8`).
2. **Groks, $|\mathcal K| = 4$, non-trivial PosEmbed.** 4-freqs + 2-dim PosEmbed; next probe `d_model = 10`.
3. **Groks, $|\mathcal K| = 5$.** The extra slack went to one more frequency; matches the "5×2" original hypothesis.
4. **Memorizes but does not grok within 20k epochs.** `d_model = 12` is below practical floor; bracket from above with `d_model = 16`.
5. **Train loss does not converge.** Capacity-bounded; surprising at d_model=12 but possible (would point to `d_mlp` or another knob).

## Status

Not yet run.

## Running

From the repo root:

```bash
# Single seed
python scripts/train.py --experiment 002_min_dmodel_p113 --tag seed0

# Multi-seed parallel sweep
python scripts/sweep.py --experiment 002_min_dmodel_p113 --seeds 0..2 --concurrency 3 --epochs 20000
```

Post-run analysis (the analysis scripts live with experiment 001 but are
`--ckpt`-driven and `d_model`-agnostic):

```bash
python experiments/001_mul_p113/scripts/analyze_embedding.py \
    --ckpt experiments/002_min_dmodel_p113/runs/seed0
```

## Layout

```
config.py    CONFIG: ExperimentConfig with d_model=12, num_epochs=20000
runs/        checkpoints, losses.pt, metrics.pt per --tag    (gitignored)
datasets/    cached tokenized datasets, one per (frac_train, seed)  (gitignored)
figures/     plots from the analysis scripts                  (gitignored)
```
