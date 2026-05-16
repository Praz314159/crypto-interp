# Experiment 001 — modular multiplication mod p

## Task

Given two tokens `(a, b)`, predict `(a * b) mod p` for `p = 113`. Format is
identical to Nanda et al.'s modular-addition setup (`[a, b, =]`, vocab
`p + 1`), so architecture, training loop, and Fourier-analysis utilities are
reusable.

## Hypothesis

Modular multiplication is the multiplicative-group analog of modular addition.
If Nanda's model learns the DFT-trick circuit for `+`, the equivalent circuit
for `×` should live in the **multiplicative character basis** on `(Z/p)*`
(which is `Z/(p-1)`). Concretely, projecting `W_E` along the value-token axis
into the multiplicative-Fourier basis should give a sparse spectrum, and each
MLP neuron should specialize to a single frequency `k`, with energy
concentrated on the 4 "matched bigrams" `(cos k, cos k)`, `(sin k, sin k)`,
`(sin k, cos k)`, `(cos k, sin k)` in the 2D decomposition.

## Status

MVP complete. Trained 16-seed sweep at `p=113`, frac_train=0.3, 8 000 epochs.
All seeds grok; the multiplicative-basis interpretation appears to hold.
Cleanup of the analysis scripts and a paper writeup are in flight in
`notes/`.

## Running

From the repo root (with `pip install -e .` already done):

```bash
# Single training run
python scripts/train.py --experiment 001_mul_p113 --tag main_run

# 16-seed parallel sweep (laggard-grokking dynamics)
python scripts/sweep.py --experiment 001_mul_p113 --seeds 0..15 --concurrency 4

# Analysis scripts (run from this directory; default --ckpt = runs/mul_baseline)
cd experiments/001_mul_p113
python scripts/analyze_embedding.py
python scripts/analyze_neurons.py
python scripts/analyze_ablation.py
python scripts/seed_summary.py --runs runs/mul_sweep_seed*
```

## Layout

```
config.py         CONFIG: ExperimentConfig — single source of truth for hparams
runs/             per-tag training artifacts (checkpoints, losses.pt, metrics.pt)
datasets/         cached tokenized datasets, one per (frac_train, seed)
figures/          plots produced by the analysis scripts
scripts/          experiment-specific analyses
                    analyze_embedding.py    W_E projection onto add vs. mul basis
                    analyze_neurons.py      per-neuron frequency specialization
                    analyze_ablation.py     key-only / anti-key embedding ablation
                    analyze_ablation_curve.py  ablation vs. K
                    analyze_unembed.py      unembedding analysis
                    analyze_trajectories.py per-frequency energy during training
                    analyze_order_classes.py
                    analyze_weight_norms.py weight-norm laggard analysis
                    analyze_laggards*.py    late-grokking seeds
                    analyze_always_firing.py
                    seed_summary.py         cross-seed structural summary
                    extend_laggards.sh
```

## Findings

Document key empirical takeaways here as they land. Initial highlights from
the MVP go in `notes/content/01_mod_mul_algorithm.tex`.
