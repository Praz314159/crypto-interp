# Experiment: sqrt mod p

## Task

Given two integers `(a, b)`, predict `canonical_sqrt(a · b mod p)` where the
canonical root is `min(x, p − x)` so that there is a unique answer per input.
The dataset is restricted to pairs whose product is a quadratic residue (or
zero), so the square root is well-defined.

This reformulation (rather than the unary `y → sqrt(y)`) keeps format
compatibility with Nanda's modular-addition setup (input shape `[a, b, =]`,
about p² examples), so we can reuse the architecture, training loop, and
Fourier-analysis utilities directly.

## Choice of prime

`p = 113`. Same as Nanda. `113 − 1 = 2⁴ · 7`, so Tonelli–Shanks has non-trivial
2-adic structure (`s = 4`, `q = 7`). Expected dataset size is ~6500 pairs
(roughly half of `p² = 12769`).

## Architecture

Mirrors Nanda exactly:

| Setting | Value |
|---|---|
| Layers | 1 |
| `d_model` | 128 |
| `d_mlp` | 512 |
| Attention heads | 4 |
| `d_head` | 32 |
| Activation | ReLU |
| LayerNorm | None |
| `d_vocab` | `p + 1` (with `=` as token `p`) |
| `n_ctx` | 3 |
| Total params | ~227k |

## Training

| Setting | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 1e-3 |
| Weight decay | 1.0 |
| Betas | (0.9, 0.98) |
| LR warmup | linear over 10 steps |
| Batch | full-batch |
| Train fraction | 0.3 (matches Nanda) |
| Cross-entropy | float64 (CPU) to avoid float32 underflow |

Float64 is required for accurate grokking dynamics (Nanda appendix). MPS
doesn't support float64, so default device is CPU; pass `--device mps` to
override at the cost of float32 noise.

## Files

| File | Purpose |
|---|---|
| `reference.py` | Tonelli–Shanks and Cipolla; ground-truth label generator. |
| `data.py` | Dataset construction. |
| `model.py` | Transformer with hook points at every internal activation. |
| `train.py` | Training loop with checkpointing. |
| `interp.py` | (TBD) Fourier-basis projections, ablations, plots. |
| `runs/<tag>/` | Saved checkpoints and loss curves per run. |

## Running

```bash
# Sanity checks
python3 reference.py
python3 data.py
python3 model.py

# Smoke test
python3 train.py --num-epochs 50 --tag smoke_test

# Real run
python3 train.py --num-epochs 50000 --tag main_run
```

## Hypotheses entering the experiment

1. Model groks on this task with the same hyperparameters Nanda used.
2. The learned algorithm is *not* trivially the additive Fourier transform on
   `Z/p` Nanda found for `+`. Sqrt is a multiplicative-group operation, so
   the relevant Fourier structure (if any) likely lives on `(Z/p)*` ≅
   `Z/(p−1)`, and the right basis to look in is the multiplicative-character
   basis, not the additive-character basis.
3. The model's circuit may resemble Cipolla (algebraic, parallel) more than
   Tonelli–Shanks (iterative, conditional), since a one-layer feedforward
   transformer is poorly suited to representing a variable-step iterative
   algorithm.

These are the predictions; the experiment falsifies, confirms, or surprises
us on each.
