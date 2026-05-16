# crypto_interp

Mechanistic interpretability of cryptographically-important algorithms learned
by small transformers. We train Nanda-style 1-layer transformers on modular-
arithmetic tasks (multiplication, square-root mod p, Legendre symbol, ...) and
reverse-engineer the algorithms they learn. The central question: do ML models
converge on the algorithms cryptographers have designed, find variants, or
invent something different?

See `research_directions.md` for the strategy doc.

## Repo layout

```
crypto_interp/                  importable library
  data/         task registry: mul, sqrt, ... → tokenized Datasets
  models/       transformer (Nanda-style 1L, hook points everywhere)
  reference/    cryptographer's reference algorithms (Tonelli-Shanks, Cipolla, ...)
  training/     ExperimentConfig dataclass, training loop, multi-seed sweep
  interp/       Fourier bases, activation caching, ablation, progress measures

experiments/                    one dir per concrete experiment
  001_mul_p113/                 e.g. modular multiplication, p=113
    config.py                   CONFIG = ExperimentConfig(...)
    README.md                   hypothesis, status, findings
    runs/                       checkpoints (gitignored)
    datasets/                   cached tokenized datasets (gitignored)
    figures/                    plots (gitignored)
    scripts/                    experiment-specific analysis scripts

scripts/                        global CLI entry points
  train.py                      python scripts/train.py --experiment 001_mul_p113
  sweep.py                      python scripts/sweep.py --experiment 001_mul_p113 --seeds 0..15

notes/                          paper drafts (LaTeX)
papers/                         reference PDFs and notebooks
research_directions.md          strategy doc
```

## Setup

```bash
pip install -e .
```

## Conventions

- Each experiment has a fixed numeric id prefix (`001_`, `002_`, ...) and a short
  descriptive name. The id is permanent so cross-references in the paper /
  notes don't drift.
- Library code lives in `crypto_interp/`. Anything experiment-specific lives in
  `experiments/<id_name>/`. If you find yourself copy-pasting a function between
  experiment scripts, promote it into the library.
- Checkpoints, datasets, and figures are regenerable from code + config; they
  live under each experiment's dir and are excluded from git.
