"""crypto_interp: mechanistic interpretability of cryptographic algorithms.

See ``research_directions.md`` for the project's strategy doc. Top-level
subpackages:

  data       — task registry: (a, b) → tokenized Datasets for each task
  models     — Nanda-style 1-layer transformer with hook points
  reference  — cryptographer's reference algorithms (Tonelli-Shanks, ...)
  training   — ExperimentConfig dataclass, training loop, multi-seed sweep
  interp     — Fourier bases, activation caching, ablation, progress measures
"""

__version__ = "0.1.0"
