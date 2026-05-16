"""Typed config for a training experiment.

Each ``experiments/<id>/config.py`` exports a top-level ``CONFIG`` of this type.
Scripts read it; the training loop consumes it. Adding a new knob means adding
one field here, not parsing a new CLI flag.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field


@dataclass
class ExperimentConfig:
    # --- Task / data ---
    task: str                       # see crypto_interp.data.available_tasks()
    p: int                          # prime modulus
    frac_train: float = 0.3
    seed: int = 0

    # --- Architecture (Nanda 1L transformer; d_vocab is derived from dataset) ---
    d_model: int = 128
    d_mlp: int = 512
    num_heads: int = 4
    d_head: int = 32
    n_ctx: int = 3
    num_layers: int = 1

    # --- Optimization ---
    lr: float = 1e-3
    weight_decay: float = 1.0
    beta1: float = 0.9
    beta2: float = 0.98
    warmup_steps: int = 10
    num_epochs: int = 50_000

    # --- Logging / checkpointing ---
    save_every: int = 500
    log_every: int = 100
    metrics_every: int = 0          # 0 disables per-frequency tracking

    # --- Device ---
    device: str = "auto"            # "auto" | "cpu" | "mps" | "cuda"

    # --- Early stopping ---
    early_stop_loss: float = 0.0    # 0 disables
    early_stop_patience: int = 500

    # --- Per-experiment metadata (free-form) ---
    notes: str = ""
    extra: dict = field(default_factory=dict)

    def to_checkpoint_dict(self) -> dict:
        """Compact dict stamped onto each checkpoint for reproducibility."""
        d = asdict(self)
        d.pop("extra", None)
        return d
