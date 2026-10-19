"""Self-describing checkpoints.

A checkpoint file holds everything needed to rebuild its model and dataset:

    schema_version   2
    epoch            last completed epoch
    task             TaskSpec fields (task, p, frac_train, seed)
    model_config     every TransformerConfig field, including d_vocab
    train_config     the ExperimentConfig the run was trained with
    model_state, optimizer_state, scheduler_state
    train_losses, test_losses   full history up to ``epoch``

Schema 1 (everything written before October 2026) kept one flat ``config``
dict, sometimes without ``task``, ``n_ctx`` or ``num_layers``, plus an absolute
``datasets_dir``. :func:`load_checkpoint` reads both and always returns a
:class:`Checkpoint`.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path

import torch

from ..data.spec import TaskSpec
from ..models import Transformer, TransformerConfig

SCHEMA_VERSION = 2

# Schema-1 config keys that describe where the file was written, not the run.
_V1_LOCAL_KEYS = ("datasets_dir",)


@dataclass
class Checkpoint:
    epoch: int
    task: TaskSpec
    model_config: TransformerConfig
    train_config: dict
    model_state: dict
    optimizer_state: dict | None = None
    scheduler_state: dict | None = None
    train_losses: list[float] = field(default_factory=list)
    test_losses: list[float] = field(default_factory=list)
    #: Schema of the file this was read from (writing always uses SCHEMA_VERSION).
    source_schema: int = SCHEMA_VERSION

    def build_model(self, device: str | torch.device = "cpu") -> Transformer:
        model = Transformer.from_config(self.model_config).to(device)
        model.load_state_dict(self.model_state)
        model.eval()
        return model

    def to_dict(self) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "epoch": self.epoch,
            "task": self.task.to_dict(),
            "model_config": dataclasses.asdict(self.model_config),
            "train_config": self.train_config,
            "model_state": self.model_state,
            "optimizer_state": self.optimizer_state,
            "scheduler_state": self.scheduler_state,
            "train_losses": list(self.train_losses),
            "test_losses": list(self.test_losses),
        }


def save_checkpoint(ckpt: Checkpoint, path: str | Path) -> Path:
    path = Path(path)
    torch.save(ckpt.to_dict(), path)
    return path


def load_checkpoint(
    path: str | Path,
    map_location: str | torch.device = "cpu",
    v1_defaults: dict | None = None,
) -> Checkpoint:
    """Read a checkpoint of any schema.

    ``v1_defaults`` fills config keys a schema-1 file lacks (e.g. ``{"task": "mul"}``
    for the earliest runs); it is ignored for schema 2.
    """
    raw = torch.load(Path(path), map_location=map_location, weights_only=False)
    return from_dict(raw, v1_defaults=v1_defaults)


def from_dict(raw: dict, v1_defaults: dict | None = None) -> Checkpoint:
    version = raw.get("schema_version", 1)
    if version == 2:
        return Checkpoint(
            epoch=raw["epoch"],
            task=TaskSpec.from_dict(raw["task"]),
            model_config=TransformerConfig.from_dict(raw["model_config"]),
            train_config=raw["train_config"],
            model_state=raw["model_state"],
            optimizer_state=raw.get("optimizer_state"),
            scheduler_state=raw.get("scheduler_state"),
            train_losses=list(raw.get("train_losses", [])),
            test_losses=list(raw.get("test_losses", [])),
            source_schema=2,
        )
    if version == 1:
        return _from_v1(raw, v1_defaults or {})
    raise ValueError(f"unknown checkpoint schema_version {version!r}")


def _from_v1(raw: dict, defaults: dict) -> Checkpoint:
    config = {**defaults, **raw["config"]}
    missing = [k for k in ("task", "p", "frac_train", "seed") if k not in config]
    if missing:
        raise KeyError(f"schema-1 checkpoint config lacks {missing}; pass them via v1_defaults")
    # d_vocab was never stored; the embedding matrix is (d_model, d_vocab).
    d_vocab = raw["model_state"]["embed.W_E"].shape[1]
    train_config = {k: v for k, v in config.items() if k not in _V1_LOCAL_KEYS}
    return Checkpoint(
        epoch=raw["epoch"],
        task=TaskSpec.from_dict(config),
        model_config=TransformerConfig.from_dict({**config, "d_vocab": d_vocab}),
        train_config=train_config,
        model_state=raw["model_state"],
        optimizer_state=raw.get("optimizer_state"),
        scheduler_state=raw.get("scheduler_state"),
        train_losses=list(raw.get("train_losses", [])),
        test_losses=list(raw.get("test_losses", [])),
        source_schema=1,
    )


def checkpoint_epoch(path: str | Path) -> int:
    """Epoch encoded in a ``checkpoint_<epoch>.pt`` filename."""
    return int(Path(path).stem.split("_")[-1])


def checkpoint_paths(run_dir: str | Path) -> list[Path]:
    """A run's checkpoints in epoch order. Only the run directory itself is
    searched, so quarantined files under ``_strays/`` are never picked up."""
    return sorted(Path(run_dir).glob("checkpoint_*.pt"), key=checkpoint_epoch)
