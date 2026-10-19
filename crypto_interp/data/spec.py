"""TaskSpec: the four values that fully determine a dataset."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from .base import Dataset


@dataclass(frozen=True)
class TaskSpec:
    """A task and its train/test split. ``build()`` is deterministic in these fields."""

    task: str
    p: int
    frac_train: float = 0.3
    seed: int = 0

    @classmethod
    def from_dict(cls, d: dict) -> TaskSpec:
        return cls(task=d["task"], p=d["p"], frac_train=d["frac_train"], seed=d["seed"])

    def to_dict(self) -> dict:
        return asdict(self)

    def build(self, cache_dir: str | Path | None = None) -> Dataset:
        """Build the dataset, reading/writing ``cache_dir`` if one is given."""
        from . import build, load_or_build

        if cache_dir is None:
            return build(task=self.task, p=self.p, frac_train=self.frac_train, seed=self.seed)
        return load_or_build(
            cache_dir=cache_dir,
            task=self.task,
            p=self.p,
            frac_train=self.frac_train,
            seed=self.seed,
        )
