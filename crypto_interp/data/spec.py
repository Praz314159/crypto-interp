"""TaskSpec: the four values that fully determine a dataset."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .base import Dataset


@dataclass(frozen=True)
class TaskSpec:
    """A task and its train/test split. ``build()`` is deterministic in these fields."""

    task: str
    p: int  # the modulus: prime for the mul tasks, any n for add (the group Z/n)
    frac_train: float = 0.3
    seed: int = 0

    @classmethod
    def from_dict(cls, d: dict) -> TaskSpec:
        return cls(task=d["task"], p=d["p"], frac_train=d["frac_train"], seed=d["seed"])

    def to_dict(self) -> dict:
        return asdict(self)

    def structure(self):
        """The task's :class:`~crypto_interp.data.structure.CyclicStructure`
        (group tasks only), without building the dataset."""
        from . import structure

        return structure(self.task, self.p)

    def build(self) -> Dataset:
        """Build the dataset. Deterministic in the spec and fast (0.2 s at p = 181),
        so datasets are never cached to disk."""
        from . import build

        return build(task=self.task, p=self.p, frac_train=self.frac_train, seed=self.seed)
