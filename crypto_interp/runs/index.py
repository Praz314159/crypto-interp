"""The run index: find runs by what they are, not by how their directories are named.

::

    from crypto_interp import runs

    pop = runs.query(experiment="003", d_mlp=20, weight_decay=2.0)
    pop.table()                      # one row per run (pandas)
    for run in pop:
        ckpt = run.load()            # latest Checkpoint
        ...

A run is a directory ``<root>/<experiment>/runs/<name>/`` holding
``checkpoint_*.pt`` files. Its identity comes from ``manifest.json``
(see :mod:`crypto_interp.runs.manifest`), or from its latest checkpoint when the
manifest predates version 2. Runs whose manifest marks them ``excluded`` are
left out unless asked for.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..data.spec import TaskSpec
from .checkpoint import Checkpoint, checkpoint_epoch, checkpoint_paths, load_checkpoint
from .manifest import MANIFEST_VERSION, describe, read_manifest

DEFAULT_ROOT = Path("experiments")


@dataclass(frozen=True)
class Run:
    path: Path
    task: TaskSpec
    model_config: dict
    train_config: dict
    excluded: str | None = None
    provenance: dict = field(default_factory=dict, compare=False)

    @property
    def experiment(self) -> str:
        return self.path.parent.parent.name

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def id(self) -> str:
        return f"{self.experiment}/{self.name}"

    def config(self) -> dict:
        """Task, architecture and training fields in one flat dict (task fields win)."""
        return {**self.train_config, **self.model_config, **self.task.to_dict()}

    def checkpoints(self) -> list[Path]:
        return checkpoint_paths(self.path)

    def latest(self) -> Path:
        ckpts = self.checkpoints()
        if not ckpts:
            raise FileNotFoundError(f"no checkpoints in {self.path}")
        return ckpts[-1]

    def load(self, epoch: int | None = None, map_location: str = "cpu") -> Checkpoint:
        """The checkpoint at ``epoch`` (default: the latest)."""
        if epoch is None:
            return load_checkpoint(self.latest(), map_location=map_location)
        for p in self.checkpoints():
            if checkpoint_epoch(p) == epoch:
                return load_checkpoint(p, map_location=map_location)
        raise FileNotFoundError(f"{self.id} has no checkpoint at epoch {epoch}")

    @classmethod
    def open(cls, path: str | Path) -> Run:
        path = Path(path)
        manifest = read_manifest(path)
        if manifest.get("manifest_version") != MANIFEST_VERSION:
            manifest = {**manifest, **describe(load_checkpoint(checkpoint_paths(path)[-1]))}
        return cls(
            path=path,
            task=TaskSpec.from_dict(manifest["task"]),
            model_config=manifest["model_config"],
            train_config=manifest["train_config"],
            excluded=manifest.get("excluded"),
            provenance=manifest.get("provenance", {}),
        )


Filter = Any  # a value, a list/tuple/set of allowed values, or a predicate


def _matches(value: Any, want: Filter) -> bool:
    if callable(want):
        return bool(want(value))
    if isinstance(want, (list, tuple, set, frozenset, range)):
        return value in want
    return value == want


def _natural_key(run: Run) -> list:
    """Sort key that orders ``seed2`` before ``seed10``."""
    return [int(s) if s.isdigit() else s for s in re.split(r"(\d+)", run.id)]


class RunSet(Sequence[Run]):
    """An ordered collection of runs (by id, with numbers in numeric order)."""

    def __init__(self, runs: Sequence[Run]):
        self._runs = sorted(runs, key=_natural_key)

    def __getitem__(self, i):
        return RunSet(self._runs[i]) if isinstance(i, slice) else self._runs[i]

    def __len__(self) -> int:
        return len(self._runs)

    def __iter__(self) -> Iterator[Run]:
        return iter(self._runs)

    def __repr__(self) -> str:
        return f"RunSet({len(self)} runs)"

    def filter(self, predicate: Callable[[Run], bool] | None = None, **fields: Filter) -> RunSet:
        """Runs satisfying ``predicate`` and every ``field=value`` on :meth:`Run.config`."""
        out = []
        for run in self._runs:
            cfg = run.config()
            if predicate is not None and not predicate(run):
                continue
            if all(k in cfg and _matches(cfg[k], v) for k, v in fields.items()):
                out.append(run)
        return RunSet(out)

    def table(self):
        """One row per run: id, experiment, name, the flat config, checkpoints, latest epoch."""
        import pandas as pd

        rows = []
        for run in self._runs:
            ckpts = run.checkpoints()
            rows.append(
                {
                    "id": run.id,
                    "experiment": run.experiment,
                    "name": run.name,
                    **run.config(),
                    "n_checkpoints": len(ckpts),
                    "latest_epoch": checkpoint_epoch(ckpts[-1]) if ckpts else None,
                    "excluded": run.excluded,
                }
            )
        return pd.DataFrame(rows).set_index("id") if rows else pd.DataFrame()


def discover(root: str | Path = DEFAULT_ROOT, include_excluded: bool = False) -> RunSet:
    """Every run under ``<root>/*/runs/*`` that has at least one checkpoint."""
    runs = []
    for path in sorted(Path(root).glob("*/runs/*")):
        if path.is_dir() and checkpoint_paths(path):
            run = Run.open(path)
            if include_excluded or run.excluded is None:
                runs.append(run)
    return RunSet(runs)


def query(
    root: str | Path = DEFAULT_ROOT,
    experiment: str | None = None,
    include_excluded: bool = False,
    **fields: Filter,
) -> RunSet:
    """Runs under ``root`` matching config ``fields``.

    ``experiment`` matches the experiment directory by prefix (``"003"`` selects
    ``003_dmodel_sweep_p113``). Field filters take a value, a collection of
    allowed values, or a predicate: ``query(p=113, seed=range(1, 11), d_mlp=lambda d: d < 64)``.
    """
    found = discover(root, include_excluded=include_excluded)
    if experiment is not None:
        found = found.filter(lambda r: r.experiment.startswith(experiment))
    return found.filter(**fields)
