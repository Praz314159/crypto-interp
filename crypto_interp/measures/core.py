"""Measures: named, versioned quantities computed from one checkpoint.

A measure is a pure function ``fn(session, **params) -> Result``, where
``Result`` is a dataclass. The :func:`measure` decorator registers it and
adds three ways to evaluate it:

- ``m(session, **params)``: compute directly on a :class:`Session`; no caching.
- ``m.at(checkpoint, **params)``: one checkpoint file, through the results store.
- ``m.over(target, **params)``: a run, a :class:`RunSet` or a list of runs,
  at the latest checkpoint of each (``checkpoints="latest"``) or at every
  checkpoint (``checkpoints="all"``, a trajectory). Returns a pandas
  DataFrame with one row per (run, checkpoint): the run's config columns,
  ``epoch``, and every scalar or tuple field of the result. Array fields stay
  in the store; ``at`` returns them.

Results are cached under ``<run>/measures/`` (see :mod:`.store`) and keyed by
measure version, parameters and checkpoint content, so a figure over a
population only recomputes what changed. Bump ``version`` whenever a measure's
output changes for the same input.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..runs import Run, RunSet, checkpoint_epoch
from .store import Store

REGISTRY: dict[str, Measure] = {}

# Config fields that describe logging or bookkeeping rather than the run.
_TABLE_SKIP = {"notes", "device", "log_every", "save_every", "metrics_every"}


@dataclass(frozen=True)
class Measure:
    name: str
    version: int
    fn: Callable[..., Any]
    result_type: type

    def __call__(self, session, **params):
        return self.fn(session, **params)

    def at(self, checkpoint: str | Path, *, cache: bool = True, **params):
        """The result at one checkpoint file, from the store when possible."""
        checkpoint = Path(checkpoint)
        store = Store(checkpoint.parent)
        if cache:
            hit = store.get(self, checkpoint, params)
            if hit is not None:
                return hit
        result = self(_session(checkpoint), **params)
        if cache:
            store.put(self, checkpoint, params, result)
        return result

    def over(
        self,
        target: Run | RunSet | Iterable[Run] | str | Path,
        *,
        checkpoints: str = "latest",
        cache: bool = True,
        **params,
    ):
        """One row per (run, checkpoint) with the run's config and the result's scalars."""
        import pandas as pd

        if checkpoints not in ("latest", "all"):
            raise ValueError(f"checkpoints must be 'latest' or 'all', got {checkpoints!r}")
        rows = []
        for run in _runs(target):
            paths = [run.latest()] if checkpoints == "latest" else run.checkpoints()
            config = {k: v for k, v in run.config().items() if k not in _TABLE_SKIP}
            for path in paths:
                result = self.at(path, cache=cache, **params)
                rows.append(
                    {
                        "run": run.id,
                        "experiment": run.experiment,
                        **config,
                        "epoch": checkpoint_epoch(path),
                        **scalar_fields(result),
                    }
                )
        return pd.DataFrame(rows)


def measure(name: str, version: int) -> Callable[[Callable], Measure]:
    """Register ``fn(session, **params) -> dataclass`` as a measure."""

    def wrap(fn: Callable) -> Measure:
        if name in REGISTRY:
            raise ValueError(f"measure {name!r} is already registered")
        m = Measure(name, version, fn, _return_type(fn))
        REGISTRY[name] = m
        return m

    return wrap


def _return_type(fn: Callable) -> type:
    """The dataclass named by ``fn``'s return annotation (a string under
    ``from __future__ import annotations``), resolved in ``fn``'s module."""
    rtype = fn.__annotations__.get("return")
    if isinstance(rtype, str):
        rtype = eval(rtype, fn.__globals__)
    if not dataclasses.is_dataclass(rtype):
        raise TypeError(f"{fn.__qualname__} must be annotated to return a dataclass")
    return rtype


def scalar_fields(result) -> dict:
    """A result's fields that fit in a table cell: scalars, strings, tuples of scalars."""
    out = {}
    for f in dataclasses.fields(result):
        v = getattr(result, f.name)
        if v is None or isinstance(v, (bool, int, float, str)):
            out[f.name] = v
        elif isinstance(v, (tuple, list)) and all(isinstance(x, (int, float, str)) for x in v):
            out[f.name] = tuple(v)
    return out


def _session(checkpoint: Path):
    from ..interp import Session
    from ..interp.load import load_run

    model, ds, _ = load_run(checkpoint)
    return Session(model, ds)


def _runs(target) -> list[Run]:
    if isinstance(target, Run):
        return [target]
    if isinstance(target, (str, Path)):
        return [Run.open(target)]
    return list(target)
