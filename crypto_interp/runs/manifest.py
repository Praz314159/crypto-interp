"""``manifest.json``: what a run is, without opening its checkpoints.

Layout (version 2)::

    manifest_version   2
    task               TaskSpec fields
    model_config       every TransformerConfig field
    train_config       the ExperimentConfig the run was trained with
    provenance         where and when it was trained (device, hostname, git_sha,
                       started_at, ...); whatever is known
    excluded           optional: why the run must not be used in populations
    migrations         optional: record of past repairs to the run directory

The checkpoint list is not stored: it is read from the directory, so it
cannot go stale.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

from ..data.spec import TaskSpec
from ..models import TransformerConfig
from .checkpoint import Checkpoint, checkpoint_paths, load_checkpoint

MANIFEST = "manifest.json"
MANIFEST_VERSION = 2

# Keys of older manifests that describe provenance rather than the run.
_PROVENANCE_KEYS = ("device", "hostname", "git_sha", "started_at", "source", "backfilled")
# Keys of backfilled manifests that duplicate (and go stale against) the directory.
_DERIVED_KEYS = ("n_checkpoints", "checkpoint_steps", "file_dates", "has_metrics", "has_losses")


def read_manifest(run_dir: str | Path) -> dict:
    path = Path(run_dir) / MANIFEST
    return json.loads(path.read_text()) if path.exists() else {}


def write_manifest(run_dir: str | Path, manifest: dict) -> None:
    (Path(run_dir) / MANIFEST).write_text(json.dumps(manifest, indent=2, default=str))


def make_manifest(
    task: TaskSpec,
    model_config: TransformerConfig,
    train_config: dict,
    provenance: dict | None = None,
) -> dict:
    manifest = {
        "manifest_version": MANIFEST_VERSION,
        "task": task.to_dict(),
        "model_config": dataclasses.asdict(model_config),
        "train_config": train_config,
    }
    if provenance:
        manifest["provenance"] = provenance
    return manifest


def describe(ckpt: Checkpoint) -> dict:
    """The manifest fields that identify a run, taken from one of its checkpoints."""
    return make_manifest(ckpt.task, ckpt.model_config, ckpt.train_config)


def rebuild_manifest(run_dir: str | Path) -> dict:
    """Bring a run's manifest to version 2 from its latest checkpoint.

    Identity fields are rewritten from the checkpoint; provenance, exclusion
    and migration records are kept; keys that only restate the directory's
    contents are dropped.
    """
    run_dir = Path(run_dir)
    paths = checkpoint_paths(run_dir)
    if not paths:
        raise FileNotFoundError(f"no checkpoints in {run_dir}")
    old = read_manifest(run_dir)
    provenance = dict(old.get("provenance", {}))
    for k in _PROVENANCE_KEYS:
        if k in old:
            provenance[k] = old[k]
    if "file_dates" in old:
        provenance.setdefault("file_dates", old["file_dates"])

    new = describe(load_checkpoint(paths[-1]))
    if provenance:
        new["provenance"] = provenance
    for k in ("excluded", "migrations"):
        if k in old:
            new[k] = old[k]
    write_manifest(run_dir, new)
    return new
