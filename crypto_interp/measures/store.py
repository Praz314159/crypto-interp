"""The results store: cached measure results next to the run they describe.

Layout::

    <run>/measures/<measure>/v<version>/<checkpoint stem>-<params hash>.json
    <run>/measures/<measure>/v<version>/<checkpoint stem>-<params hash>.npz   # array fields

The JSON file holds the result's non-array fields plus a ``_key`` block:
measure name and version, parameters, and the checkpoint's SHA-256 with the
file size and mtime it was computed at. An entry is used only if the
checkpoint still hashes the same (the hash is recomputed only when size or
mtime changed), so rewriting a checkpoint invalidates its results.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

import numpy as np

STORE_DIR = "measures"


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def params_key(params: dict) -> str:
    if not params:
        return "default"
    blob = json.dumps(params, sort_keys=True, default=_jsonable)
    return hashlib.sha1(blob.encode()).hexdigest()[:10]


def _jsonable(x):
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    if isinstance(x, (tuple, set, frozenset, range)):
        return list(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    raise TypeError(f"cannot store {type(x).__name__}")


class Store:
    def __init__(self, run_dir: str | Path):
        self.root = Path(run_dir) / STORE_DIR

    def _base(self, measure, checkpoint: Path, params: dict) -> Path:
        return (
            self.root
            / measure.name
            / f"v{measure.version}"
            / f"{checkpoint.stem}-{params_key(params)}"
        )

    def get(self, measure, checkpoint: Path, params: dict):
        base = self._base(measure, checkpoint, params)
        meta_path = base.with_suffix(".json")
        if not meta_path.exists():
            return None
        meta = json.loads(meta_path.read_text())
        key = meta.pop("_key")
        if not self._same_checkpoint(checkpoint, key):
            return None
        rtype = measure.result_type
        arrays = (
            dict(np.load(base.with_suffix(".npz"))) if base.with_suffix(".npz").exists() else {}
        )
        fields = {**meta, **arrays}
        for f in dataclasses.fields(rtype):
            if isinstance(fields.get(f.name), list):
                fields[f.name] = tuple(fields[f.name])
        return rtype(**fields)

    def put(self, measure, checkpoint: Path, params: dict, result) -> None:
        base = self._base(measure, checkpoint, params)
        base.parent.mkdir(parents=True, exist_ok=True)
        stat = checkpoint.stat()
        scalars, arrays = {}, {}
        for f in dataclasses.fields(result):
            v = getattr(result, f.name)
            (arrays if isinstance(v, np.ndarray) else scalars)[f.name] = v
        scalars["_key"] = {
            "measure": measure.name,
            "version": measure.version,
            "params": params,
            "result_type": type(result).__qualname__,
            "checkpoint_sha256": file_sha256(checkpoint),
            "checkpoint_size": stat.st_size,
            "checkpoint_mtime": stat.st_mtime,
        }
        if arrays:
            np.savez(base.with_suffix(".npz"), **arrays)
        base.with_suffix(".json").write_text(json.dumps(scalars, indent=1, default=_jsonable))

    @staticmethod
    def _same_checkpoint(checkpoint: Path, key: dict) -> bool:
        if not checkpoint.exists():
            return False
        stat = checkpoint.stat()
        if stat.st_size == key["checkpoint_size"] and stat.st_mtime == key["checkpoint_mtime"]:
            return True
        return file_sha256(checkpoint) == key["checkpoint_sha256"]
