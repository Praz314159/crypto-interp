"""The measures framework: caching, invalidation, and mapping over runs."""

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import torch

from crypto_interp.measures import measure
from crypto_interp.runs import Run, checkpoint_paths, load_checkpoint, save_checkpoint
from crypto_interp.training import ExperimentConfig, train

REFERENCE_CKPT = (
    Path(__file__).parent / "fixtures" / "mul_p113_d24_m32_seed1" / "checkpoint_009999.pt"
)

CALLS = []


@dataclass
class Probe:
    p: int
    n_test: int
    scale: float
    ks: tuple
    weights: np.ndarray


@measure("_test_probe", version=1)
def probe(S, scale: float = 1.0) -> Probe:
    CALLS.append(scale)
    W = S.model.embed.W_E.detach().double().numpy()
    return Probe(S.ds.p, S.ds.n_test, scale, (1, 2), scale * W[:2, :3])


@pytest.fixture
def run_dir(tmp_path):
    d = tmp_path / "exp" / "runs" / "ref"
    d.mkdir(parents=True)
    shutil.copy(REFERENCE_CKPT, d / REFERENCE_CKPT.name)
    CALLS.clear()
    return d


def test_at_caches_and_round_trips_arrays(run_dir):
    ckpt = checkpoint_paths(run_dir)[0]
    first = probe.at(ckpt)
    again = probe.at(ckpt)
    assert CALLS == [1.0]  # second call came from the store
    assert (again.p, again.ks) == (113, (1, 2))
    assert np.array_equal(again.weights, first.weights)
    assert (run_dir / "measures" / "_test_probe" / "v1").is_dir()


def test_params_and_cache_flag(run_dir):
    ckpt = checkpoint_paths(run_dir)[0]
    probe.at(ckpt, scale=2.0)
    probe.at(ckpt, scale=3.0)
    probe.at(ckpt, scale=2.0)
    probe.at(ckpt, scale=2.0, cache=False)
    assert CALLS == [2.0, 3.0, 2.0]


def test_changed_checkpoint_invalidates(run_dir):
    ckpt = checkpoint_paths(run_dir)[0]
    before = probe.at(ckpt).weights
    ck = load_checkpoint(ckpt)
    ck.model_state["embed.W_E"] = ck.model_state["embed.W_E"] * 2
    save_checkpoint(ck, ckpt)
    os.utime(ckpt, (1, 1))
    assert np.allclose(probe.at(ckpt).weights, 2 * before)
    assert len(CALLS) == 2


def test_entries_with_stale_fields_are_recomputed(run_dir):
    """An entry written by an earlier form of the result type (same version,
    different fields) is a miss, and the fresh result replaces it."""
    import json

    ckpt = checkpoint_paths(run_dir)[0]
    probe.at(ckpt)
    (entry,) = (run_dir / "measures" / "_test_probe" / "v1").glob("*.json")
    meta = json.loads(entry.read_text())
    del meta["ks"]
    meta["retired_field"] = 0
    entry.write_text(json.dumps(meta))

    assert probe.at(ckpt).ks == (1, 2)
    assert probe.at(ckpt).ks == (1, 2)
    assert CALLS == [1.0, 1.0]  # recomputed once, then cached again


def test_over_runs_and_trajectories(tmp_path):
    cfg = dict(
        task="mul",
        p=7,
        d_model=8,
        d_mlp=16,
        num_heads=2,
        d_head=4,
        num_epochs=11,
        save_every=5,
        log_every=1000,
    )
    runs = []
    for seed in (0, 1):
        path = tmp_path / "exp" / "runs" / f"r{seed}"
        train(ExperimentConfig(**cfg, seed=seed), path)
        runs.append(Run.open(path))

    latest = probe.over(runs)
    assert list(latest["run"]) == ["exp/r0", "exp/r1"]
    assert list(latest["seed"]) == [0, 1] and list(latest["epoch"]) == [10, 10]
    assert {"p", "n_test", "scale", "ks"} <= set(latest.columns)
    assert "weights" not in latest.columns  # arrays stay in the store

    traj = probe.over(runs[0], checkpoints="all", scale=2.0)
    assert list(traj["epoch"]) == [0, 5, 10] and (traj["scale"] == 2.0).all()

    with pytest.raises(ValueError, match="checkpoints"):
        probe.over(runs, checkpoints="some")
    assert torch.is_tensor(load_checkpoint(runs[0].latest()).model_state["embed.W_E"])


def test_measure_must_return_a_dataclass():
    with pytest.raises(TypeError, match="dataclass"):

        @measure("_test_bad", version=1)
        def bad(S) -> dict:
            return {}
