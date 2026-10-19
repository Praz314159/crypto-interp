"""The run migration on synthetic runs with the failure modes found on disk."""

import json
import shutil

import pytest
import torch

from crypto_interp.interp import latest_checkpoint
from crypto_interp.runs import checkpoint_paths, load_checkpoint
from crypto_interp.runs.migrate import main as migrate
from crypto_interp.training import ExperimentConfig, train


def _cfg(**kw):
    base = dict(
        task="mul",
        p=7,
        d_model=8,
        d_mlp=16,
        num_heads=2,
        d_head=4,
        num_epochs=20,
        save_every=5,
        log_every=1000,
        metrics_every=5,
        fine_grained_until=3,
    )
    return ExperimentConfig(**{**base, **kw})


def _to_schema_1(run, drop=()):
    """Rewrite a run's checkpoints the way the old training loop stored them."""
    for path in checkpoint_paths(run):
        ck = load_checkpoint(path)
        config = {**ck.train_config, "datasets_dir": "/elsewhere/datasets"}
        for k in drop:
            config.pop(k)
        torch.save(
            {
                "epoch": ck.epoch,
                "model_state": ck.model_state,
                "optimizer_state": ck.optimizer_state,
                "scheduler_state": ck.scheduler_state,
                "train_losses": ck.train_losses,
                "test_losses": ck.test_losses,
                "config": config,
            },
            path,
        )


@pytest.fixture
def mixed_run(tmp_path):
    """A seed-0 run in schema 1, plus a stray early-stop checkpoint from another
    same-seed run that sorts last, as in 005/dmodel_24_dmlp_32_seed11."""
    run = tmp_path / "runs" / "toy_seed0"
    train(_cfg(), run, tmp_path / "datasets")
    _to_schema_1(run)
    other = tmp_path / "other"
    train(_cfg(lr=3e-3, num_epochs=24), other, tmp_path / "datasets")
    shutil.copy(other / "checkpoint_000023.pt", run / "checkpoint_000023.pt")
    return run


def test_dry_run_changes_nothing(mixed_run):
    before = sorted(p.name for p in mixed_run.iterdir())
    migrate([str(mixed_run)])
    assert sorted(p.name for p in mixed_run.iterdir()) == before


def test_apply_quarantines_strays_and_rewrites_schema_1(mixed_run):
    assert latest_checkpoint(mixed_run).name == "checkpoint_000023.pt"  # the stray

    migrate([str(mixed_run), "--apply"])

    assert (mixed_run / "_strays" / "checkpoint_000023.pt").exists()
    assert latest_checkpoint(mixed_run).name == "checkpoint_000019.pt"
    assert all(load_checkpoint(p).source_schema == 2 for p in checkpoint_paths(mixed_run))
    # losses, metrics and fine-grained files belong to the primary run and stay.
    assert {"losses.pt", "metrics.pt", "fine_grained.pt"} <= {p.name for p in mixed_run.iterdir()}

    manifest = json.loads((mixed_run / "manifest.json").read_text())
    assert manifest["migrations"][0]["quarantined"] == ["checkpoint_000023.pt"]
    assert "excluded" not in manifest

    migrate([str(mixed_run), "--apply"])  # idempotent
    manifest = json.loads((mixed_run / "manifest.json").read_text())
    assert len(manifest["migrations"]) == 1


def test_seed_named_directory_with_another_seed_is_excluded(tmp_path):
    run = tmp_path / "runs" / "toy_seed5"
    train(_cfg(), run, tmp_path / "datasets")  # trained with seed 0

    migrate([str(run), "--apply"])
    manifest = json.loads((run / "manifest.json").read_text())
    assert "seed 0" in manifest["excluded"]

    migrate([str(run), "--apply"])
    assert len(json.loads((run / "manifest.json").read_text())["migrations"]) == 1


def test_task_is_inferred_from_recorded_losses(tmp_path):
    """The earliest runs never stored a task; the right one is the task whose
    dataset reproduces the losses the run recorded (here not the default, mul)."""
    run = tmp_path / "runs" / "early_run"
    train(_cfg(task="sqrt_of_product"), run, tmp_path / "datasets")
    _to_schema_1(run, drop=("task",))
    with pytest.raises(KeyError, match="task"):
        load_checkpoint(checkpoint_paths(run)[-1])

    migrate([str(run), "--apply"])

    ckpts = [load_checkpoint(p) for p in checkpoint_paths(run)]
    assert {(c.source_schema, c.task.task) for c in ckpts} == {(2, "sqrt_of_product")}
    notes = json.loads((run / "manifest.json").read_text())["migrations"][0]["notes"]
    assert any("sqrt_of_product" in n for n in notes)


def test_wrong_task_is_corrected(tmp_path):
    """A schema-2 file stamped with the wrong task (as a bad default would leave it)."""
    run = tmp_path / "runs" / "mislabelled"
    train(_cfg(task="sqrt_of_product"), run, tmp_path / "datasets")
    for path in checkpoint_paths(run):
        ck = load_checkpoint(path)
        torch.save({**ck.to_dict(), "task": {**ck.task.to_dict(), "task": "add"}}, path)

    migrate([str(run), "--apply"])

    assert load_checkpoint(checkpoint_paths(run)[-1]).task.task == "sqrt_of_product"
