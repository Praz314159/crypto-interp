import json

import pytest

from crypto_interp import runs
from crypto_interp.training import ExperimentConfig, train


def _train(root, experiment, name, **kw):
    cfg = dict(
        task="mul",
        p=7,
        d_model=8,
        d_mlp=16,
        num_heads=2,
        d_head=4,
        num_epochs=6,
        save_every=5,
        log_every=1000,
    )
    path = root / experiment / "runs" / name
    train(ExperimentConfig(**{**cfg, **kw}), path)
    return path


@pytest.fixture(scope="module")
def root(tmp_path_factory):
    root = tmp_path_factory.mktemp("experiments")
    _train(root, "001_toy", "a_seed0", seed=0)
    _train(root, "001_toy", "a_seed1", seed=1)
    _train(root, "002_other", "wide_seed0", seed=0, d_mlp=32)
    (root / "001_toy" / "runs" / "empty").mkdir()
    excluded = _train(root, "002_other", "bad_seed3", seed=4)
    runs.write_manifest(excluded, {**runs.read_manifest(excluded), "excluded": "mislabelled"})
    return root


def test_discover_skips_empty_and_excluded_runs(root):
    assert [r.id for r in runs.discover(root)] == [
        "001_toy/a_seed0",
        "001_toy/a_seed1",
        "002_other/wide_seed0",
    ]
    assert len(runs.discover(root, include_excluded=True)) == 4


def test_query_by_experiment_prefix_and_config_fields(root):
    assert len(runs.query(root, experiment="001")) == 2
    assert [r.name for r in runs.query(root, seed=1)] == ["a_seed1"]
    assert [r.name for r in runs.query(root, d_mlp=lambda d: d > 16)] == ["wide_seed0"]
    assert len(runs.query(root, seed=range(0, 2), d_mlp=16)) == 2
    assert len(runs.query(root, no_such_field=1)) == 0


def test_runs_sort_numbers_numerically():
    from crypto_interp.runs.index import _natural_key

    ids = ["e/x_seed10", "e/x_seed2", "e/x_seed1"]
    fake = [type("R", (), {"id": i})() for i in ids]
    assert [r.id for r in sorted(fake, key=_natural_key)] == [
        "e/x_seed1",
        "e/x_seed2",
        "e/x_seed10",
    ]


def test_table_and_loading(root):
    pop = runs.query(root, experiment="001")
    table = pop.table()
    assert list(table.index) == ["001_toy/a_seed0", "001_toy/a_seed1"]
    assert (table["latest_epoch"] == 5).all() and (table["n_checkpoints"] == 2).all()
    run = pop[1]
    assert run.load().epoch == 5 and run.load(epoch=0).epoch == 0
    with pytest.raises(FileNotFoundError):
        run.load(epoch=3)


def test_old_manifest_is_read_through(tmp_path):
    """A backfilled (version-1) manifest: identity comes from the checkpoint."""
    run_dir = _train(tmp_path, "003_old", "x_seed0")
    (run_dir / "manifest.json").write_text(json.dumps({"backfilled": True, "config": {"seed": 99}}))
    assert runs.Run.open(run_dir).task.seed == 0
