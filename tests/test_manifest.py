import json

from crypto_interp.runs import read_manifest, rebuild_manifest
from crypto_interp.training import ExperimentConfig, train


def _train(path, **kw):
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
    train(ExperimentConfig(**{**cfg, **kw}), path, path.parent / "datasets")
    return path


def test_training_writes_a_version_2_manifest(tmp_path):
    m = read_manifest(_train(tmp_path / "run", seed=1))
    assert m["manifest_version"] == 2
    assert m["task"] == {"task": "mul", "p": 7, "frac_train": 0.3, "seed": 1}
    assert m["model_config"]["d_vocab"] == 8
    assert {"device", "hostname", "started_at"} <= set(m["provenance"])


def test_rebuild_upgrades_a_backfilled_manifest(tmp_path):
    """Identity comes from the latest checkpoint; provenance and exclusion are kept;
    keys that restate the directory are dropped."""
    run_dir = _train(tmp_path / "run")
    old = {
        "backfilled": True,
        "source": "runpod-a100",
        "config": {"seed": 99},
        "n_checkpoints": 7,
        "checkpoint_steps": [0],
        "excluded": "reason",
    }
    (run_dir / "manifest.json").write_text(json.dumps(old))

    new = rebuild_manifest(run_dir)

    assert new == read_manifest(run_dir)
    assert new["manifest_version"] == 2 and new["task"]["seed"] == 0
    assert new["provenance"] == {"backfilled": True, "source": "runpod-a100"}
    assert new["excluded"] == "reason"
    assert not {"config", "n_checkpoints", "checkpoint_steps"} & set(new)
