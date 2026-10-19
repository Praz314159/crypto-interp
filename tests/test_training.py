from crypto_interp.runs import checkpoint_paths, load_checkpoint
from crypto_interp.training import ExperimentConfig, resume, train


def _cfg(**kw):
    kw.setdefault("num_epochs", 20)
    return ExperimentConfig(
        task="mul",
        p=7,
        d_model=8,
        d_mlp=16,
        num_heads=2,
        d_head=4,
        save_every=10,
        log_every=1000,
        **kw,
    )


def test_train_and_resume_write_schema_2(tmp_path):
    run = tmp_path / "run"
    train(_cfg(), run, tmp_path / "datasets")

    ckpts = checkpoint_paths(run)
    assert [load_checkpoint(c).epoch for c in ckpts] == [0, 10, 19]
    last = load_checkpoint(ckpts[-1])
    assert last.source_schema == 2 and len(last.train_losses) == 20

    resume(_cfg(), ckpts[-1], additional_epochs=5, datasets_dir=tmp_path / "datasets")
    resumed = load_checkpoint(checkpoint_paths(run)[-1])
    assert resumed.source_schema == 2
    assert (resumed.epoch, len(resumed.train_losses)) == (24, 25)
