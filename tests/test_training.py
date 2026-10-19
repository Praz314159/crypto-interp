import torch

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


def test_train_writes_schema_2_and_resume_continues_exactly(tmp_path):
    run = tmp_path / "run"
    train(_cfg(), run)

    ckpts = checkpoint_paths(run)
    assert [load_checkpoint(c).epoch for c in ckpts] == [0, 10, 19]
    last = load_checkpoint(ckpts[-1])
    assert last.source_schema == 2 and len(last.train_losses) == 20

    resume(_cfg(), ckpts[-1], additional_epochs=5)
    resumed = load_checkpoint(checkpoint_paths(run)[-1])

    straight = tmp_path / "straight"
    train(_cfg(num_epochs=25), straight)
    reference = load_checkpoint(checkpoint_paths(straight)[-1])

    assert resumed.epoch == reference.epoch == 24
    assert resumed.train_losses == reference.train_losses
    for k, v in reference.model_state.items():
        assert torch.equal(resumed.model_state[k], v), k
