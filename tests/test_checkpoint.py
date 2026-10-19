import pytest
import torch

from crypto_interp.data import TaskSpec
from crypto_interp.models import Transformer, TransformerConfig
from crypto_interp.runs import (
    Checkpoint,
    checkpoint_paths,
    load_checkpoint,
    save_checkpoint,
)

P = 7
INPUTS = torch.tensor([[a, b, P] for a in range(P) for b in range(P)])


def _model(**arch):
    torch.manual_seed(0)
    return Transformer(
        TransformerConfig(d_vocab=P + 1, d_model=8, d_mlp=16, num_heads=2, d_head=4, **arch)
    )


def test_v2_round_trip(tmp_path):
    model = _model(num_layers=2)
    ckpt = Checkpoint(
        epoch=12,
        task=TaskSpec("mul", P, 0.3, 4),
        model_config=model.cfg,
        train_config={"lr": 1e-3, "weight_decay": 1.0},
        model_state=model.state_dict(),
        train_losses=[1.0, 0.5],
        test_losses=[2.0, 1.5],
    )
    loaded = load_checkpoint(save_checkpoint(ckpt, tmp_path / "checkpoint_000012.pt"))

    assert loaded.source_schema == 2
    assert (loaded.epoch, loaded.task, loaded.model_config) == (12, ckpt.task, model.cfg)
    assert loaded.train_losses == [1.0, 0.5]
    assert torch.equal(loaded.build_model()(INPUTS), model(INPUTS))


def test_reads_schema_1(tmp_path):
    """Schema 1: flat config without d_vocab, n_ctx or num_layers, plus a local path."""
    model = _model()
    raw = {
        "epoch": 3,
        "model_state": model.state_dict(),
        "train_losses": [1.0],
        "test_losses": [2.0],
        "config": {
            "task": "mul",
            "p": P,
            "frac_train": 0.3,
            "seed": 1,
            "d_model": 8,
            "d_mlp": 16,
            "num_heads": 2,
            "d_head": 4,
            "lr": 1e-3,
            "datasets_dir": "/somewhere/else",
        },
    }
    path = tmp_path / "checkpoint_000003.pt"
    torch.save(raw, path)

    ckpt = load_checkpoint(path)

    assert ckpt.source_schema == 1
    assert ckpt.task == TaskSpec("mul", P, 0.3, 1)
    assert ckpt.model_config == model.cfg  # d_vocab recovered from W_E
    assert "datasets_dir" not in ckpt.train_config
    assert torch.equal(ckpt.build_model()(INPUTS), model(INPUTS))
    assert load_checkpoint(save_checkpoint(ckpt, tmp_path / "v2.pt")).source_schema == 2


def test_schema_1_without_task_needs_a_default(tmp_path):
    model = _model()
    cfg = {
        "p": P,
        "frac_train": 0.3,
        "seed": 0,
        "d_model": 8,
        "d_mlp": 16,
        "num_heads": 2,
        "d_head": 4,
    }
    torch.save({"epoch": 0, "model_state": model.state_dict(), "config": cfg}, tmp_path / "c.pt")
    with pytest.raises(KeyError, match="task"):
        load_checkpoint(tmp_path / "c.pt")
    assert load_checkpoint(tmp_path / "c.pt", v1_defaults={"task": "mul"}).task.task == "mul"


def test_checkpoint_paths_orders_by_epoch_and_skips_strays(tmp_path):
    for name in ["checkpoint_000500.pt", "checkpoint_000000.pt", "checkpoint_1000000.pt"]:
        (tmp_path / name).touch()
    (tmp_path / "_strays").mkdir()
    (tmp_path / "_strays" / "checkpoint_999999.pt").touch()
    assert [p.name for p in checkpoint_paths(tmp_path)] == [
        "checkpoint_000000.pt",
        "checkpoint_000500.pt",
        "checkpoint_1000000.pt",
    ]


def test_task_spec_build_is_deterministic():
    a, b = TaskSpec("mul", 11, 0.3, 2).build(), TaskSpec("mul", 11, 0.3, 2).build()
    assert torch.equal(a.train_mask, b.train_mask)
