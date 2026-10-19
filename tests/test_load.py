import torch

from crypto_interp.interp import load_run
from crypto_interp.models import Transformer, TransformerConfig


def test_load_run_keeps_architecture(tmp_path):
    """A 2-layer checkpoint loads as a 2-layer model with identical weights."""
    p = 7
    cfg = {
        "task": "mul",
        "p": p,
        "frac_train": 0.3,
        "seed": 0,
        "d_model": 8,
        "d_mlp": 16,
        "num_heads": 2,
        "d_head": 4,
        "n_ctx": 3,
        "num_layers": 2,
    }
    torch.manual_seed(0)
    model = Transformer(
        TransformerConfig(
            d_vocab=p + 1,
            **{
                k: cfg[k]
                for k in ("d_model", "d_mlp", "num_heads", "d_head", "n_ctx", "num_layers")
            },
        )
    )
    ckpt = tmp_path / "run" / "checkpoint_000000.pt"
    ckpt.parent.mkdir()
    torch.save({"model_state": model.state_dict(), "config": cfg}, ckpt)

    loaded, ds, _ = load_run(ckpt, datasets_dir=tmp_path / "datasets")

    assert len(loaded.blocks) == 2
    for (name, a), (_, b) in zip(model.state_dict().items(), loaded.state_dict().items()):
        assert torch.equal(a, b), name
    assert torch.equal(model(ds.inputs), loaded(ds.inputs))
