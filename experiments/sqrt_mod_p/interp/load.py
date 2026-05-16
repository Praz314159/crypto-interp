"""Load a trained checkpoint into a model + dataset, ready for analysis."""

from pathlib import Path

import torch

import data
from model import Transformer, TransformerConfig


def load_run(ckpt_path: str | Path, device: str = "cpu"):
    """Load a checkpoint and return (model, dataset, ckpt_dict)."""
    ckpt = torch.load(ckpt_path, weights_only=False)
    cfg_dict = ckpt["config"]

    ds = data.load_or_build(
        task=cfg_dict["task"],
        p=cfg_dict["p"],
        frac_train=cfg_dict["frac_train"],
        seed=cfg_dict["seed"],
    )
    cfg = TransformerConfig(
        d_vocab=ds.vocab_size,
        d_model=cfg_dict["d_model"],
        d_mlp=cfg_dict["d_mlp"],
        num_heads=cfg_dict["num_heads"],
        d_head=cfg_dict["d_head"],
    )
    model = Transformer(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ds, ckpt


def latest_checkpoint(run_dir: str | Path) -> Path:
    run_dir = Path(run_dir)
    ckpts = sorted(run_dir.glob("checkpoint_*.pt"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints in {run_dir}")
    return ckpts[-1]
