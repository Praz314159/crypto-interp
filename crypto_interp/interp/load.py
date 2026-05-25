"""Load a trained checkpoint into a (model, dataset, ckpt) triple."""

from pathlib import Path

import torch

from .. import data
from ..models import Transformer, TransformerConfig


def _infer_datasets_dir(ckpt_path: Path) -> Path:
    """Convention: checkpoints live at ``<experiment>/runs/<tag>/checkpoint_*.pt``,
    so the dataset cache for that experiment is ``<experiment>/datasets/``.
    """
    return ckpt_path.resolve().parents[2] / "datasets"


def load_run(
    ckpt_path: str | Path,
    device: str = "cpu",
    datasets_dir: str | Path | None = None,
):
    """Load a checkpoint and return (model, dataset, ckpt_dict).

    ``datasets_dir`` defaults to ``<ckpt>/../../datasets`` per repo convention.
    Pass it explicitly when loading a checkpoint from a non-standard location.
    """
    ckpt_path = Path(ckpt_path)
    # map_location=device lets checkpoints trained on another device (e.g. a Colab
    # GPU) load on this machine; without it CUDA-saved tensors fail on a CPU box.
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg_dict = ckpt["config"]

    if datasets_dir is None:
        # Prefer the datasets_dir baked into the checkpoint, but only if it still
        # exists — checkpoints trained elsewhere (e.g. Colab) bake in a path that
        # isn't present locally; in that case infer from the checkpoint location.
        baked = cfg_dict.get("datasets_dir")
        datasets_dir = baked if (baked and Path(baked).exists()) else _infer_datasets_dir(ckpt_path)
    datasets_dir = Path(datasets_dir)

    ds = data.load_or_build(
        cache_dir=datasets_dir,
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
