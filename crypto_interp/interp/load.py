"""Load a trained checkpoint into a (model, dataset, checkpoint) triple."""

from pathlib import Path

from ..runs.checkpoint import Checkpoint, latest_checkpoint, load_checkpoint

__all__ = ["latest_checkpoint", "load_run"]


def load_run(ckpt_path: str | Path, device: str = "cpu"):
    """Load a checkpoint and return ``(model, dataset, Checkpoint)``.

    Reads checkpoints of any schema. The dataset is rebuilt from the
    checkpoint's task spec.
    """
    # map_location=device lets checkpoints trained on another device (e.g. a GPU)
    # load on this machine; without it CUDA-saved tensors fail on a CPU box.
    ckpt: Checkpoint = load_checkpoint(ckpt_path, map_location=device)
    ds = ckpt.task.build()
    return ckpt.build_model(device), ds, ckpt
