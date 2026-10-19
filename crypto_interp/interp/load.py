"""Load a trained checkpoint into a (model, dataset, checkpoint) triple."""

from pathlib import Path

from ..runs.checkpoint import Checkpoint, checkpoint_paths, load_checkpoint


def load_run(
    ckpt_path: str | Path,
    device: str = "cpu",
    datasets_dir: str | Path | None = None,
):
    """Load a checkpoint and return ``(model, dataset, Checkpoint)``.

    Reads checkpoints of any schema. The dataset is rebuilt from the
    checkpoint's task spec, in memory unless ``datasets_dir`` names a cache.
    """
    # map_location=device lets checkpoints trained on another device (e.g. a GPU)
    # load on this machine; without it CUDA-saved tensors fail on a CPU box.
    ckpt: Checkpoint = load_checkpoint(ckpt_path, map_location=device)
    ds = ckpt.task.build(cache_dir=datasets_dir)
    return ckpt.build_model(device), ds, ckpt


def latest_checkpoint(run_dir: str | Path) -> Path:
    """The run's highest-epoch checkpoint (quarantined ``_strays/`` are ignored)."""
    ckpts = checkpoint_paths(run_dir)
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints in {run_dir}")
    return ckpts[-1]
