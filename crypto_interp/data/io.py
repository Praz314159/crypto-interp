"""Persist datasets to disk so they can be inspected and reused.

A built `Dataset` is fully determined by (task, p, frac_train, seed). We cache
to ``<cache_dir>/{task}_p{p}_ft{frac_train}_seed{seed}.pt`` and load if present.

The ``cache_dir`` is supplied by the caller (typically ``<experiment>/datasets/``
from the experiment config). For convenience there is no library-wide default
— experiments are explicit about where their cache lives.
"""

from pathlib import Path

import torch

from .base import Dataset


def cache_path(cache_dir: Path, task: str, p: int, frac_train: float, seed: int) -> Path:
    name = f"{task}_p{p}_ft{frac_train:.3f}_seed{seed}.pt"
    return Path(cache_dir) / name


def save(ds: Dataset, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "p": ds.p,
        "task": ds.task,
        "inputs": ds.inputs,
        "labels": ds.labels,
        "train_mask": ds.train_mask,
        "test_mask": ds.test_mask,
        "vocab_size": ds.vocab_size,
        "n_answer_tokens": ds.n_answer_tokens,
        "eq_token": ds.eq_token,
    }
    torch.save(payload, path)
    return path


def load(path: Path) -> Dataset:
    payload = torch.load(path, weights_only=False)
    return Dataset(**payload)


def load_or_build(
    cache_dir: Path,
    task: str,
    p: int,
    frac_train: float = 0.3,
    seed: int = 0,
) -> Dataset:
    path = cache_path(cache_dir, task, p, frac_train, seed)
    if path.exists():
        return load(path)
    from . import build as build_fn  # lazy to avoid circular import
    ds = build_fn(task=task, p=p, frac_train=frac_train, seed=seed)
    save(ds, path)
    return ds


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Pre-generate and save a dataset.")
    parser.add_argument("--cache-dir", type=Path, required=True,
                        help="Directory to cache the dataset under.")
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--p", type=int, default=113)
    parser.add_argument("--frac-train", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true",
                        help="Regenerate even if cached file exists.")
    args = parser.parse_args()

    path = cache_path(args.cache_dir, args.task, args.p, args.frac_train, args.seed)
    if path.exists() and not args.force:
        print(f"Already exists: {path}")
        ds = load(path)
    else:
        from . import build as build_fn
        ds = build_fn(task=args.task, p=args.p,
                      frac_train=args.frac_train, seed=args.seed)
        save(ds, path)
        print(f"Saved: {path}")
    print(f"task={ds.task}, p={ds.p}, total={len(ds.inputs)}, "
          f"train={ds.n_train}, test={ds.n_test}, vocab={ds.vocab_size}")
