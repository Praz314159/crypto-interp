"""Persist datasets to disk so they can be inspected and reused.

A built `Dataset` is fully determined by (task, p, frac_train, seed). We cache
to `datasets/{task}_p{p}_ft{frac_train}_seed{seed}.pt` and load if present.

Usage:
    from data import load_or_build
    ds = load_or_build("mul", p=113, frac_train=0.3, seed=0)
"""

from pathlib import Path

import torch

from .base import Dataset

DATASETS_DIR = Path(__file__).resolve().parent.parent / "datasets"


def cache_path(task: str, p: int, frac_train: float, seed: int) -> Path:
    name = f"{task}_p{p}_ft{frac_train:.3f}_seed{seed}.pt"
    return DATASETS_DIR / name


def save(ds: Dataset, path: Path | None = None) -> Path:
    path = path or cache_path(ds.task, ds.p, _infer_frac_train(ds), seed=0)
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


def _infer_frac_train(ds: Dataset) -> float:
    return ds.n_train / (ds.n_train + ds.n_test)


def load_or_build(task: str, p: int, frac_train: float = 0.3, seed: int = 0) -> Dataset:
    path = cache_path(task, p, frac_train, seed)
    if path.exists():
        ds = load(path)
        return ds
    # Lazy import to avoid circular imports
    from . import build as build_fn

    ds = build_fn(task=task, p=p, frac_train=frac_train, seed=seed)
    save(ds, path)
    return ds


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Pre-generate and save datasets.")
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--p", type=int, default=113)
    parser.add_argument("--frac-train", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true",
                        help="Regenerate even if cached file exists.")
    args = parser.parse_args()

    path = cache_path(args.task, args.p, args.frac_train, args.seed)
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
