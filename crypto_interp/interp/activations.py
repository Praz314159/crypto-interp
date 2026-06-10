"""Forward-pass + cache utilities.

Two verbs:

- :func:`run_with_cache` — forward with every ``HookPoint`` saved into an
  :class:`ActivationCache`. The TL-style "cache once, query many times"
  primitive.
- :func:`cache_all` — convenience wrapper that runs the model on the full
  ``p²`` grid of (a, b) value-token pairs from a :class:`Dataset` and adds
  ``logits``, ``logits_final``, ``all_inputs`` to the cache. The cache's
  ``p`` is set so ``cache.grid(name)`` works without a manual argument.

``reshape_pp`` and ``summary`` are retained as small helpers.
"""

from __future__ import annotations

import einops
import torch

from ..data.base import Dataset
from .cache import ActivationCache


@torch.no_grad()
def run_with_cache(model, inputs: torch.Tensor) -> tuple[torch.Tensor, ActivationCache]:
    """Run forward and capture every HookPoint's activation.

    Returns ``(logits, cache)``. Cleans up its own hooks before returning so
    subsequent forward passes are not affected.
    """
    cache_dict: dict = {}
    model.remove_all_hooks()
    model.cache_all(cache_dict)
    try:
        logits = model(inputs)
    finally:
        model.remove_all_hooks()
    return logits, ActivationCache(cache_dict)


@torch.no_grad()
def cache_all(model, ds: Dataset) -> ActivationCache:
    """Run the model on every (a, b) pair in the dataset and return an :class:`ActivationCache`.

    Convenience wrapper around :func:`run_with_cache` that:

    - Builds the canonical ``p²`` lexicographic input grid for value tokens.
    - Adds ``logits`` (full), ``logits_final`` (answer-token slice), and
      ``all_inputs`` to the cache.
    - Sets the cache's ``p`` so ``cache.grid(name)`` works without args.
    """
    p = ds.p
    eq = ds.eq_token
    all_inputs = torch.tensor(
        [(a, b, eq) for a in range(p) for b in range(p)],
        dtype=torch.long,
        device=next(model.parameters()).device,
    )
    logits, cache = run_with_cache(model, all_inputs)
    cache._put("logits", logits)
    cache._put("logits_final", logits[:, -1, : ds.n_answer_tokens])
    cache._put("all_inputs", all_inputs)
    cache._p = p
    return cache


def reshape_pp(tensor: torch.Tensor, p: int) -> torch.Tensor:
    """Reshape a ``(p², ...)`` tensor to ``(p, p, ...)`` assuming lexicographic (a, b)."""
    return einops.rearrange(tensor, "(a b) ... -> a b ...", a=p, b=p)


def summary(cache) -> None:
    """Print a one-line summary of each cache key. Accepts dict or ActivationCache."""
    for k, v in cache.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k:40s} shape={tuple(v.shape)} dtype={v.dtype}")
