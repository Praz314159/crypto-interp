"""Cache activations on all dataset inputs.

The model has HookPoints at every internal activation. We attach a save-hook
to each, run forward on the full p² input batch, and return a dict mapping
hook-name -> tensor. The caller then reshapes to (p, p, ...) for 2D analysis.
"""

import einops
import torch

from ..data.base import Dataset


@torch.no_grad()
def cache_all(model, ds: Dataset) -> dict:
    """Run the model on every (a, b) pair in the dataset and cache activations.

    Returns a dict of activation tensors, indexed by hook name. Each tensor's
    first dimension has size p² (same order as `ds.inputs`); reshape to
    (p, p, ...) downstream when needed.
    """
    cache: dict[str, torch.Tensor] = {}
    model.remove_all_hooks()
    model.cache_all(cache)

    # Use the canonical full p² input (lexicographically ordered so reshape works).
    p = ds.p
    eq = ds.eq_token
    all_inputs = torch.tensor(
        [(a, b, eq) for a in range(p) for b in range(p)],
        dtype=torch.long,
        device=next(model.parameters()).device,
    )
    logits = model(all_inputs)  # (p², 3, vocab)
    cache["logits"] = logits.detach()
    cache["logits_final"] = logits[:, -1, : ds.n_answer_tokens].detach()
    cache["all_inputs"] = all_inputs

    model.remove_all_hooks()
    return cache


def reshape_pp(tensor: torch.Tensor, p: int) -> torch.Tensor:
    """Reshape a (p², ...) tensor to (p, p, ...). Assumes lexicographic (a, b)."""
    return einops.rearrange(tensor, "(a b) ... -> a b ...", a=p, b=p)


def summary(cache: dict) -> None:
    for k, v in cache.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k:40s} shape={tuple(v.shape)} dtype={v.dtype}")
