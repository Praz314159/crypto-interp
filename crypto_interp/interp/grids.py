"""The two input grids and forward passes over them.

Every analysis works on one of two orderings of the value-token pairs (a, b):

- the **full grid**: a, b in 0..p-1, all p² inputs, lexicographic in (a, b).
  This is the dataset the model is trained and evaluated on.
- the **units grid**: a, b in 1..p-1, the (p-1)² pairs in (Z/p)* x (Z/p)*,
  where the multiplicative characters live. It is exactly the full grid
  with row and column 0 removed, in the same order.

``full_grid_inputs`` / ``units_grid_inputs`` build the token tensors,
``as_full_grid`` reshapes a flat full-grid tensor to (p, p, ...), and
``to_units`` takes the units view of a full-grid tensor (flat or 2-D).
"""

from __future__ import annotations

import einops
import torch


def full_grid_inputs(ds) -> torch.Tensor:
    """All (a, b, '=') triples for a, b in 0..p-1, lexicographic. Shape (p², 3)."""
    p = ds.p
    aa, bb = torch.meshgrid(torch.arange(p), torch.arange(p), indexing="ij")
    eq = torch.full_like(aa, ds.eq_token)
    return torch.stack([aa, bb, eq], dim=-1).reshape(-1, 3)


def units_grid_inputs(ds) -> torch.Tensor:
    """All (a, b, '=') triples for a, b in 1..p-1, lexicographic. Shape ((p-1)², 3)."""
    return to_units(full_grid_inputs(ds), ds.p).reshape(-1, 3)


def as_full_grid(t: torch.Tensor, p: int) -> torch.Tensor:
    """Reshape a flat full-grid tensor (p², ...) to (p, p, ...)."""
    return einops.rearrange(t, "(a b) ... -> a b ...", a=p, b=p)


def to_units(t: torch.Tensor, p: int) -> torch.Tensor:
    """Units view (p-1, p-1, ...) of a full-grid tensor, flat (p², ...) or (p, p, ...)."""
    if t.shape[0] == p * p:
        t = as_full_grid(t, p)
    if t.shape[:2] != (p, p):
        raise ValueError(f"expected a full-grid tensor for p={p}, got shape {tuple(t.shape)}")
    return t[1:, 1:]


@torch.no_grad()
def compute_logits_grid(model, ds, W_E_override: torch.Tensor | None = None) -> torch.Tensor:
    """Final-position logits over the units grid, shape (p-1, p-1, vocab), float64.

    If ``W_E_override`` is given it is temporarily swapped into the embedding
    via :func:`crypto_interp.interp.interventions.weight_patch` and restored
    on exit (so an exception in the forward pass cannot leak the override).
    """
    p = ds.p
    inputs = units_grid_inputs(ds).to(next(model.parameters()).device)
    if W_E_override is None:
        logits = model(inputs)[:, -1, :].double()
    else:
        from .interventions import weight_patch

        with weight_patch(model, "embed.W_E", W_E_override):
            logits = model(inputs)[:, -1, :].double()
    return logits.reshape(p - 1, p - 1, -1)


@torch.no_grad()
def compute_activation_grid(model, ds, hook: str) -> torch.Tensor:
    """Final-position activations at ``hook`` over the units grid.

    Shape (p-1, p-1, width); ``hook`` selects the width:
    ``"blocks.0.hook_mlp_out"`` -> d_model, ``"blocks.0.mlp.hook_post"`` -> d_mlp.
    """
    p = ds.p
    inputs = units_grid_inputs(ds).to(next(model.parameters()).device)
    cache: dict[str, torch.Tensor] = {}
    model.remove_all_hooks()
    model.cache_all(cache)
    try:
        model(inputs)
    finally:
        model.remove_all_hooks()
    act = cache[hook][:, -1, :].double()
    return act.reshape(p - 1, p - 1, -1)
