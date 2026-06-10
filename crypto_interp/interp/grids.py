"""Forward passes over the full (a, b) input grid.

Consolidates the per-script ``compute_logits_grid`` / ``compute_mlp_output_grid``
helpers. Prime-parametric (everything derives from ``ds.p``).
"""
from __future__ import annotations

import torch


def ab_grid_inputs(ds) -> torch.Tensor:
    """All (a, b, '=') triples for a, b in 1..p-1, lexicographic in (a, b).
    Returns a (``(p-1)**2``, 3) long tensor."""
    p = ds.p
    aa, bb = torch.meshgrid(torch.arange(1, p), torch.arange(1, p), indexing="ij")
    eq = torch.full_like(aa, ds.eq_token)
    return torch.stack([aa, bb, eq], dim=-1).reshape(-1, 3)


@torch.no_grad()
def compute_logits_grid(model, ds, W_E_override: torch.Tensor | None = None) -> torch.Tensor:
    """Final-position logits over the (a, b) grid, shape (p-1, p-1, vocab).

    If ``W_E_override`` is given it is temporarily swapped into the embedding
    via :func:`crypto_interp.interp.interventions.weight_patch` and restored
    on exit (so an exception in the forward pass cannot leak the override).
    """
    p = ds.p
    inputs = ab_grid_inputs(ds).to(next(model.parameters()).device)
    if W_E_override is None:
        logits = model(inputs)[:, -1, :].double()
    else:
        from .interventions import weight_patch
        with weight_patch(model, "embed.W_E", W_E_override):
            logits = model(inputs)[:, -1, :].double()
    return logits.reshape(p - 1, p - 1, -1)


@torch.no_grad()
def compute_activation_grid(model, ds, hook: str) -> torch.Tensor:
    """Final-position activations at ``hook`` over the (a, b) grid.

    Shape (p-1, p-1, width); ``hook`` selects the width:
    ``"blocks.0.hook_mlp_out"`` -> d_model, ``"blocks.0.mlp.hook_post"`` -> d_mlp.
    """
    p = ds.p
    inputs = ab_grid_inputs(ds).to(next(model.parameters()).device)
    cache: dict[str, torch.Tensor] = {}
    model.remove_all_hooks()
    model.cache_all(cache)
    try:
        model(inputs)
    finally:
        model.remove_all_hooks()
    act = cache[hook][:, -1, :].double()
    return act.reshape(p - 1, p - 1, -1)
