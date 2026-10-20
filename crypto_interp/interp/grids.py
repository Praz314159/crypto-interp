"""The two input grids and forward passes over them.

Every analysis works on one of two sets of value-token pairs (a, b):

- the **full grid**: a, b in 0..V-1 for the V value tokens, all V² inputs,
  lexicographic in (a, b). For ``mul`` and ``add`` this is the whole dataset.
- the **group grid**: a, b over the tokens of the task's cyclic group
  (``ds.structure.tokens``), the n² pairs on which the group operation and its
  characters live, in the same lexicographic order. For ``add`` mod n it is
  the full grid; for ``mul`` mod p it is the units, the full grid without
  token 0.

``full_grid_inputs`` / ``group_grid_inputs`` build the token tensors,
``as_full_grid`` reshapes a flat full-grid tensor to (V, V, ...), and
``to_group`` takes the group view of a full-grid tensor (flat or 2-D).

A dataset need not cover the full grid (``mul_units`` covers only the group),
so per-example masks such as ``ds.test_mask`` are placed on the group grid by
their (a, b) tokens with :func:`group_mask`, not by reshaping.
"""

from __future__ import annotations

import einops
import torch


def full_grid_inputs(ds) -> torch.Tensor:
    """All (a, b, '=') triples over the value tokens 0..p-1, lexicographic. Shape (p², 3)."""
    p = ds.p
    aa, bb = torch.meshgrid(torch.arange(p), torch.arange(p), indexing="ij")
    eq = torch.full_like(aa, ds.eq_token)
    return torch.stack([aa, bb, eq], dim=-1).reshape(-1, 3)


def group_grid_inputs(ds) -> torch.Tensor:
    """All (a, b, '=') triples for a, b over the group's tokens, lexicographic. Shape (n², 3)."""
    return to_group(full_grid_inputs(ds), ds.structure).reshape(-1, 3)


def group_mask(ds, mask: torch.Tensor) -> torch.Tensor:
    """A per-example mask of ``ds`` (e.g. ``ds.test_mask``) as an (n, n) bool grid
    over the group. Pairs the dataset lacks, or with a token outside the group, are False."""
    s = ds.structure
    position = torch.full((ds.p,), -1, dtype=torch.long)
    position[torch.as_tensor(s.token_array)] = torch.arange(s.n)
    i, j = position[ds.inputs[:, 0]], position[ds.inputs[:, 1]]
    keep = mask.bool().cpu() & (i >= 0) & (j >= 0)
    out = torch.zeros(s.n, s.n, dtype=torch.bool)
    out[i[keep], j[keep]] = True
    return out


def as_full_grid(t: torch.Tensor, p: int) -> torch.Tensor:
    """Reshape a flat full-grid tensor (p², ...) to (p, p, ...)."""
    return einops.rearrange(t, "(a b) ... -> a b ...", a=p, b=p)


def to_group(t: torch.Tensor, structure) -> torch.Tensor:
    """Group view (n, n, ...) of a full-grid tensor, flat (V², ...) or (V, V, ...)."""
    V = structure.n_values
    if t.shape[0] == V * V:
        t = as_full_grid(t, V)
    if t.shape[:2] != (V, V):
        raise ValueError(f"expected a full-grid tensor for {V} values, got shape {tuple(t.shape)}")
    idx = torch.as_tensor(structure.token_array, device=t.device)
    return t[idx][:, idx]


@torch.no_grad()
def compute_logits_grid(model, ds, W_E_override: torch.Tensor | None = None) -> torch.Tensor:
    """Final-position logits over the group grid, shape (n, n, vocab), float64.

    If ``W_E_override`` is given it is temporarily swapped into the embedding
    via :func:`crypto_interp.interp.interventions.weight_patch` and restored
    on exit (so an exception in the forward pass cannot leak the override).
    """
    n = ds.structure.n
    inputs = group_grid_inputs(ds).to(next(model.parameters()).device)
    if W_E_override is None:
        logits = model(inputs)[:, -1, :].double()
    else:
        from .interventions import weight_patch

        with weight_patch(model, "embed.W_E", W_E_override):
            logits = model(inputs)[:, -1, :].double()
    return logits.reshape(n, n, -1)


@torch.no_grad()
def compute_activation_grid(model, ds, hook: str) -> torch.Tensor:
    """Final-position activations at ``hook`` over the group grid.

    Shape (n, n, width); ``hook`` selects the width:
    ``"blocks.0.hook_mlp_out"`` -> d_model, ``"blocks.0.mlp.hook_post"`` -> d_mlp.
    """
    n = ds.structure.n
    inputs = group_grid_inputs(ds).to(next(model.parameters()).device)
    cache: dict[str, torch.Tensor] = {}
    model.remove_all_hooks()
    model.cache_all(cache)
    try:
        model(inputs)
    finally:
        model.remove_all_hooks()
    act = cache[hook][:, -1, :].double()
    return act.reshape(n, n, -1)
