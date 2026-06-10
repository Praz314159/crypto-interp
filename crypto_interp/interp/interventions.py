"""Weight-space interventions as context managers.

Mirror the activation-space ``hooks()`` context manager from
:mod:`crypto_interp.interp.hooks`, but for *weights*: temporarily replace
or freeze a model parameter, run something, restore on exit. Generalizes
the ``try/finally W_E.copy_(...)/.copy_(orig)`` pattern that was inlined
in ``grids.py``, ``harmonic.py``, and ``ablate.py``.

Verbs:

- :func:`weight_patch` — replace a parameter's data with a new tensor.
- :func:`ablate_char_w` — wrap :func:`crypto_interp.interp.ablate.ablate_character`
  as a context manager; zeros character ``k`` from W_E (value-token block).
- :func:`freeze_param` — set ``requires_grad=False`` on a parameter (gradient
  intervention, not an activation intervention).

All three restore the original state on exit, even if an exception is raised
during the wrapped block.
"""

from __future__ import annotations

import contextlib

import torch


def _resolve_param(model, name: str) -> torch.nn.Parameter:
    """Walk a dotted attribute path to a parameter — e.g., ``embed.W_E``,
    ``blocks.0.mlp.W_out``."""
    obj = model
    for part in name.split("."):
        obj = getattr(obj, part)
    if not isinstance(obj, torch.nn.Parameter):
        raise TypeError(f"{name!r} is not a Parameter (got {type(obj).__name__})")
    return obj


@contextlib.contextmanager
def weight_patch(model, param_name: str, new_value: torch.Tensor):
    """Temporarily replace a parameter's value; restore on exit.

    ``param_name`` is a dotted path under ``model`` (e.g., ``"embed.W_E"``,
    ``"blocks.0.mlp.W_out"``). ``new_value`` is cast to the parameter's dtype
    and device automatically.

    Example::

        with weight_patch(model, "embed.W_E", new_W_E):
            logits = model(inputs)
    """
    param = _resolve_param(model, param_name)
    orig = param.data.clone()
    try:
        param.data.copy_(new_value.to(dtype=param.dtype, device=param.device))
        yield model
    finally:
        param.data.copy_(orig)


@contextlib.contextmanager
def ablate_char_w(model, k: int, basis: torch.Tensor, ci):
    """Temporarily Gram-Schmidt-ablate character ``k`` from W_E (value-token block)
    and restore on exit.

    The ``=`` token (last column of W_E) is preserved unchanged.

    Example::

        from crypto_interp.interp import ablate_char_w
        from crypto_interp.interp.ablate import evaluate_loss
        with ablate_char_w(model, k=8, basis=basis, ci=ci):
            _, test_loss, _ = evaluate_loss(model, ds)
    """
    from .ablate import ablate_character
    p = basis.shape[0]
    W_E = model.embed.W_E
    orig = W_E.data.clone()
    new = orig.clone()
    W_ab = ablate_character(orig[:, :p], basis, ci, k)
    new[:, :p] = W_ab.to(new.dtype)
    try:
        W_E.data.copy_(new)
        yield model
    finally:
        W_E.data.copy_(orig)


@contextlib.contextmanager
def freeze_param(model, param_name: str):
    """Set ``requires_grad=False`` on a parameter; restore prior state on exit.

    Useful for ablation-during-training experiments (e.g., freeze W_E mid-run
    and see what the rest of the network does).
    """
    param = _resolve_param(model, param_name)
    orig = param.requires_grad
    try:
        param.requires_grad_(False)
        yield model
    finally:
        param.requires_grad_(orig)
