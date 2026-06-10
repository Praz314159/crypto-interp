"""Hook-based interventions — the activation-level intervention primitive.

Two TL-style verbs built on the model's existing ``HookPoint`` infrastructure:

- :func:`run_with_hooks` — run a forward pass with hooks attached at named
  hook points; cleanly remove them after. Returns logits.
- :func:`hooks` — context manager equivalent for cases where you need cleanup
  around several forward calls.

Hook function signature follows the model's existing convention:
::

    def my_hook(activation: Tensor, name: str) -> Tensor | None:
        # return a tensor to replace the activation; return None to pass through.

Three small intervention factories are provided as convenience closures:

- :func:`zero_hook` — zero the activation (optionally only a slice).
- :func:`patch_hook` — replace the activation with a tensor (optionally a slice).
- :func:`project_hook` — project onto a basis and keep only specified rows.

Hook names accept short suffixes (e.g., ``"mlp_post"`` resolves to
``"blocks.0.mlp.hook_post"``) when the short form is unambiguous — the same
convention used by :class:`ActivationCache`.
"""

from __future__ import annotations

import contextlib
from typing import Callable, Sequence

import torch


HookFn = Callable[[torch.Tensor, str], "torch.Tensor | None"]


def _resolve_hook_name(model, name: str) -> str:
    """Resolve ``name`` (long or short) to the model's HookPoint full name.

    Uses the same short-name convention as :class:`ActivationCache` — drops
    ``blocks`` / digit path segments, strips ``hook_`` prefixes, joins with ``_``.
    """
    from .cache import _short_name
    by_name = {hp.name: hp for hp in model.hook_points()}
    if name in by_name:
        return name
    short_to_long: dict[str, str] = {}
    collisions: dict[str, list[str]] = {}
    for n in by_name:
        s = _short_name(n)
        collisions.setdefault(s, []).append(n)
    for s, longs in collisions.items():
        if len(longs) == 1:
            short_to_long[s] = longs[0]
    if name in short_to_long:
        return short_to_long[name]
    if name in collisions:
        raise KeyError(f"ambiguous short name {name!r}: matches {collisions[name]}")
    raise KeyError(f"hook name {name!r} not found; short names: {sorted(short_to_long)}")


@contextlib.contextmanager
def hooks(model, fwd_hooks: Sequence[tuple[str, HookFn]]):
    """Context manager: attach ``(name, fn)`` hooks on entry, remove on exit.

    Cleanup is best-effort — exceptions during forward are propagated.
    """
    by_name = {hp.name: hp for hp in model.hook_points()}
    attached = []
    try:
        for raw_name, fn in fwd_hooks:
            name = _resolve_hook_name(model, raw_name)
            hp = by_name[name]
            hp.add_hook(fn)
            attached.append(hp)
        yield model
    finally:
        for hp in attached:
            hp.remove_hooks()


@torch.no_grad()
def run_with_hooks(model, inputs: torch.Tensor,
                   fwd_hooks: Sequence[tuple[str, HookFn]]) -> torch.Tensor:
    """Run forward with hooks attached, then clean up. Returns logits."""
    with hooks(model, fwd_hooks):
        return model(inputs)


# ---------- intervention factories (closures that produce hook fns) ----------

def zero_hook(slicer: tuple | None = None) -> HookFn:
    """Hook that zeros out the activation (or the slice indexed by ``slicer``).

    ``slicer`` is an indexing tuple suitable for tensor ``__getitem__`` (e.g.,
    ``(slice(None), -1)`` to zero only the final position).
    """
    def _h(act: torch.Tensor, name: str):
        if slicer is None:
            return torch.zeros_like(act)
        out = act.clone()
        out[slicer] = 0
        return out
    return _h


def patch_hook(value: torch.Tensor, slicer: tuple | None = None) -> HookFn:
    """Hook that replaces the activation (or a slice) with ``value``."""
    def _h(act: torch.Tensor, name: str):
        if slicer is None:
            return value.to(act.dtype).to(act.device)
        out = act.clone()
        v = value[slicer] if (isinstance(value, torch.Tensor) and value.shape == act.shape) else value
        out[slicer] = v
        return out
    return _h


def project_hook(basis: torch.Tensor, keep_mask: torch.Tensor) -> HookFn:
    """Hook that projects the last-dim of the activation onto ``basis`` rows
    selected by ``keep_mask`` and reconstructs.

    Args:
        basis:     (k, d) tensor; each row is a basis vector for the last dim.
        keep_mask: (k,) bool tensor selecting basis rows to keep.

    Activation shape ``(..., d)`` → coefficients ``(..., k)`` → masked → reconstructed ``(..., d)``.
    """
    def _h(act: torch.Tensor, name: str):
        b = basis.to(act.dtype).to(act.device)
        m = keep_mask.to(act.dtype).to(act.device)
        coef = torch.einsum("kd,...d->...k", b, act)
        coef = coef * m
        return torch.einsum("kd,...k->...d", b, coef)
    return _h
