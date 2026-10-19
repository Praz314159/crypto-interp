"""ActivationCache — wraps cached activations with short-name aliases.

An ``ActivationCache`` is produced by :func:`crypto_interp.interp.activations.run_with_cache`
(or :func:`cache_all`) after a forward pass with save-hooks attached at every
``HookPoint``. It maps hook names to the captured tensors and adds a few
ergonomic accessors:

- **Short-name resolution.** Long names like ``"blocks.0.mlp.hook_post"`` can be
  accessed as ``cache["mlp_post"]`` when only one block has that hook, or as
  ``cache["mlp_post", 0]`` to name the block (TransformerLens-style). In a
  multi-block model the bare short name is ambiguous and raises ``KeyError``.
- ``cache.final(name)`` — the final-position slice ``t[..., -1, :]``.
- ``cache.grid(name)`` — reshape the ``(p², ...)`` first dim to ``(p, p, ...)``
  for value-token (a, b) grid analyses.
- ``cache.decompose_resid()`` — return the {resid_pre, attn_out, mlp_out}
  triple that sums to the post-block residual (1-block models only).

Dict-like (``__getitem__``, ``__contains__``, ``__iter__``, ``len``, ``keys``,
``values``, ``items``) but **not** a ``dict`` subclass — keeps the surface
explicit. Treat instances as immutable from outside.
"""

from __future__ import annotations
from typing import Iterable, Iterator, Union

import einops
import torch
from jaxtyping import Float

#: A hook name: full (``"blocks.1.mlp.hook_post"``), short (``"mlp_post"``), or
#: short plus block index (``("mlp_post", 1)``).
HookKey = Union[str, tuple[str, int]]


def _short_name(full: str) -> str:
    """``blocks.0.mlp.hook_post`` → ``mlp_post``;  ``blocks.0.hook_attn_out`` → ``attn_out``.

    Path segments ``blocks`` and pure digits are dropped; segments starting
    with ``hook_`` have that prefix stripped; remaining segments are joined
    with ``_``.
    """
    out: list[str] = []
    for seg in full.split("."):
        if seg == "blocks" or seg.isdigit():
            continue
        out.append(seg[len("hook_") :] if seg.startswith("hook_") else seg)
    return "_".join(out)


def _layer_of(full: str) -> int | None:
    """``blocks.1.mlp.hook_post`` → ``1``; names outside a block → ``None``."""
    segs = full.split(".")
    if len(segs) > 1 and segs[0] == "blocks" and segs[1].isdigit():
        return int(segs[1])
    return None


def resolve_name(names: Iterable[str], key: HookKey) -> str:
    """Resolve ``key`` to exactly one of ``names``, or raise ``KeyError``.

    ``key`` may be a full name, a short name that matches a single hook, or a
    ``(short_name, layer)`` pair.
    """
    names = list(names)
    if isinstance(key, tuple):
        short, layer = key
        matches = [n for n in names if _short_name(n) == short and _layer_of(n) == layer]
        if len(matches) == 1:
            return matches[0]
        raise KeyError(f"no hook {short!r} in layer {layer}")
    if key in names:
        return key
    matches = [n for n in names if _short_name(n) == key]
    if len(matches) == 1:
        return matches[0]
    if matches:
        raise KeyError(f"{key!r} is ambiguous: matches {matches}; pass ({key!r}, layer)")
    shorts = sorted({_short_name(n) for n in names})
    raise KeyError(f"no hook matches {key!r}; short names available: {shorts}")


class ActivationCache:
    def __init__(self, cache_dict: dict, p: int | None = None):
        # detach tensors to drop graphs; pass non-tensors through (e.g., logits, all_inputs)
        self._cache = {
            k: (v.detach() if isinstance(v, torch.Tensor) else v) for k, v in cache_dict.items()
        }
        self._p = p

    # ---------- dict-like interface ----------
    def __getitem__(self, key: HookKey):
        return self._cache[self._resolve(key)]

    def __contains__(self, key: HookKey) -> bool:
        try:
            self._resolve(key)
            return True
        except KeyError:
            return False

    def __iter__(self) -> Iterator[str]:
        return iter(self._cache)

    def __len__(self) -> int:
        return len(self._cache)

    def keys(self):
        return self._cache.keys()

    def values(self):
        return self._cache.values()

    def items(self):
        return self._cache.items()

    def __repr__(self) -> str:
        return f"ActivationCache({len(self._cache)} keys, p={self._p})"

    # ---------- name resolution ----------
    def _resolve(self, key: HookKey) -> str:
        return resolve_name(self._cache, key)

    # ---------- ergonomic accessors ----------
    def final(self, key: HookKey) -> Float[torch.Tensor, "batch ..."]:
        """Slice the final sequence position: ``t[..., -1, :]`` for ``(batch, seq, *)`` shapes."""
        t = self[key]
        if t.dim() < 2:
            return t
        # Standard hook shapes have seq as the second-to-last axis (batch, seq, d) or
        # for attention scores (batch, heads, seq_q, seq_k). The final-token slice is on
        # the query/seq axis; for (batch, seq, d) that's [..., -1, :]; for attention
        # patterns it's [..., -1, :]. Both behave the same for our use.
        return t[..., -1, :]

    def grid(self, key: HookKey, p: int | None = None) -> Float[torch.Tensor, "p p ..."]:
        """Reshape ``(p², ...)`` → ``(p, p, ...)`` over value-token (a, b) inputs.

        Requires p (passed at construction or here).
        """
        p = p or self._p
        if p is None:
            raise ValueError(
                "grid() requires p (pass at construction via run_with_cache+ds, or here)"
            )
        return einops.rearrange(self[key], "(a b) ... -> a b ...", a=p, b=p)

    def decompose_resid(self) -> dict[str, torch.Tensor]:
        """Per-component contributions to the residual stream (1-block model).

        Returns ``{"resid_pre", "attn_out", "mlp_out"}``. Their sum equals
        ``resid_post`` (the input to the unembed). Raises ``ValueError`` on a
        multi-block cache; index components per block there, e.g.
        ``cache["mlp_out", 1]``.
        """
        layers = {_layer_of(k) for k in self._cache} - {None}
        if len(layers) > 1:
            raise ValueError(
                f"decompose_resid is defined for 1-block models; cache has blocks {sorted(layers)}"
            )
        return {
            "resid_pre": self["resid_pre"],
            "attn_out": self["attn_out"],
            "mlp_out": self["mlp_out"],
        }

    # ---------- internal mutation (used by run_with_cache to attach logits) ----------
    def _put(self, key: str, value) -> None:
        """Add a non-hook entry (e.g., logits). Internal only."""
        self._cache[key] = value.detach() if isinstance(value, torch.Tensor) else value
