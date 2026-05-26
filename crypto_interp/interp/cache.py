"""ActivationCache — wraps cached activations with short-name aliases.

An ``ActivationCache`` is produced by :func:`crypto_interp.interp.activations.run_with_cache`
(or :func:`cache_all`) after a forward pass with save-hooks attached at every
``HookPoint``. It maps hook names to the captured tensors and adds a few
ergonomic accessors:

- **Short-name resolution.** Long names like ``"blocks.0.mlp.hook_post"`` can be
  accessed as ``cache["mlp_post"]`` (suffix match against ``hook_<name>``).
  Disambiguation is required if a short name matches more than one hook (which
  only happens once we have multi-block models — single-block is unambiguous).
- ``cache.final(name)`` — the final-position slice ``t[..., -1, :]``.
- ``cache.grid(name)`` — reshape the ``(p², ...)`` first dim to ``(p, p, ...)``
  for value-token (a, b) grid analyses.
- ``cache.decompose_resid()`` — return the {resid_pre, attn_out, mlp_out}
  triple that sums to the post-block residual (1-block models; will generalize
  to per-layer dicts when multi-block arrives).

Dict-like (``__getitem__``, ``__contains__``, ``__iter__``, ``len``, ``keys``,
``values``, ``items``) but **not** a ``dict`` subclass — keeps the surface
explicit. Treat instances as immutable from outside.
"""

from __future__ import annotations
from typing import Iterator

import einops
import torch


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
        out.append(seg[len("hook_"):] if seg.startswith("hook_") else seg)
    return "_".join(out)


class ActivationCache:
    def __init__(self, cache_dict: dict, p: int | None = None):
        # detach tensors to drop graphs; pass non-tensors through (e.g., logits, all_inputs)
        self._cache = {k: (v.detach() if isinstance(v, torch.Tensor) else v)
                       for k, v in cache_dict.items()}
        self._p = p
        # forward index of short names → long names; ambiguous shorts left out.
        self._short: dict[str, str] = {}
        seen: dict[str, list[str]] = {}
        for k in self._cache:
            seen.setdefault(_short_name(k), []).append(k)
        for short, longs in seen.items():
            if len(longs) == 1:
                self._short[short] = longs[0]

    # ---------- dict-like interface ----------
    def __getitem__(self, key: str): return self._cache[self._resolve(key)]
    def __contains__(self, key: str) -> bool:
        try: self._resolve(key); return True
        except KeyError: return False
    def __iter__(self) -> Iterator[str]: return iter(self._cache)
    def __len__(self) -> int: return len(self._cache)
    def keys(self): return self._cache.keys()
    def values(self): return self._cache.values()
    def items(self): return self._cache.items()
    def __repr__(self) -> str:
        return f"ActivationCache({len(self._cache)} keys, p={self._p})"

    # ---------- name resolution ----------
    def _resolve(self, key: str) -> str:
        if key in self._cache:
            return key
        if key in self._short:
            return self._short[key]
        raise KeyError(
            f"no cached hook matches {key!r}; "
            f"short names available: {sorted(self._short)}"
        )

    # ---------- ergonomic accessors ----------
    def final(self, key: str) -> torch.Tensor:
        """Slice the final sequence position: ``t[..., -1, :]`` for (batch, seq, *) shapes."""
        t = self[key]
        if t.dim() < 2:
            return t
        # Standard hook shapes have seq as the second-to-last axis (batch, seq, d) or
        # for attention scores (batch, heads, seq_q, seq_k). The final-token slice is on
        # the query/seq axis; for (batch, seq, d) that's [..., -1, :]; for attention
        # patterns it's [..., -1, :]. Both behave the same for our use.
        return t[..., -1, :]

    def grid(self, key: str, p: int | None = None) -> torch.Tensor:
        """Reshape ``(p², ...)`` → ``(p, p, ...)`` over value-token (a, b) inputs.

        Requires p (passed at construction or here).
        """
        p = p or self._p
        if p is None:
            raise ValueError("grid() requires p (pass at construction via run_with_cache+ds, or here)")
        return einops.rearrange(self[key], "(a b) ... -> a b ...", a=p, b=p)

    def decompose_resid(self) -> dict[str, torch.Tensor]:
        """Per-component contributions to the residual stream (1-block model).

        Returns ``{"resid_pre", "attn_out", "mlp_out"}``. Their sum equals
        ``resid_post`` (the input to the unembed). Extends to per-layer dicts
        in the multi-block setting later.
        """
        return {"resid_pre": self["resid_pre"],
                "attn_out": self["attn_out"],
                "mlp_out":  self["mlp_out"]}

    # ---------- internal mutation (used by run_with_cache to attach logits) ----------
    def _put(self, key: str, value) -> None:
        """Add a non-hook entry (e.g., logits). Internal only."""
        self._cache[key] = value.detach() if isinstance(value, torch.Tensor) else value
