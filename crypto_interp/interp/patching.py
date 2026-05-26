"""Activation patching — the canonical causal probe.

Given a forward pass that *worked* (clean) and one that *didn't* (corrupted),
patch the clean activation at a chosen hook into the corrupted forward and
measure how the answer moves. If the metric shifts toward the clean answer,
the patched activation was carrying the information that mattered.

For our 1-block, 3-token (a, b, =) setting the iteration axis is **position**
— we patch each token's activation in turn and watch the metric. (When we
add multi-block models the layer axis comes for free; head-axis bookkeeping
is deferred until attention does nontrivial routing.)

Two verbs:

- :func:`act_patch` — generic. ``(model, corrupted_inputs, clean_cache,
  hook_name, metric_fn[, positions])`` → tensor of metrics, one per patched
  position. ``positions=None`` patches each position separately;
  ``positions="all"`` patches the entire activation at once and returns a
  scalar.
- :func:`patch_mlp_out` / :func:`patch_resid_pre` / :func:`patch_attn_out`
  — thin wrappers that fix the hook name. The wrappers are the verbs you'd
  type interactively; ``act_patch`` is the engine.

Modeled on TransformerLens's ``generic_activation_patch`` but stripped
(no attention-head iteration, no pandas DataFrames, no tqdm). ~120 LOC.
"""

from __future__ import annotations

from typing import Callable, Sequence, Union

import torch

from .cache import ActivationCache
from .hooks import _resolve_hook_name, run_with_hooks


MetricFn = Callable[[torch.Tensor], float]


def _patch_position_hook(clean_act: torch.Tensor, pos: int):
    """Closure that returns a forward-hook replacing ``act[:, pos]`` with ``clean_act[:, pos]``."""
    def _h(act: torch.Tensor, name: str):
        out = act.clone()
        out[:, pos] = clean_act[:, pos].to(out.dtype).to(out.device)
        return out
    return _h


def _patch_all_hook(clean_act: torch.Tensor):
    """Closure that returns a forward-hook replacing the activation entirely."""
    def _h(act: torch.Tensor, name: str):
        return clean_act.to(act.dtype).to(act.device)
    return _h


@torch.no_grad()
def act_patch(model,
              corrupted_inputs: torch.Tensor,
              clean_cache: ActivationCache,
              hook_name: str,
              metric_fn: MetricFn,
              positions: Union[Sequence[int], str, None] = None,
              ) -> Union[torch.Tensor, float]:
    """Causal probe — patch the clean activation at ``hook_name`` into the
    corrupted forward, run forward, measure ``metric_fn(patched_logits)``.

    Args:
        model: the network.
        corrupted_inputs: token tensor to run forward on (``(batch, seq)``).
        clean_cache: an :class:`ActivationCache` from a *clean* forward pass.
            Must contain ``hook_name``; the activation's batch and sequence
            dimensions must match ``corrupted_inputs``.
        hook_name: long name (``"blocks.0.hook_mlp_out"``) or short name
            (``"mlp_out"``).
        metric_fn: callable ``(logits_tensor) -> float`` (e.g.,
            ``Session.test_loss_metric`` or a custom logit-difference).
        positions:
            - ``None`` (default): iterate over all sequence positions, patching
              one at a time. Returns a tensor of length ``seq``.
            - a list of int positions: iterate over only those positions.
              Returns a tensor of length ``len(positions)``.
            - ``"all"``: patch the *entire* activation at once. Returns a
              scalar (the metric on the fully-patched run).

    Returns: a tensor of metric values (length = number of positions probed),
    or a single float when ``positions="all"``.
    """
    full_name = _resolve_hook_name(model, hook_name)
    clean_act = clean_cache[full_name]
    if clean_act.shape[:2] != corrupted_inputs.shape[:2]:
        raise ValueError(
            f"clean activation at {full_name!r} has batch/seq "
            f"{tuple(clean_act.shape[:2])}, expected "
            f"{tuple(corrupted_inputs.shape[:2])} to match corrupted_inputs"
        )

    if positions == "all":
        patched = run_with_hooks(model, corrupted_inputs,
                                 [(full_name, _patch_all_hook(clean_act))])
        return float(metric_fn(patched))

    seq = corrupted_inputs.shape[1]
    if positions is None:
        positions = list(range(seq))
    positions = list(positions)

    out = torch.zeros(len(positions), dtype=torch.float64)
    for i, pos in enumerate(positions):
        if not (0 <= pos < seq):
            raise IndexError(f"position {pos} out of range [0, {seq})")
        patched = run_with_hooks(model, corrupted_inputs,
                                 [(full_name, _patch_position_hook(clean_act, pos))])
        out[i] = float(metric_fn(patched))
    return out


# ---------- thin wrappers for common hook points ----------

def patch_mlp_out(model, corrupted_inputs, clean_cache, metric_fn, positions=None):
    """:func:`act_patch` fixed to the block's MLP output (``blocks.0.hook_mlp_out``)."""
    return act_patch(model, corrupted_inputs, clean_cache, "mlp_out", metric_fn, positions)


def patch_resid_pre(model, corrupted_inputs, clean_cache, metric_fn, positions=None):
    """:func:`act_patch` fixed to the block's pre-attention residual stream
    (``blocks.0.hook_resid_pre``)."""
    return act_patch(model, corrupted_inputs, clean_cache, "resid_pre", metric_fn, positions)


def patch_attn_out(model, corrupted_inputs, clean_cache, metric_fn, positions=None):
    """:func:`act_patch` fixed to the block's attention output
    (``blocks.0.hook_attn_out``)."""
    return act_patch(model, corrupted_inputs, clean_cache, "attn_out", metric_fn, positions)
