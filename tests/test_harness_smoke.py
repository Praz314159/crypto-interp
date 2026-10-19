"""Smoke tests for Session and the intervention verbs on the reference model.

The reference model (``conftest.REFERENCE_CKPT``) grokked with
K = {3, 10, 20, 51}: three primaries plus χ_20, a doubling helper of χ_10.
"""
import torch

from crypto_interp.interp import (
    ablate_char_w,
    correlate,
    patch_mlp_out,
    weight_patch,
)


def test_cost_atom(S):
    """Essential set, the doubling helper, and per-primary neuron budgets."""
    ess = S.essential(threshold=0.05)
    assert sorted(ess["K"]) == [3, 10, 20, 51]

    triples = sorted((h, m, mult) for (h, m, mult, _e) in S.helpers(ess["K"]))
    assert triples == [(20, 10, 2)]

    # The helper costs no dedicated neurons; the primaries share the other 32.
    assert S.cluster_sizes(ess["K"]) == {3: 11, 10: 13, 20: 0, 51: 8}


def test_mechanism_verification(S):
    """Each primary's neuron cluster reproduces cos(θ_k(a) + θ_k(b))."""
    for k, floor in [(3, 0.97), (10, 0.95)]:
        corr = correlate(S.cluster_signal(k), S.reference_signal(k))
        assert corr > floor, f"χ_{k} cluster vs reference correlation {corr:.3f} too low"
    # A helper has no cluster of its own.
    assert S.cluster_signal(20) is None


def test_causal_intervention(S):
    """Ablating χ_10 from W_E breaks the model; patching clean mlp_out back in
    at the '=' position fully recovers it, and patching at a or b does not."""
    _, clean_cache = S.run_with_cache(S.ds.inputs)
    with ablate_char_w(S.model, k=10, basis=S.basis, ci=S.ci):
        _, te, _ = S.evaluate()
        assert te > 10.0
        per_pos = patch_mlp_out(S.model, S.ds.inputs, clean_cache, S.test_loss_metric)
    assert float(per_pos[2]) < 1e-3, f"pos-2 patch failed to recover: {per_pos[2]}"
    assert float(per_pos[0]) > 10.0 and float(per_pos[1]) > 10.0


def test_weight_patch_restores(S):
    """weight_patch context manager must restore W_E exactly on exit."""
    W_E_before = S.model.embed.W_E.detach().clone()
    new_E = torch.zeros_like(S.model.embed.W_E)
    with weight_patch(S.model, "embed.W_E", new_E):
        # While inside, evaluating should give a large test loss.
        _, te, _ = S.evaluate()
        assert te > 1.0
    W_E_after = S.model.embed.W_E.detach()
    assert torch.equal(W_E_before, W_E_after), "weight_patch did not restore W_E exactly"
