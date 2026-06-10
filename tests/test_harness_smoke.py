"""Harness v1 smoke test — Session + cost-atom playbook recipe on the
reference run, plus an end-to-end activation-patching check.

Mirrors the playbooks (``basis_discovery.md``, ``economy_analysis.md``,
``causal_intervention.md``) so that if these tests break, the playbooks are
also out of date. Skipped automatically if the reference run isn't present.
"""
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from crypto_interp.interp import (
    Session,
    ablate_char_w,
    act_patch,
    correlate,
    patch_mlp_out,
    weight_patch,
)

REPO = Path(__file__).resolve().parents[1]
SEED1 = REPO / "experiments/003_dmodel_sweep_p113/runs/dmodel_24_dmlp_20_wd2_seed1"


@pytest.fixture(scope="module")
def S():
    if not (SEED1.exists() and any(SEED1.glob("checkpoint_*.pt"))):
        pytest.skip("reference run dmodel_24_dmlp_20_wd2_seed1 not present")
    return Session.from_run(SEED1)


def test_cost_atom_playbook(S):
    """The cost-atom recipe from playbooks/economy_analysis.md."""
    ess = S.essential(threshold=0.05)
    assert sorted(ess["K"]) == [8, 10, 33, 46, 52]

    pairs = S.helpers(ess["K"])
    triples = sorted((h, m, mult) for (h, m, mult, _e) in pairs)
    # Doubling-economy: ×2 helpers only; χ_8 = 2·χ_52, χ_46 = 2·χ_33.
    assert triples == [(8, 52, 2), (46, 33, 2)]

    sizes = S.cluster_sizes(ess["K"])
    # Helpers cost 0 dedicated neurons; primaries 6-8.
    assert sizes[8] == 0 and sizes[46] == 0
    assert sizes[10] == 8 and sizes[33] == 6 and sizes[52] == 6


def test_mechanism_verification_playbook(S):
    """Mechanism-verification: the primary χ_10 cluster's signal correlates
    > 0.94 with the algebraic reference cos(θ_10(a) + θ_10(b))."""
    sig = S.cluster_signal(10)
    ref = S.reference_signal(10)
    corr = correlate(sig, ref)
    assert corr > 0.94, f"χ_10 cluster vs reference correlation {corr:.3f} too low"


def test_causal_intervention_playbook(S):
    """Activation-patching recipe: mlp_out at position 2 (= token) fully
    recovers when χ_10 is ablated from W_E."""
    _, clean_cache = S.run_with_cache(S.ds.inputs)
    with ablate_char_w(S.model, k=10, basis=S.basis, ci=S.ci):
        per_pos = patch_mlp_out(S.model, S.ds.inputs, clean_cache, S.test_loss_metric)
    # Position 2 (the '=' token) carries the answer-position information; 0 and 1 do not.
    assert float(per_pos[2]) < 1e-3, f"pos-2 patch failed to recover: {per_pos[2]}"
    assert float(per_pos[0]) > 1.0, f"pos-0 patch unexpectedly recovered: {per_pos[0]}"


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
