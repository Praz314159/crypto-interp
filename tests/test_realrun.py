"""Real-run regression tests against the documented harmonic-helper findings.
Skipped automatically (via the loaded_run fixture) if the run dir is absent."""
import torch

from crypto_interp.interp import (
    ablate_character,
    compute_logits_grid,
    essential_characters,
    find_primary_helper_pairs,
)


def test_essential_characters_and_helper_pairs(loaded_run):
    model, ds, basis, ci = loaded_run
    res = essential_characters(model, ds, ci, basis)

    # the three documented primaries are essential
    for k in (3, 10, 51):
        assert res["per_char"][k]["cls"] == "essential", k

    # all five documented characters are at least load-bearing
    lb = {k for k, v in res["per_char"].items() if v["cls"] in ("load-bearing", "essential")}
    assert {3, 6, 10, 20, 51} <= lb

    K_aug = sorted(set(res["K"]) | lb)
    pairs = {(h, prim) for h, prim, _ in find_primary_helper_pairs(model, ds, ci, basis, K_aug)}
    # the Sylow-2 doubling helpers documented in note 06
    assert (6, 3) in pairs
    assert (20, 10) in pairs


def test_compute_logits_grid_restores_W_E(loaded_run):
    model, ds, basis, ci = loaded_run
    p = ds.p
    before = model.embed.W_E.detach().clone()
    W_ab = ablate_character(model.embed.W_E.detach()[:, :p], basis, ci, 10)
    W_over = model.embed.W_E.detach().clone()
    W_over[:, :p] = W_ab.to(W_over.dtype)
    _ = compute_logits_grid(model, ds, W_E_override=W_over)
    assert torch.allclose(model.embed.W_E.detach(), before)
