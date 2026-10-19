"""The full (p²) and units ((p-1)²) grids must agree on ordering."""

import pytest
import torch

from crypto_interp.data import TaskSpec
from crypto_interp.interp import (
    cache_all,
    compute_logits_grid,
    full_grid_inputs,
    to_units,
    units_grid_inputs,
)


def test_units_grid_is_the_full_grid_without_zero():
    ds = TaskSpec("mul", 11).build()
    full, units = full_grid_inputs(ds), units_grid_inputs(ds)
    assert full.shape == (121, 3) and units.shape == (100, 3)
    assert torch.equal(full, ds.inputs)  # the dataset's own order
    assert torch.equal(to_units(full, 11).reshape(-1, 3), units)
    assert units[:, :2].min() == 1


def test_to_units_rejects_other_shapes():
    with pytest.raises(ValueError, match="full-grid"):
        to_units(torch.zeros(100, 3), 11)


def test_cache_units_view_matches_logits_grid(loaded_run):
    model, ds, _, _ = loaded_run
    cache = cache_all(model, ds)
    from_cache = cache.grid("logits", units=True)[:, :, -1, :].double()
    direct = compute_logits_grid(model, ds)
    assert from_cache.shape == direct.shape == (ds.p - 1, ds.p - 1, ds.vocab_size)
    assert torch.allclose(from_cache, direct, atol=1e-5)
