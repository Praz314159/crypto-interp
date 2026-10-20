"""The full and group grids must agree on ordering, for both kinds of group."""

import pytest
import torch

from crypto_interp.data import TaskSpec
from crypto_interp.interp import (
    cache_all,
    compute_logits_grid,
    full_grid_inputs,
    group_grid_inputs,
    to_group,
)


def test_mul_group_grid_is_the_full_grid_without_zero():
    ds = TaskSpec("mul", 11).build()
    full, group = full_grid_inputs(ds), group_grid_inputs(ds)
    assert full.shape == (121, 3) and group.shape == (100, 3)
    assert torch.equal(full, ds.inputs)  # the dataset's own order
    assert torch.equal(to_group(full, ds.structure).reshape(-1, 3), group)
    assert group[:, :2].min() == 1


def test_add_group_grid_is_the_full_grid():
    ds = TaskSpec("add", 12).build()
    assert torch.equal(group_grid_inputs(ds), full_grid_inputs(ds))


def test_to_group_rejects_other_shapes():
    with pytest.raises(ValueError, match="full-grid"):
        to_group(torch.zeros(100, 3), TaskSpec("mul", 11).build().structure)


def test_cache_group_view_matches_logits_grid(loaded_run):
    model, ds, _, _ = loaded_run
    cache = cache_all(model, ds)
    from_cache = cache.grid("logits", group=True)[:, :, -1, :].double()
    direct = compute_logits_grid(model, ds)
    assert from_cache.shape == direct.shape == (ds.p - 1, ds.p - 1, ds.vocab_size)
    assert torch.allclose(from_cache, direct, atol=1e-5)
