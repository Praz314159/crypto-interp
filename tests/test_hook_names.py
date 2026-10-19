"""Hook-name resolution in 1-block and multi-block models."""
import pytest
import torch

from crypto_interp.interp import patch_mlp_out, run_with_cache, run_with_hooks, zero_hook
from crypto_interp.models import Transformer, TransformerConfig


@pytest.fixture(scope="module")
def two_layer():
    torch.manual_seed(0)
    model = Transformer(TransformerConfig(d_vocab=8, d_model=8, d_mlp=16, num_heads=2,
                                          d_head=4, n_ctx=3, num_layers=2))
    inputs = torch.tensor([[a, b, 7] for a in range(7) for b in range(7)])
    return model, inputs


def test_layer_qualified_names(two_layer):
    model, inputs = two_layer
    _, cache = run_with_cache(model, inputs)
    for layer in (0, 1):
        assert torch.equal(cache["mlp_post", layer], cache[f"blocks.{layer}.mlp.hook_post"])
    with pytest.raises(KeyError, match="ambiguous"):
        cache["mlp_post"]
    with pytest.raises(KeyError, match="layer 2"):
        cache["mlp_post", 2]
    with pytest.raises(ValueError, match="1-block"):
        cache.decompose_resid()


def test_hooks_target_one_layer(two_layer):
    model, inputs = two_layer
    clean = model(inputs)
    with pytest.raises(KeyError, match="ambiguous"):
        run_with_hooks(model, inputs, [("mlp_out", zero_hook())])
    by_layer = [run_with_hooks(model, inputs, [(("mlp_out", l), zero_hook())]) for l in (0, 1)]
    assert not torch.allclose(by_layer[0], clean)
    assert not torch.allclose(by_layer[1], clean)
    assert not torch.allclose(by_layer[0], by_layer[1])


def test_patch_with_layer(two_layer):
    """Patching a layer's clean activation into the same inputs changes nothing."""
    model, inputs = two_layer
    _, cache = run_with_cache(model, inputs)
    metric = lambda logits: float(logits.abs().sum())
    with torch.no_grad():
        base = metric(model(inputs))
    out = patch_mlp_out(model, inputs, cache, metric, layer=1)
    assert torch.allclose(out, torch.full_like(out, base))


def test_single_block_short_names(loaded_run):
    model, ds, _, _ = loaded_run
    _, cache = run_with_cache(model, ds.inputs[:5])
    assert torch.equal(cache["mlp_post"], cache["mlp_post", 0])
    assert set(cache.decompose_resid()) == {"resid_pre", "attn_out", "mlp_out"}
