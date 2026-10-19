from crypto_interp.models import Transformer, TransformerConfig


def test_config_from_dict_ignores_other_keys_and_defaults_missing():
    cfg = TransformerConfig.from_dict({"d_vocab": 8, "d_model": 16, "lr": 1e-3, "task": "mul"})
    assert cfg == TransformerConfig(d_vocab=8, d_model=16)


def test_from_config_accepts_dict_or_config():
    d = {"d_vocab": 8, "d_model": 16, "d_mlp": 32, "num_heads": 2, "d_head": 8, "num_layers": 2}
    a = Transformer.from_config(d)
    b = Transformer.from_config(TransformerConfig.from_dict(d))
    assert a.cfg == b.cfg and len(a.blocks) == 2
