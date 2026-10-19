"""mul_units: multiplication on the units only, and the analyses that run on it."""

import torch

from crypto_interp.data import TaskSpec
from crypto_interp.interp import Session, to_units, units_mask
from crypto_interp.interp.metrics import EmbeddingEnergyCallback
from crypto_interp.measures import energy_support, kernel_noise
from crypto_interp.runs import checkpoint_paths
from crypto_interp.training import ExperimentConfig, train


def test_mul_units_is_the_group_without_zero():
    ds = TaskSpec("mul_units", 11, seed=3).build()
    assert ds.inputs.shape == (100, 3) and ds.inputs[:, :2].min() == 1
    a, b = ds.inputs[:, 0], ds.inputs[:, 1]
    assert torch.equal(ds.labels, (a * b) % 11) and ds.labels.min() >= 1
    assert ds.n_train == 30 and ds.vocab_size == 12 and ds.n_answer_tokens == 11


def test_units_mask_places_examples_by_coordinates():
    full = TaskSpec("mul", 11, seed=3).build()
    assert torch.equal(units_mask(full, full.test_mask), to_units(full.test_mask, 11))

    units = TaskSpec("mul_units", 11, seed=3).build()
    m = units_mask(units, units.test_mask)
    assert m.shape == (10, 10) and int(m.sum()) == units.n_test
    for (a, b, _), test in zip(units.inputs.tolist(), units.test_mask.tolist(), strict=True):
        assert bool(m[a - 1, b - 1]) == test


def test_a_mul_units_run_trains_and_measures(tmp_path):
    cfg = ExperimentConfig(
        task="mul_units", p=11, d_model=8, d_mlp=16, num_heads=2, d_head=4,
        num_epochs=20, save_every=10, log_every=1000, metrics_every=10,
    )  # fmt: skip
    task = TaskSpec(cfg.task, cfg.p, cfg.frac_train, cfg.seed)
    train(cfg, tmp_path / "run", callbacks=[EmbeddingEnergyCallback.for_task(task, every=10)])
    assert (tmp_path / "run" / "metrics.pt").exists()

    assert len(checkpoint_paths(tmp_path / "run")) == 3
    S = Session.from_run(tmp_path / "run")
    assert S.task == "mul_units"
    r = kernel_noise(S, K=energy_support(S).K)
    assert r.kappa_obs.shape == (10,) and r.regime in (
        "grokked",
        "algorithmic-but-noisy",
        "CRT-fail",
        "intermediate",
    )
