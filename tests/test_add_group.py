"""Addition mod n runs through the same analyses as multiplication mod p."""

import matplotlib
import pytest

matplotlib.use("Agg")

from crypto_interp import measures  # noqa: E402
from crypto_interp.cli import main  # noqa: E402
from crypto_interp.data import TaskSpec  # noqa: E402
from crypto_interp.interp import Session  # noqa: E402
from crypto_interp.interp.metrics import EmbeddingEnergyCallback  # noqa: E402
from crypto_interp.runs import Run  # noqa: E402
from crypto_interp.training import ExperimentConfig, train  # noqa: E402


@pytest.fixture(scope="module")
def add_run(tmp_path_factory):
    """A tiny add mod 12 run (not grokked: the analyses must run, not succeed)."""
    path = tmp_path_factory.mktemp("exp") / "001_add" / "runs" / "n12"
    cfg = ExperimentConfig(
        task="add", p=12, d_model=8, d_mlp=16, num_heads=2, d_head=4,
        num_epochs=40, save_every=20, log_every=1000, metrics_every=10,
    )  # fmt: skip
    task = TaskSpec(cfg.task, cfg.p, cfg.frac_train, cfg.seed)
    train(cfg, path, callbacks=[EmbeddingEnergyCallback.for_task(task, every=10)])
    return Run.open(path)


def test_every_measure_runs_on_add(add_run):
    assert add_run.config()["n"] == 12
    S = Session.from_run(add_run)
    assert S.ci.n == 12 and S.logits_grid.shape[:2] == (12, 12)
    for name, m in measures.REGISTRY.items():
        if name.startswith("_"):
            continue
        r = m(add_run) if isinstance(m, measures.RunMeasure) else m.at(add_run.latest())
        assert r is not None, name
    kn = measures.kernel_noise.at(add_run.latest())
    assert kn.kappa_obs.shape == (12,) and kn.regime in measures.REGIMES
    assert measures.errors.at(add_run.latest()).n_total == 144
    assert measures.dynamics_markers(add_run).epochs.size > 0  # energies were recorded


def test_single_run_figures_render_on_add(add_run, tmp_path):
    run = str(add_run.path)
    k = str(measures.energy_support.at(add_run.latest()).K[0])
    for name, extra in [
        ("walkthrough", ["--a", "5", "--b", "9"]),
        ("cluster", ["--k", k]),
        ("delta", ["--k", k]),
        ("error-profile", []),
        ("char-ablation", []),
    ]:
        out = tmp_path / f"{name}.png"
        main(["figure", name, "--run", run, *extra, "--out", str(out)])
        assert out.stat().st_size > 0, name
