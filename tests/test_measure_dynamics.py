"""Run measures: grok_status and dynamics_markers on small training runs."""

import pytest

from crypto_interp.measures import dynamics_markers, grok_status
from crypto_interp.runs import Run
from crypto_interp.training import ExperimentConfig, train


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    from crypto_interp.data import TaskSpec
    from crypto_interp.interp.metrics import EmbeddingEnergyCallback

    path = tmp_path_factory.mktemp("exp") / "runs" / "r"
    cfg = ExperimentConfig(
        task="mul",
        p=11,
        d_model=16,
        d_mlp=32,
        num_heads=2,
        d_head=8,
        num_epochs=300,
        save_every=100,
        log_every=1000,
        lr=3e-3,
        fine_grained_until=20,
    )
    train(cfg, path, callbacks=[EmbeddingEnergyCallback.for_task(TaskSpec("mul", 11), every=10)])
    return Run.open(path)


def test_grok_status_reads_the_loss_history(run):
    g = grok_status(run)
    hist = run.load().train_losses
    assert g.status in {"grokked", "memorized-only", "no-fit"}
    assert g.final_train_loss == pytest.approx(hist[-1])
    if g.memorized_epoch is not None:
        assert hist[g.memorized_epoch] < 0.1 <= hist[g.memorized_epoch - 1]
    table = grok_status.over([run])
    assert list(table["status"]) == [g.status]


def test_dynamics_markers_read_metrics_and_fine_grained(run):
    # A high threshold keeps K a proper subset of the 5 characters at p = 11.
    d = dynamics_markers(run, threshold=0.5)
    assert d.K and len(d.epochs) == len(d.ratio) == 30
    assert d.ratio_at is not None
    assert d.bifurcation_epoch is None or d.bifurcation_epoch in set(d.epochs.tolist())
    assert d.bifurcation_epoch_fine is None or d.bifurcation_epoch_fine < 20


def test_dynamics_markers_when_K_is_everything(run):
    d = dynamics_markers(run, threshold=0.0)
    assert len(d.K) == 5 and d.bifurcation_epoch is None and d.ratio_at is None
