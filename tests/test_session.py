import pytest

from crypto_interp import data
from crypto_interp.interp import Session, char_index


def test_session_refuses_tasks_without_a_default_basis(loaded_run):
    model, _, _, _ = loaded_run
    ds = data.build(task="add", p=113, frac_train=0.3, seed=0)
    with pytest.raises(ValueError, match="no default basis for task 'add'"):
        Session(model, ds)
    basis, ci = char_index(ds.p)
    assert Session(model, ds, basis=basis, ci=ci).task == "add"


def test_session_from_run_accepts_a_run_or_a_directory(tmp_path):
    from crypto_interp.runs import Run
    from crypto_interp.training import ExperimentConfig, train

    path = tmp_path / "exp" / "runs" / "r"
    cfg = ExperimentConfig(
        task="mul",
        p=7,
        d_model=8,
        d_mlp=16,
        num_heads=2,
        d_head=4,
        num_epochs=3,
        save_every=5,
        log_every=1000,
    )
    train(cfg, path, tmp_path / "exp" / "datasets")
    a, b = Session.from_run(Run.open(path)), Session.from_run(path)
    assert a.ds.p == b.ds.p == 7
