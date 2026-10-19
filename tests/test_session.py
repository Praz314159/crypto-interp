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
