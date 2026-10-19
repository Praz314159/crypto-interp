from pathlib import Path

import pytest

FIXTURES = Path(__file__).resolve().parent / "fixtures"
# The reference grokked model (p=113, d_model=24, d_mlp=32, seed 1): the run
# documented in note 06. Committed as a weights-only checkpoint; its dataset is
# rebuilt deterministically from the config.
REFERENCE_CKPT = FIXTURES / "mul_p113_d24_m32_seed1" / "checkpoint_009999.pt"


@pytest.fixture(scope="session")
def loaded_run(tmp_path_factory):
    """``(model, ds, basis, ci)`` for the reference model."""
    from crypto_interp.interp import char_index, load_run

    model, ds, _ = load_run(REFERENCE_CKPT, datasets_dir=tmp_path_factory.mktemp("datasets"))
    model.eval()
    basis, ci = char_index(ds.p)
    return model, ds, basis, ci


@pytest.fixture(scope="session")
def S(loaded_run):
    """A :class:`Session` on the reference model."""
    from crypto_interp.interp import Session

    model, ds, basis, ci = loaded_run
    return Session(model, ds, basis=basis, ci=ci)
