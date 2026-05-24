from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
RUN_DIR = REPO / "experiments/003_dmodel_sweep_p113/runs/dmodel_24_dmlp_32_seed1"
HAS_RUN = RUN_DIR.exists() and any(RUN_DIR.glob("checkpoint_*.pt"))


@pytest.fixture(scope="session")
def loaded_run():
    """The reference grokked model (p=113, d_model=24, d_mlp=32, seed 1).
    Skips if the (gitignored) run dir is absent."""
    if not HAS_RUN:
        pytest.skip("reference run dmodel_24_dmlp_32_seed1 not present")
    from crypto_interp.interp import char_index, load_run

    ck = sorted(RUN_DIR.glob("checkpoint_*.pt"))[-1]
    model, ds, _ = load_run(ck)
    model.eval()
    basis, ci = char_index(ds.p)
    return model, ds, basis, ci
