"""Figures render from measure-shaped tables (smoke tests on synthetic data)."""

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from crypto_interp.figures import kernel as kfig  # noqa: E402

REGIMES = ["grokked", "algorithmic-but-noisy", "CRT-fail", "intermediate"]


@pytest.fixture
def traj():
    rows = []
    for p in (113, 127):
        for seed, reg in enumerate(REGIMES):
            for epoch in range(0, 2000, 500):
                decay = 10.0 ** (-epoch / 400)
                rows.append(
                    dict(
                        run=f"e{p}/s{seed}",
                        p=p,
                        seed=seed,
                        epoch=epoch,
                        regime=reg,
                        L_emp=4 * decay + 1e-6,
                        L_sym=3 * decay + 1e-8,
                        L_K=3 * decay,
                        crt_floor=float(np.log(2)) if reg == "CRT-fail" else 0.0,
                        K=(2, 4) if reg == "CRT-fail" else (3, 10, 51),
                    )
                )
    return pd.DataFrame(rows)


def test_all_kernel_figures_render(traj):
    finals = traj[traj["epoch"] == traj["epoch"].max()]
    one = traj[traj["run"] == "e113/s0"]
    for fig in (
        kfig.regimes(finals),
        kfig.trajectory(one),
        kfig.trajectory_grid(traj, "t"),
        kfig.trajectory_overlay(traj),
    ):
        assert fig.axes
        matplotlib.pyplot.close(fig)


def test_error_figures_render(S):
    from crypto_interp.figures import errors as efig
    from crypto_interp.measures import errors

    e = errors(S)
    for fig in (efig.profile(e, "ref"), efig.population({"ref": e}, {"ref": "grokked"})):
        assert fig.axes
        matplotlib.pyplot.close(fig)


def test_clock_figures_render():
    from crypto_interp.figures import clock as cfig

    tokens = pd.DataFrame(
        [
            {"regime": REGIMES[i % 2], "err_a": 0.02 * (i % 3), "err_b": 0.0, "in_dev": 0.01 * i}
            for i in range(40)
        ]
    )
    fig = cfig.residual_vs_error(tokens)
    assert fig.axes
    matplotlib.pyplot.close(fig)
