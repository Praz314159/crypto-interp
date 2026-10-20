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
    for fig in (
        efig.profile(e, S.ds.structure, "ref"),
        efig.population({"ref": e}, {"ref": "grokked"}),
    ):
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


def test_noise_figures_render():
    from crypto_interp.figures import noise as nfig
    from crypto_interp.measures import NOISE_CLASSES

    rows = [
        {
            "seed": s,
            "regime": REGIMES[s],
            "L_emp": 1e-3 * (s + 1),
            "L_sym": 1e-4,
            "noise_at_target_var": 1e-3 * (s + 1),
            "total_power": 10.0,
            **{c: 10.0 / len(NOISE_CLASSES) for c in NOISE_CLASSES},
        }
        for s in range(4)
    ]
    fig = nfig.population(pd.DataFrame(rows))
    assert fig.axes
    matplotlib.pyplot.close(fig)


def test_character_figures_render(S):
    from crypto_interp.figures import characters as chfig
    from crypto_interp.measures import (
        char_ablation,
        energy_support,
        fourier2d,
        helpers,
        neuron_clusters,
    )

    h, nc = helpers(S), neuron_clusters(S)
    for fig in (
        chfig.ablation(char_ablation(S), energy_support(S), "ref"),
        chfig.delta_spectrum(h),
        chfig.delta(h, 10, S.ds.structure),
        chfig.cluster(nc, 10, S.ds.structure),
        chfig.fourier2d(fourier2d(S)),
    ):
        assert fig.axes
        matplotlib.pyplot.close(fig)


def test_population_and_theory_figures_render():
    from crypto_interp.figures import populations as pf
    from crypto_interp.figures import theory as tf
    from crypto_interp.measures.dynamics import DynamicsMarkers

    lat = pd.DataFrame(
        [
            {"p": p, "regime": r, "K": (3, 10), "helper_mult": (2,)}
            for p in (113, 127)
            for r in ("grokked", "CRT-fail")
        ]
    )
    dm = pd.DataFrame(
        [
            {"d_mlp": d, "status": s, "K": (3, 10), "cliff_epoch": 100 * d}
            for d in (16, 32)
            for s in ("grokked", "no-fit")
        ]
    )
    dyn = pd.DataFrame(
        [
            {"K_class": c, "bifurcation_epoch": 100, "cliff_epoch": 900, "ratio_at": 5.0}
            for c in ("primitive", "mixed")
        ]
    )
    marker = DynamicsMarkers(
        (3,), "primitive", 900, 100, 200, None, 5.0, np.arange(0, 1000, 100), np.linspace(1, 50, 10)
    )
    fvu = {"widths": [1, 2], "primary": [0.5, 0.2], "helper": [0.3, 0.05]}
    for fig in (
        pf.lattice(lat),
        pf.by_d_mlp(dm),
        pf.cliff_vs_bifurcation(dyn),
        pf.bifurcation_overlay({1: marker}),
        tf.helper_fvu(fvu, 10),
    ):
        assert fig.axes
        matplotlib.pyplot.close(fig)
