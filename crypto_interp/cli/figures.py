"""Figure recipes for ``crypto-interp figure NAME``.

A recipe computes the measures a figure needs (through the results store),
joins their tables, and calls a function in :mod:`crypto_interp.figures`.
Single-run recipes require the selection to be exactly one run.
"""

from __future__ import annotations

from collections.abc import Callable

from .. import measures as M
from ..figures import characters, clock, errors, kernel, noise, populations, theory

RECIPES: dict[str, tuple[str, Callable]] = {}


def recipe(name: str, help: str):
    def wrap(fn):
        RECIPES[name] = (help, fn)
        return fn

    return wrap


def _one(pop):
    if len(pop) != 1:
        raise SystemExit(f"this figure is for one run; the selection has {len(pop)}")
    return pop[0]


def _with_regime(df, pop):
    return df.merge(M.kernel_noise.over(pop)[["run", "regime"]], on="run")


# ---- populations --------------------------------------------------------


@recipe("regimes", "L_emp vs L_sym per run, and CRT failures against their floor")
def _regimes(pop, args):
    return kernel.regimes(M.kernel_noise.over(pop))


@recipe("trajectory-grid", "kernel/noise losses over training, a panel per run")
def _trajectory_grid(pop, args):
    return kernel.trajectory_grid(M.kernel_noise.over(pop, checkpoints="all"), args.title or "")


@recipe("trajectory-overlay", "per prime: median kernel/noise trajectories and exemplars")
def _trajectory_overlay(pop, args):
    return kernel.trajectory_overlay(M.kernel_noise.over(pop, checkpoints="all"))


@recipe("lattice", "per prime: grokking rate, |K|, helper multipliers")
def _lattice(pop, args):
    df = M.kernel_noise.over(pop).merge(M.helpers.over(pop)[["run", "helper_mult"]], on="run")
    return populations.lattice(df)


@recipe("by-d-mlp", "|K| and grokking cliff against MLP width")
def _by_d_mlp(pop, args):
    df = M.grok_status.over(pop).merge(M.energy_support.over(pop)[["run", "K"]], on="run")
    return populations.by_d_mlp(df)


@recipe("cliff-vs-bifurcation", "when K outgrows the rest vs when the run grokks")
def _cliff_vs_bifurcation(pop, args):
    return populations.cliff_vs_bifurcation(M.dynamics_markers.over(pop))


@recipe("bifurcation-overlay", "K / non-K energy ratio over training, a line per run")
def _bifurcation_overlay(pop, args):
    return populations.bifurcation_overlay({r.name: M.dynamics_markers(r) for r in pop})


@recipe("noise-population", "noise mode composition, target noise vs CE gap")
def _noise_population(pop, args):
    return noise.population(_with_regime(M.noise_modes.over(pop), pop))


@recipe("residual-vs-error", "token clock deviation against token error rate")
def _residual_vs_error(pop, args):
    return clock.residual_vs_error(M.token_table(pop))


@recipe("error-population", "margin distributions and error counts across runs")
def _error_population(pop, args):
    regimes = M.kernel_noise.over(pop).set_index("run")["regime"]
    return errors.population(
        {r.name: M.errors.at(r.latest()) for r in pop}, {r.name: regimes[r.id] for r in pop}
    )


# ---- one run ------------------------------------------------------------


@recipe("trajectory", "one run's kernel/noise losses and |K| over training")
def _trajectory(pop, args):
    return kernel.trajectory(M.kernel_noise.over(_one(pop), checkpoints="all"))


@recipe("error-profile", "one run's margins, error map and per-token error rates")
def _error_profile(pop, args):
    run = _one(pop)
    return errors.profile(M.errors.at(run.latest()), run.id)


@recipe("char-ablation", "one run's per-character ablation impact")
def _char_ablation(pop, args):
    run = _one(pop)
    ck = run.latest()
    return characters.ablation(M.char_ablation.at(ck), M.energy_support.at(ck), run.id)


@recipe("delta-spectrum", "one run's Δ_k spectra (primaries and helpers)")
def _delta_spectrum(pop, args):
    run = _one(pop)
    return characters.delta_spectrum(M.helpers.at(run.latest()), run.id)


@recipe("delta", "one run's Δ_k map for --k")
def _delta(pop, args):
    run = _one(pop)
    return characters.delta(M.helpers.at(run.latest()), args.k, run.task.p)


@recipe("cluster", "one run's χ_k neuron cluster (--k) against its reference")
def _cluster(pop, args):
    run = _one(pop)
    return characters.cluster(M.neuron_clusters.at(run.latest()), args.k, run.task.p)


@recipe("fourier2d", "one run's MLP output energy over character pairs")
def _fourier2d(pop, args):
    run = _one(pop)
    return characters.fourier2d(M.fourier2d.at(run.latest()), run.id)


@recipe("noise-classes", "one run's residual power by mode class")
def _noise_classes(pop, args):
    run = _one(pop)
    return noise.classes(M.noise_modes.at(run.latest()), run.id)


# ---- no runs ------------------------------------------------------------


@recipe("helper-fvu", "ReLU MLP fit of cos(k(φa+φb)) with and without χ_2k (no runs)")
def _helper_fvu(pop, args):
    from ..theory.relu_harmonics import product_fit_fvu

    k = args.k or 10
    return theory.helper_fvu(product_fit_fvu(112, k, widths=(1, 2, 3, 4, 6, 8)), k)


NO_RUNS = {"helper-fvu"}
