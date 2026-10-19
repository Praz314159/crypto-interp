"""Measures: named, versioned, cached quantities computed from checkpoints.

::

    from crypto_interp import measures, runs

    pop = runs.query(experiment="003", d_mlp=20, weight_decay=2.0)
    df = measures.kernel_noise.over(pop)                            # latest checkpoints
    traj = measures.kernel_noise.over(pop[0], checkpoints="all")    # one trajectory
    result = measures.kernel_noise.at(pop[0].latest())              # with array fields

See :mod:`crypto_interp.measures.core` for the framework.
"""

from .characters import (
    CharAblation,
    EnergySupport,
    Fourier2D,
    Helpers,
    NeuronClusters,
    char_ablation,
    energy_support,
    fourier2d,
    helpers,
    neuron_clusters,
)
from .clock import (
    ClockDefect,
    ClockRepair,
    ClockResiduals,
    clock_defect,
    clock_repair,
    clock_residuals,
    token_table,
)
from .core import REGISTRY, Measure, measure
from .errors import ErrorProfile, errors
from .kernel import REGIMES, KernelNoise, kernel_noise
from .noise import NOISE_CLASSES, NoiseModes, noise_modes

__all__ = [
    "neuron_clusters",
    "helpers",
    "fourier2d",
    "energy_support",
    "char_ablation",
    "NeuronClusters",
    "Helpers",
    "Fourier2D",
    "EnergySupport",
    "CharAblation",
    "NOISE_CLASSES",
    "REGIMES",
    "REGISTRY",
    "ClockDefect",
    "ClockRepair",
    "ClockResiduals",
    "ErrorProfile",
    "KernelNoise",
    "Measure",
    "NoiseModes",
    "clock_defect",
    "clock_repair",
    "clock_residuals",
    "errors",
    "kernel_noise",
    "measure",
    "noise_modes",
    "token_table",
]
