"""Measures: named, versioned, cached quantities computed from checkpoints.

::

    from crypto_interp import measures, runs

    pop = runs.query(experiment="003", d_mlp=20, weight_decay=2.0)
    df = measures.kernel_noise.over(pop)                            # latest checkpoints
    traj = measures.kernel_noise.over(pop[0], checkpoints="all")    # one trajectory
    result = measures.kernel_noise.at(pop[0].latest())              # with array fields

See :mod:`crypto_interp.measures.core` for the framework.
"""

from .core import REGISTRY, Measure, measure
from .errors import ErrorProfile, errors
from .kernel import REGIMES, KernelNoise, kernel_noise

__all__ = [
    "REGIMES",
    "REGISTRY",
    "ErrorProfile",
    "KernelNoise",
    "Measure",
    "errors",
    "kernel_noise",
    "measure",
]
