"""Mechanistic interpretability primitives for grokked modular-arithmetic models.

Reusable building blocks only — experiment-specific analysis scripts live in
``experiments/<id>/scripts/`` and import from here.
"""

from .load import load_run, latest_checkpoint
from .activations import cache_all, reshape_pp, summary
from .bases import (
    additive_fourier_basis,
    multiplicative_fourier_basis,
    primitive_root,
    discrete_log_table,
    project_1d,
    project_2d,
)
from .ablate import project_back, evaluate_loss, ablate_embedding
from .neurons import matched_bigrams, compute_per_neuron_frequency_energy
from .metrics import per_frequency_energy_from_embedding, EmbeddingEnergyTracker

__all__ = [
    "load_run",
    "latest_checkpoint",
    "cache_all",
    "reshape_pp",
    "summary",
    "additive_fourier_basis",
    "multiplicative_fourier_basis",
    "primitive_root",
    "discrete_log_table",
    "project_1d",
    "project_2d",
    "project_back",
    "evaluate_loss",
    "ablate_embedding",
    "matched_bigrams",
    "compute_per_neuron_frequency_energy",
    "per_frequency_energy_from_embedding",
    "EmbeddingEnergyTracker",
]
