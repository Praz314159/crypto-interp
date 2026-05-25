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
    CharIndex,
    char_index,
    fold_frequency,
)
from .ablate import (
    project_back,
    evaluate_loss,
    ablate_embedding,
    ablate_character,
    essential_characters,
)
from .neurons import matched_bigrams, compute_per_neuron_frequency_energy
from .metrics import (
    per_frequency_energy_from_embedding,
    EmbeddingEnergyTracker,
    char_energy,
    char_energy_batch,
    order_energy,
    order_of,
    correlate,
    find_cliff,
    topk_recall,
)
from .grids import ab_grid_inputs, compute_logits_grid, compute_activation_grid
from .reductions import reduce_to_ab, reduce_to_diff, fourier_spectrum_1d
from .dynamics import cliff_step, bifurcation_step, commit_step, grokking_status
from .harmonic import delta_k, delta_k_spectrum, find_primary_helper_pairs, helper_multiplier

__all__ = [
    # load
    "load_run", "latest_checkpoint",
    # activations
    "cache_all", "reshape_pp", "summary",
    # bases
    "additive_fourier_basis", "multiplicative_fourier_basis", "primitive_root",
    "discrete_log_table", "project_1d", "project_2d", "CharIndex", "char_index",
    "fold_frequency",
    # ablate
    "project_back", "evaluate_loss", "ablate_embedding", "ablate_character",
    "essential_characters",
    # neurons
    "matched_bigrams", "compute_per_neuron_frequency_energy",
    # metrics
    "per_frequency_energy_from_embedding", "EmbeddingEnergyTracker",
    "char_energy", "char_energy_batch", "order_energy", "order_of",
    "correlate", "find_cliff", "topk_recall",
    # grids
    "ab_grid_inputs", "compute_logits_grid", "compute_activation_grid",
    # reductions
    "reduce_to_ab", "reduce_to_diff", "fourier_spectrum_1d",
    # dynamics
    "cliff_step", "bifurcation_step", "commit_step", "grokking_status",
    # harmonic
    "delta_k", "delta_k_spectrum", "find_primary_helper_pairs", "helper_multiplier",
]
