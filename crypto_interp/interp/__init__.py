"""Mechanistic interpretability primitives for grokked modular-arithmetic models.

Reusable building blocks only — experiment-specific analysis scripts live in
``experiments/<id>/scripts/`` and import from here.
"""

from .ablate import (
    ablate_character,
    ablate_embedding,
    essential_characters,
    evaluate_loss,
    project_back,
)
from .activations import cache_all, reshape_pp, run_with_cache, summary
from .bases import (
    CharIndex,
    additive_fourier_basis,
    char_index,
    discrete_log_table,
    fold_frequency,
    multiplicative_fourier_basis,
    primitive_root,
    project_1d,
    project_2d,
)
from .cache import ActivationCache
from .dynamics import bifurcation_step, cliff_step, commit_step, grokking_status
from .grids import (
    as_full_grid,
    compute_activation_grid,
    compute_logits_grid,
    full_grid_inputs,
    to_units,
    units_grid_inputs,
)
from .harmonic import delta_k, delta_k_spectrum, find_primary_helper_pairs, helper_multiplier
from .hooks import hooks, patch_hook, project_hook, run_with_hooks, zero_hook
from .interventions import ablate_char_w, freeze_param, weight_patch
from .load import latest_checkpoint, load_run
from .metrics import (
    EmbeddingEnergyTracker,
    char_energy,
    char_energy_batch,
    correlate,
    find_cliff,
    order_energy,
    order_of,
    per_frequency_energy_from_embedding,
    topk_recall,
)
from .neurons import (
    cluster_signal,
    compute_per_neuron_frequency_energy,
    matched_bigrams,
    per_neuron_dominant_char,
    reference_cos_signal,
)
from .patching import act_patch, patch_attn_out, patch_mlp_out, patch_resid_pre
from .reductions import fourier_spectrum_1d, reduce_to_ab, reduce_to_diff
from .session import Session

__all__ = [
    # load
    "load_run",
    "latest_checkpoint",
    # activations + cache + hooks
    "cache_all",
    "reshape_pp",
    "summary",
    "run_with_cache",
    "ActivationCache",
    "run_with_hooks",
    "hooks",
    "zero_hook",
    "patch_hook",
    "project_hook",
    "weight_patch",
    "ablate_char_w",
    "freeze_param",
    "act_patch",
    "patch_mlp_out",
    "patch_resid_pre",
    "patch_attn_out",
    "Session",
    # bases
    "additive_fourier_basis",
    "multiplicative_fourier_basis",
    "primitive_root",
    "discrete_log_table",
    "project_1d",
    "project_2d",
    "CharIndex",
    "char_index",
    "fold_frequency",
    # ablate
    "project_back",
    "evaluate_loss",
    "ablate_embedding",
    "ablate_character",
    "essential_characters",
    # neurons
    "matched_bigrams",
    "compute_per_neuron_frequency_energy",
    "per_neuron_dominant_char",
    "cluster_signal",
    "reference_cos_signal",
    # metrics
    "per_frequency_energy_from_embedding",
    "EmbeddingEnergyTracker",
    "char_energy",
    "char_energy_batch",
    "order_energy",
    "order_of",
    "correlate",
    "find_cliff",
    "topk_recall",
    # grids
    "as_full_grid",
    "full_grid_inputs",
    "to_units",
    "units_grid_inputs",
    "compute_logits_grid",
    "compute_activation_grid",
    # reductions
    "reduce_to_ab",
    "reduce_to_diff",
    "fourier_spectrum_1d",
    # dynamics
    "cliff_step",
    "bifurcation_step",
    "commit_step",
    "grokking_status",
    # harmonic
    "delta_k",
    "delta_k_spectrum",
    "find_primary_helper_pairs",
    "helper_multiplier",
]
