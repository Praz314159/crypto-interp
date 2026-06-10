"""Config for experiment 004: modular multiplication mod 127 (Sylow-2 test).

p=127 is the "minimal Sylow-2" prime: p-1 = 126 = 2 * 3^2 * 7, so the Sylow-2
part of the character group is Z/2 (vs Z/16 at p=113). The harmonic-helper
theory says the doubling chain chi_k -> chi_2k -> ... is a Sylow-2 ladder, so at
p=127 helper chains should be at most ONE step deep (only for even-order
characters) -- a sharp, falsifiable contrast with the up-to-4-step chains at
p=113. Architecture mirrors the p=113 helper analysis (d_model=24) so K and
helper pairs are directly comparable.

run_dmlp_seed_sweep.py overrides d_mlp and seed per child run:
    python experiments/003_dmodel_sweep_p113/scripts/run_dmlp_seed_sweep.py \
        --experiment 004_p127 --d-mlps 32,128 --seeds 1,2,3 --epochs 40000
Analyze any resulting run with the canonical library analyses, e.g.:
    python -m crypto_interp.analysis.delta_spectrum --run-dir experiments/004_p127/runs/<run>
    python -m crypto_interp.analysis.bifurcation --data-dir experiments/004_p127/data  # if collected
"""

from crypto_interp.training import ExperimentConfig


CONFIG = ExperimentConfig(
    task="mul",
    p=127,
    frac_train=0.3,
    seed=0,

    # Match the p=113 helper-analysis architecture for direct comparability.
    d_model=24,
    d_mlp=512,          # baseline; the sweep overrides this (e.g. 32, 128)
    num_heads=4, d_head=32, n_ctx=3, num_layers=1,

    lr=1e-3, weight_decay=1.0, beta1=0.9, beta2=0.98, warmup_steps=10,
    num_epochs=40_000,

    # Capture the early basis-commitment dynamics natively (one pass).
    fine_grained_until=1_000,

    save_every=500, log_every=500, metrics_every=50,
    device="auto",

    notes=(
        "Sylow-2 test at p=127 (p-1 = 2*3^2*7, Sylow-2 = Z/2). Predicts shallow "
        "(<=1 step) harmonic-helper chains vs up-to-4 at p=113. Train a few seeds "
        "at tight and comfortable d_mlp, then run essential_characters + "
        "find_primary_helper_pairs and compare helper-chain depth to the 2-adic "
        "valuation of p-1."
    ),
)
