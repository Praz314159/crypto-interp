"""Config for experiment 005: modular multiplication mod 181 (depth-2 Sylow test).

p=181: p-1 = 180 = 2^2 * 3^2 * 5, so the Sylow-2 part of the character group is
Z/4 (v2=2). This sits between p=127 (v2=1, depth-1 helper chains) and p=113
(v2=4, depth-4 chains). The harmonic-helper theory predicts p=181 should permit
**depth-2** order-descending doubling chains: for a primary whose order is
divisible by 4, chi_k -> chi_2k -> chi_4k with the order halving at each step
(e.g. 180 -> 90 -> 45). Finding a helper-of-a-helper (chi_4k helps chi_2k helps
chi_k) at p=181 -- but not at p=127 -- is the key falsifiable signal.

Architecture mirrors p=113/p=127 (d_model=24) for comparability. The detector is
now fold-aware (n=180 folds frequencies >90), so doublings whose raw index 2k>90
are still labeled correctly.

Run a clean, correctly-tagged sweep with the driver:
    python experiments/003_dmodel_sweep_p113/scripts/run_dmlp_seed_sweep.py \
        --experiment 005_p181 --d-mlps 32 --seeds 1,2,3 --epochs 50000
Analyze:
    python -m crypto_interp.analysis.delta_spectrum --run-dir experiments/005_p181/runs/<run>
"""

from crypto_interp.training import ExperimentConfig


CONFIG = ExperimentConfig(
    task="mul",
    p=181,
    frac_train=0.3,
    seed=0,

    d_model=24,
    d_mlp=512,          # baseline; the sweep overrides this (e.g. 32)
    num_heads=4, d_head=32, n_ctx=3, num_layers=1,

    lr=1e-3, weight_decay=1.0, beta1=0.9, beta2=0.98, warmup_steps=10,
    num_epochs=50_000,

    # Capture early commitment dynamics in one pass.
    fine_grained_until=1_000,

    # Auto-stop ~1000 epochs after a clean grok (late cliffs are common near the
    # floor); a partial grok stuck above 1e-6 will run the full budget.
    early_stop_loss=1e-6, early_stop_patience=1_000,

    save_every=500, log_every=500, metrics_every=50,
    device="auto",

    notes=(
        "Depth-2 Sylow-2 test at p=181 (p-1 = 2^2*3^2*5, Sylow-2 = Z/4, v2=2). "
        "Predicts order-descending doubling chains up to depth 2 (a helper-of-a-"
        "helper), between p=127 (depth 1) and p=113 (depth 4). Train a few seeds "
        "at d_mlp=32, then run essential_characters + find_primary_helper_pairs "
        "(fold-aware) and check max chain depth tracks v2(p-1)."
    ),
)
