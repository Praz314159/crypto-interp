"""Config for experiment 007: replicate the Discrete Log Clock setup at p = 113.

Nguyen (2026), "The Discrete Log Clock", trains Nanda et al.'s (2023)
one-layer model on a·b mod 113 over the units, {1..112}², and reports that
about 80% of seeds learn a sparse set of multiplicative characters. We train
the same configuration over many seeds and ask whether the others are CRT
failures (gcd(K ∪ {112}) > 1) or something else.

The main arm is task="mul_units", as in the paper. The control arm sets
task="mul" (inputs include 0), with everything else equal:

    crypto-interp sweep --experiment 007_dlc_p113 --seeds 1..30 \
        --grid task=mul_units,mul --tag-format '{task}_seed{seed}'
"""

from crypto_interp.training import ExperimentConfig

CONFIG = ExperimentConfig(
    task="mul_units",
    p=113,
    frac_train=0.3,
    seed=0,
    d_model=128,
    d_mlp=512,
    num_heads=4,
    d_head=32,
    n_ctx=3,
    num_layers=1,
    lr=1e-3,
    weight_decay=1.0,
    beta1=0.9,
    beta2=0.98,
    warmup_steps=10,
    num_epochs=40_000,
    save_every=2000,
    log_every=1000,
    metrics_every=100,
    device="auto",
    notes="Discrete Log Clock replication (Nguyen 2026; Nanda et al. 2023 hyperparameters).",
)
