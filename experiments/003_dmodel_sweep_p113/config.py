"""Config for experiment 003: d_model sweep for mul mod 113.

Baseline values mirror experiment 001. Runs override d_model (and d_mlp,
weight_decay, seed) per run, e.g.:

    crypto-interp sweep --experiment 003_dmodel_sweep_p113 --seeds 1..21 \
        --set d_model=24 --set d_mlp=20 --set weight_decay=2

Keeping the variants under one experiment id gives them one runs/ tree;
the run index selects among them by config.
"""

from crypto_interp.training import ExperimentConfig

CONFIG = ExperimentConfig(
    task="mul",
    p=113,
    frac_train=0.3,
    seed=0,
    # Baseline placeholder — the sweep overrides this per child.
    # Keeping it at the 001-baseline value means the un-overridden config
    # is itself a meaningful "control" run.
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
    # Long enough to either grok or land cleanly on a plateau. 20k matched
    # the d_model=12 run; we use the same budget here for direct comparability.
    num_epochs=20_000,
    save_every=500,
    log_every=500,
    metrics_every=50,
    device="auto",
    notes=(
        "d_model sweep. Asks two distinct questions: (a) algorithmic floor "
        "d_alg = min d_model where the model finds the mul-basis algorithm "
        "(even partially); (b) grokking floor d_grok = min d_model where "
        "the classic two-phase memorize→cliff dynamic occurs. From "
        "experiment 002, d_alg ≤ 12 < d_grok. This sweep brackets d_grok "
        "from above by trying d_model ∈ {16, 24, 32, 64}."
    ),
)
