"""Config for experiment 003: d_model sweep for mul mod 113.

Baseline values mirror experiment 001. The d_model value is overridden
per-run by `scripts/run_sweep.py`, which iterates over a planned set and
launches `scripts/train.py --override d_model=<N> --tag dmodel_<N>` for
each.

Why a sweep config rather than separate experiments per d_model: keeping
the d_model variants under a single experiment id lets all runs share a
dataset cache and a runs/ tree, and the comparison plots have a natural
home in figures/.
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

    d_mlp=512, num_heads=4, d_head=32, n_ctx=3, num_layers=1,
    lr=1e-3, weight_decay=1.0, beta1=0.9, beta2=0.98, warmup_steps=10,

    # Long enough to either grok or land cleanly on a plateau. 20k matched
    # the d_model=12 run; we use the same budget here for direct comparability.
    num_epochs=20_000,

    save_every=500, log_every=500, metrics_every=50,
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
