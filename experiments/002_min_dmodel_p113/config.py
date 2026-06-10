"""Config for experiment 002: minimum d_model for grokking mul mod 113.

Identical to experiment 001 except d_model is shrunk from 128 → 12 and the
training budget is doubled to allow for slower grokking onset. The
prediction (see README.md) is that d_model=12 sits just above the practical
floor of ~10, so it should grok — possibly with a smaller |K| or smaller
PosEmbed than the d_model=128 baseline.
"""

from crypto_interp.training import ExperimentConfig


CONFIG = ExperimentConfig(
    task="mul",
    p=113,
    frac_train=0.3,
    seed=0,

    # *** The variable being changed in this experiment ***
    d_model=12,

    # Everything else mirrors 001 baseline
    d_mlp=512, num_heads=4, d_head=32, n_ctx=3, num_layers=1,
    lr=1e-3, weight_decay=1.0, beta1=0.9, beta2=0.98, warmup_steps=10,

    # Longer training: shrunk models often grok later (see Nanda et al.
    # appendix; we'll also re-check this empirically here).
    num_epochs=20_000,

    save_every=500, log_every=100, metrics_every=50,
    device="auto",

    notes=(
        "Minimum-d_model probe. Prediction: groks within 20k epochs. "
        "Algorithmic floor is 2|K|=8 (4 frequencies × cos+sin); add 2 dims "
        "of rank-2 PosEmbed → 10. d_model=12 has 2 dims of slack on top "
        "of that 10. Outcome lands the model in one of five regimes "
        "documented in README.md."
    ),
)
