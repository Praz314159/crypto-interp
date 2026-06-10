"""Config for experiment 001: modular multiplication mod p=113.

Mirrors Nanda's modular-addition hyperparameters exactly so we can apply the
same Fourier / ablation methodology and read off the difference: additive vs.
multiplicative character basis.

Scripts read ``CONFIG`` from this module by convention.
"""

from crypto_interp.training import ExperimentConfig


CONFIG = ExperimentConfig(
    task="mul",
    p=113,
    frac_train=0.3,
    seed=0,

    # Architecture (Nanda exact)
    d_model=128, d_mlp=512, num_heads=4, d_head=32, n_ctx=3, num_layers=1,

    # Optimization (Nanda exact)
    lr=1e-3, weight_decay=1.0, beta1=0.9, beta2=0.98, warmup_steps=10,
    num_epochs=8_000,

    # Logging
    save_every=500, log_every=100, metrics_every=50,

    # CPU forced because MPS doesn't support float64 (Nanda's appendix says
    # float64 cross-entropy is essential for clean grokking dynamics).
    device="auto",

    notes=(
        "Top-priority MVP. Tests whether the DFT-trick generalizes from "
        "modular addition (Nanda's setup) to modular multiplication. "
        "Prediction: same circuit, but sparsity lives in the MULTIPLICATIVE "
        "character basis on (Z/p)*, not the additive basis on Z/p."
    ),
)
