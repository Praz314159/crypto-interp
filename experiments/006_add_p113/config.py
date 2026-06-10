"""Experiment 006 — modular ADDITION at p=113 (the Nanda baseline as a *task*).

Existence-proof config for the second task in the harness. Same architecture
as the mod-mul baseline (d_model=128, d_mlp=512); only the data dispatch
changes via task="add".

This config exists primarily to demonstrate that adding a task to the harness
is a small extension (one ``data/add.py`` + one ``_TASKS`` registry entry).
The actual scientific value of training mod-add at p=113 is the *contrast*:

  - mod-add on Z/p uses *additive* Fourier characters of (Z/p, +).
  - mod-mul on (Z/p)* uses *multiplicative* / Dirichlet characters via the
    discrete log.

The downstream analyses (essential characters, helper detection, cost atom)
currently default to the multiplicative basis — they will dispatch on
``task`` once additive-basis support lands; see ``Session.basis`` in
``crypto_interp/interp/session.py``.
"""

from crypto_interp.training import ExperimentConfig


CONFIG = ExperimentConfig(
    task="add",
    p=113,
    frac_train=0.3,
    seed=0,

    d_model=128, d_mlp=512, num_heads=4, d_head=32, n_ctx=3, num_layers=1,
    lr=1e-3, weight_decay=1.0, beta1=0.9, beta2=0.98, warmup_steps=10,

    num_epochs=20_000,

    save_every=500, log_every=500, metrics_every=50,
    device="auto",

    notes=(
        "Modular addition baseline at p=113. The Nanda task, run inside our "
        "harness so we can use the same `Session` verbs to compare against "
        "modular multiplication. Analysis-side basis dispatch (additive vs "
        "multiplicative) is deferred to v2 of the harness after the "
        "lattice-variation experimental program runs."
    ),
)
