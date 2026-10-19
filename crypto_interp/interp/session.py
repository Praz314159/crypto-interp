"""Session: one model, its dataset and its character basis, with shared forward passes.

A :class:`Session` is the context a measure runs in (see
:mod:`crypto_interp.measures`). It holds ``(model, ds, task, basis, ci)`` and
the primitive accessors measures share; the analyses themselves live in
measures.

Two lazy properties are built on first access and reused, so several queries
against the same forward pass pay for it once: ``cache`` (an
:class:`ActivationCache` over the full p² grid; ``cache.grid(name,
units=True)`` gives its units view) and ``logits_grid`` (logits over the (p-1)²
units grid). See :mod:`crypto_interp.interp.grids` for the two grids.

::

    from crypto_interp.interp import Session
    from crypto_interp import measures

    S = Session.from_run("experiments/003/.../seed1")
    measures.energy_support(S).K      # (3, 10, 20, 51) on the reference model
    measures.helpers(S).helpers       # (20,): χ_20 doubles χ_10
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch


class Session:
    def __init__(
        self, model, ds, task: Optional[str] = None, basis: Optional[torch.Tensor] = None, ci=None
    ):
        self.model = model
        self.ds = ds
        self.task = task or getattr(ds, "task", "mul")
        if basis is None or ci is None:
            # The default is the multiplicative-character basis on (Z/p)*, which is
            # only the right basis for ``mul``. Analysing another task in it gives
            # plausible-looking but meaningless numbers, so refuse.
            from ..data import MULTIPLICATIVE_TASKS

            if self.task not in MULTIPLICATIVE_TASKS:
                raise ValueError(
                    f"Session has no default basis for task {self.task!r} "
                    f"(only {sorted(MULTIPLICATIVE_TASKS)}); pass basis= and ci= explicitly"
                )
            from .bases import char_index

            basis, ci = char_index(ds.p)
        self.basis = basis
        self.ci = ci
        self._cache = None
        self._logits_grid = None
        self._dom = None  # cached per-neuron dominant-char tensor

    # ---------- constructors ----------

    @classmethod
    def from_run(cls, run, device: str = "cpu") -> "Session":
        """Load the latest checkpoint of ``run`` (a :class:`~crypto_interp.runs.Run` or a
        run directory); build basis/char_index from ds.p."""
        from .load import latest_checkpoint, load_run

        run_dir = getattr(run, "path", run)
        model, ds, _ = load_run(latest_checkpoint(run_dir), device=device)
        return cls(model, ds)

    # ---------- lazy properties ----------

    @property
    def cache(self):
        """Full ``p²``-grid :class:`ActivationCache` (one forward pass, reused)."""
        if self._cache is None:
            from .activations import cache_all

            self._cache = cache_all(self.model, self.ds)
        return self._cache

    @property
    def logits_grid(self) -> torch.Tensor:
        """``(p-1, p-1, vocab)`` final-position logits over the units grid (float64)."""
        if self._logits_grid is None:
            from .grids import compute_logits_grid

            self._logits_grid = compute_logits_grid(self.model, self.ds)
        return self._logits_grid

    # ---------- accessors shared by measures ----------

    def char_energy(self, W: Optional[torch.Tensor] = None) -> np.ndarray:
        """Per-character energy of ``W`` (defaults to W_E) in the natural basis."""
        from .metrics import char_energy

        if W is None:
            W = self.model.embed.W_E[:, : self.ds.p].detach()
        return char_energy(W, self.basis, self.ci)

    def per_neuron_dominant_char(self) -> tuple[np.ndarray, np.ndarray]:
        """``(char_E[d_mlp, n_chars], dominant_char[d_mlp])``; cached after first call."""
        if self._dom is None:
            from .neurons import per_neuron_dominant_char

            self._dom = per_neuron_dominant_char(
                self.model.unembed.W_U.detach(),
                self.model.blocks[0].mlp.W_out.detach(),
                self.basis,
                self.ci,
                self.ds.p,
            )
        return self._dom

    def cluster_signal(self, k: int) -> Optional[np.ndarray]:
        """Reconstruct the (a, b) → χ_k(ab) signal from the χ_k neuron cluster.

        Returns the (p-1, p-1) residual-stream signal aligned with the
        unembed's χ_k read direction, or None if no neurons dominate at k.
        """
        from .neurons import cluster_signal as _cs

        _, dom = self.per_neuron_dominant_char()
        cluster = np.where(dom == k)[0]
        if len(cluster) == 0:
            return None
        cos_k = self.basis[self.ci.cos[k]]
        W_U = self.model.unembed.W_U.detach()
        W_U_k = W_U[:, : self.ds.p].double() @ cos_k.double()
        W_U_k = W_U_k / (W_U_k.norm() + 1e-12)
        return _cs(self.model, self.ds, cluster, W_U_k)

    def reference_signal(self, k: int) -> np.ndarray:
        """Algebraic reference ``cos(θ_k(a) + θ_k(b))`` over the (a, b) grid."""
        from .neurons import reference_cos_signal

        return reference_cos_signal(k, self.ds.p)

    # ---------- metrics and forward verbs ----------

    def evaluate(self) -> tuple[float, float, float]:
        """``(train_loss, test_loss, accuracy)`` at current weights."""
        from .ablate import evaluate_loss

        return evaluate_loss(self.model, self.ds)

    def run_with_cache(self, inputs: torch.Tensor):
        """Forward + full cache; returns ``(logits, ActivationCache)``."""
        from .activations import run_with_cache

        return run_with_cache(self.model, inputs)

    def test_loss_metric(self, logits: torch.Tensor) -> float:
        """Standard patching metric: test-set cross-entropy at the final answer position."""
        import torch.nn.functional as F

        tm = self.ds.test_mask
        y = self.ds.labels[tm]
        lg = logits[:, -1, : self.ds.n_answer_tokens][tm].double()
        return F.cross_entropy(lg, y).item()
