"""Session — the analysis bundle.

A :class:`Session` packages ``(model, ds, task, basis, ci)`` together so that
analyses are method calls rather than 4-argument function calls. It does not
add new logic; every method dispatches to an existing canonical function in
``crypto_interp.interp`` (or ``crypto_interp.analysis``).

Two lazy properties — ``cache`` (the full ``p²`` :class:`ActivationCache`) and
``logits_grid`` — are built on first access and reused, so an analysis that
runs several queries against the same forward pass pays the forward cost
once.

Construction::

    from crypto_interp.interp import Session
    S = Session.from_run("experiments/003/.../seed1")
    K = S.essential()["K"]
    helpers = S.helpers(K)
    f_diff, spectrum, dom = S.delta_k_spectrum(33)
    sizes = S.cluster_sizes(K)

Example (runnable on the seed-1 fixture)::

    >>> from crypto_interp.interp import Session
    >>> S = Session.from_run(
    ...     "experiments/003_dmodel_sweep_p113/runs/dmodel_24_dmlp_20_wd2_seed1")
    >>> sorted(S.essential()["K"])
    [8, 10, 33, 46, 52]
    >>> [(h, m, mult) for (h, m, mult, _e) in S.helpers([8, 10, 33, 46, 52])]
    [(8, 52, 2), (46, 33, 2)]
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np
import torch


class Session:
    def __init__(self, model, ds, task: Optional[str] = None,
                 basis: Optional[torch.Tensor] = None, ci=None):
        self.model = model
        self.ds = ds
        self.task = task or getattr(ds, "task", "mul")
        if basis is None or ci is None:
            from .bases import char_index
            # v1: both ``add`` and ``mul`` use the multiplicative-Fourier basis on (Z/p)*
            # for analysis; v2 will dispatch on task when ``modular_add`` is added.
            basis, ci = char_index(ds.p)
        self.basis = basis
        self.ci = ci
        self._cache = None
        self._logits_grid = None
        self._dom = None  # cached per-neuron dominant-char tensor

    # ---------- constructors ----------

    @classmethod
    def from_run(cls, run_dir, device: str = "cpu") -> "Session":
        """Load model + dataset from a run directory; build basis/char_index from ds.p."""
        from .load import load_run, latest_checkpoint
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
        """``(p-1, p-1, vocab)`` answer-position logits over the (a, b) grid."""
        if self._logits_grid is None:
            from .grids import compute_logits_grid
            self._logits_grid = compute_logits_grid(self.model, self.ds)
        return self._logits_grid

    # ---------- algebraic-substrate verbs ----------

    def char_energy(self, W: Optional[torch.Tensor] = None) -> np.ndarray:
        """Per-character energy of ``W`` (defaults to W_E) in the natural basis."""
        from .metrics import char_energy
        if W is None:
            W = self.model.embed.W_E[:, : self.ds.p].detach()
        return char_energy(W, self.basis, self.ci)

    def essential(self, threshold: float = 0.05) -> dict:
        """Essential characters K + per-char ablation (energy, ablated loss, Δlog10, class)."""
        from .ablate import essential_characters
        return essential_characters(self.model, self.ds, self.ci, self.basis,
                                    threshold=threshold)

    def helpers(self, K=None, max_mult: int = 8) -> list:
        """Primary/helper classification on K: ``[(helper_k, primary_m, mult, energy), ...]``."""
        from .harmonic import find_primary_helper_pairs
        if K is None:
            K = self.essential()["K"]
        return find_primary_helper_pairs(self.model, self.ds, self.ci, self.basis,
                                         K, max_mult=max_mult)

    def delta_k_spectrum(self, k: int):
        """Δ_k(Δlog) 1-D reduction + Fourier spectrum + dominant freq. See ``harmonic.py``."""
        from .harmonic import delta_k_spectrum
        return delta_k_spectrum(self.model, self.ds, self.ci, self.basis, k)

    def order(self, k: int) -> int:
        from .metrics import order_of
        return order_of(k, self.ds.p)

    # ---------- mech-interp-discovered verbs ----------

    def per_neuron_dominant_char(self) -> tuple[np.ndarray, np.ndarray]:
        """``(char_E[d_mlp, n_chars], dominant_char[d_mlp])``; cached after first call."""
        if self._dom is None:
            from ..analysis.neuron_clusters import per_neuron_dominant_char
            self._dom = per_neuron_dominant_char(
                self.model.unembed.W_U.detach(),
                self.model.blocks[0].mlp.W_out.detach(),
                self.basis, self.ci, self.ds.p)
        return self._dom

    def cluster_sizes(self, K=None) -> dict[int, int]:
        """Count of neurons whose argmax-character is each ``k ∈ K`` (free-rider check)."""
        if K is None:
            K = self.essential()["K"]
        _, dom = self.per_neuron_dominant_char()
        cnt = Counter(dom.tolist())
        return {int(k): int(cnt.get(int(k), 0)) for k in K}

    def cluster_signal(self, k: int) -> Optional[np.ndarray]:
        """Reconstruct the (a, b) → χ_k(ab) signal from the χ_k neuron cluster.

        Returns the (p-1, p-1) residual-stream signal aligned with the
        unembed's χ_k read direction, or None if no neurons dominate at k.
        """
        from ..analysis.neuron_clusters import cluster_signal as _cs
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
        from ..analysis.neuron_clusters import reference_cos_signal
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

    # ---------- dynamics ----------

    def dynamics(self, metrics_path: Optional[Path] = None) -> dict:
        """Trajectory markers (bifurcation / commit / cliff) from a run's metrics.pt.

        ``metrics_path`` defaults to ``<ds.run_dir>/metrics.pt`` if available — pass
        explicitly otherwise. Returns a dict ``{"bifurcation", "commit", "cliff",
        "K", "epochs"}``.
        """
        from .dynamics import bifurcation_step, commit_step, cliff_step
        if metrics_path is None:
            raise ValueError("metrics_path required (point at the run's metrics.pt)")
        M = torch.load(metrics_path, weights_only=False)
        L = torch.load(Path(metrics_path).with_name("losses.pt"), weights_only=False)
        ep = np.asarray(M["epochs"]); fe = np.asarray(M["freq_energies"])
        te = np.asarray(L["test_losses"])
        n_chars = max(self.ci.freqs)
        final = fe[-1]
        K = sorted(int(k) for k in self.ci.freqs if final[k - 1] >= 0.05 * final.max())
        Km = np.zeros(n_chars, bool); Km[[k - 1 for k in K]] = True
        bi = bifurcation_step(fe, Km, 1.5); co = commit_step(fe, K, "subset")
        cl = cliff_step(te, 0.1)
        return {
            "K": K,
            "epochs": ep,
            "bifurcation": int(ep[bi]) if bi is not None else None,
            "commit": int(ep[co]) if co is not None else None,
            "cliff": int(cl) if cl is not None else None,
        }
