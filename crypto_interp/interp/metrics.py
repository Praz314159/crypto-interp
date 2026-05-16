"""Progress measures: per-frequency energy in the embedding over training.

Tracks how much of the embedding's L² norm lives in each multiplicative-Fourier
frequency over the course of training. Used to visualize the grokking transition
as energy migrates from the "uniform" (all-frequencies) initialization to
sparse concentration on a few key frequencies.
"""

from __future__ import annotations

import numpy as np
import torch


def per_frequency_energy_from_embedding(
    W_E_values: torch.Tensor,
    mul_basis: torch.Tensor,
    p: int,
) -> list[float]:
    """Return one number per frequency: the energy ``W_E`` carries at that frequency.

    Args:
        W_E_values: (d_model, p) value-token embedding (drop the '=' column).
        mul_basis: (p, p) orthonormal multiplicative-Fourier basis.
        p: prime modulus.

    Returns:
        A list of length n_freqs giving the per-frequency energy in the order
        ``k = 1, 2, ..., (p-1-1)//2`` and then ``k = (p-1)/2`` if ``p-1`` is even.
        Per-frequency energy is the sum of cos and sin contributions for that k.
    """
    coef = torch.einsum("kp,dp->kd", mul_basis.to(torch.float64),
                         W_E_values.to(torch.float64))
    per_basis_energy = (coef ** 2).sum(dim=1)  # (p,)
    n = p - 1
    freq_e: list[float] = []
    for k in range(1, (n - 1) // 2 + 1):
        freq_e.append((per_basis_energy[2 * k] + per_basis_energy[2 * k + 1]).item())
    if n % 2 == 0:
        freq_e.append(per_basis_energy[n].item())
    return freq_e


class EmbeddingEnergyTracker:
    """Accumulates per-frequency embedding energy snapshots over training.

    Usage:
        tracker = EmbeddingEnergyTracker(p=ds.p, device=device)
        for epoch in ...:
            ...
            if epoch % every == 0:
                tracker.record(epoch, model.embed.W_E[:, :ds.p])
        tracker.save(run_dir / "metrics.pt", config={...})
    """

    def __init__(self, p: int, device: str | torch.device = "cpu"):
        from .bases import multiplicative_fourier_basis
        self.p = p
        basis, _, _ = multiplicative_fourier_basis(p, device="cpu")
        self.basis = basis.to(device)
        self.epochs: list[int] = []
        self.freq_energies: list[list[float]] = []

    def record(self, epoch: int, W_E_values: torch.Tensor) -> None:
        with torch.no_grad():
            self.epochs.append(int(epoch))
            self.freq_energies.append(
                per_frequency_energy_from_embedding(W_E_values, self.basis, self.p)
            )

    def restore(self, payload: dict) -> None:
        """Restore state from a previously saved ``metrics.pt`` payload."""
        self.epochs = list(payload["epochs"])
        self.freq_energies = [list(row) for row in payload["freq_energies"]]

    def save(self, path, config: dict | None = None) -> None:
        torch.save(
            {
                "epochs": self.epochs,
                "freq_energies": np.array(self.freq_energies),
                "config": config or {},
            },
            path,
        )
