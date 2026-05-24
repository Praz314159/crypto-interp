"""Progress measures: per-frequency energy in the embedding over training.

Tracks how much of the embedding's L² norm lives in each multiplicative-Fourier
frequency over the course of training. Used to visualize the grokking transition
as energy migrates from the "uniform" (all-frequencies) initialization to
sparse concentration on a few key frequencies.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from .bases import CharIndex


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


# ---------- Per-character energy (keyed by CharIndex) ----------

def char_energy(W_E_values: torch.Tensor, basis: torch.Tensor, ci: CharIndex) -> np.ndarray:
    """Per-character energy of ``W_E_values`` (d_model, p), as a numpy array
    indexed by ``k-1`` (length ``max(ci.freqs)``). Same quantity as
    ``per_frequency_energy_from_embedding`` (re-keyed for script use)."""
    coef = torch.einsum("kp,dp->kd", basis.to(torch.float64), W_E_values.to(torch.float64))
    E = (coef ** 2).sum(dim=1)
    out = np.zeros(max(ci.freqs))
    for k, rows in ci.by_char.items():
        out[k - 1] = float(E[rows].sum())
    return out


def char_energy_batch(W_E_stack: torch.Tensor, basis: torch.Tensor, ci: CharIndex) -> np.ndarray:
    """Per-character energy over a time axis. ``W_E_stack`` is (T, d_model, p);
    returns (T, max(ci.freqs))."""
    coef = torch.einsum("kp,tdp->tkd", basis.to(torch.float64), W_E_stack.to(torch.float64))
    E = (coef ** 2).sum(dim=2)  # (T, n_basis)
    out = np.zeros((W_E_stack.shape[0], max(ci.freqs)))
    for k, rows in ci.by_char.items():
        out[:, k - 1] = E[:, rows].sum(dim=1).cpu().numpy()
    return out


def order_energy(W_E_values: torch.Tensor, basis: torch.Tensor, ci: CharIndex) -> dict[int, float]:
    """Aggregate per-character energy by character order. Returns {order: energy}."""
    ce = char_energy(W_E_values, basis, ci)
    out: dict[int, float] = {}
    for k in ci.freqs:
        o = order_of(k, ci.p)
        out[o] = out.get(o, 0.0) + float(ce[k - 1])
    return out


# ---------- Small reusable metrics ----------

def order_of(k: int, p: int) -> int:
    """Multiplicative order of character k in the character group of (Z/p)*."""
    n = p - 1
    return n // math.gcd(k, n)


def correlate(x, y) -> float:
    """Mean-centered cosine similarity (Pearson correlation) of two arrays."""
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    x = x - x.mean()
    y = y - y.mean()
    return float(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-12))


def find_cliff(losses, thresh: float = 0.1) -> int | None:
    """First index where ``losses`` drops below ``thresh``; None if it never does."""
    arr = np.asarray(losses, dtype=np.float64)
    idx = np.where(arr < thresh)[0]
    return int(idx[0]) if len(idx) else None


def topk_recall(scores, K_set, k: int | None = None) -> float:
    """Fraction of ``K_set`` (1-based character ids) present in the top-k of
    ``scores`` (indexed by k-1). Defaults k = |K_set|."""
    scores = np.asarray(scores, dtype=np.float64)
    K_set = set(int(x) for x in K_set)
    k = k if k is not None else len(K_set)
    top = set((np.argsort(scores)[::-1][:k] + 1).tolist())
    return len(top & K_set) / max(len(K_set), 1)
