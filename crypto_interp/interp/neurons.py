"""Per-neuron analysis utilities.

The mod-mul algorithm prediction (see ``experiments/001_mul_p113``): each MLP
neuron specializes to a single multiplicative-Fourier frequency k, with its
firing pattern concentrated on the four "frequency-matched" bigrams
(cos k, cos k), (sin k, sin k), (sin k, cos k), (cos k, sin k) in the 2D
multiplicative-basis decomposition. These helpers compute that decomposition.
"""

import torch


def matched_bigrams(k: int, n: int) -> list[tuple[int, int]]:
    """The 4 (or 1) basis-index pairs that carry energy for a neuron at frequency k.

    Basis layout (matches :func:`crypto_interp.interp.bases.multiplicative_fourier_basis`):
        cos k → basis index ``2k``
        sin k → basis index ``2k+1``
    For ``k = n/2`` (only when ``n`` is even), sin doesn't exist and we
    return only the (cos, cos) pair.
    """
    ci, si = 2 * k, 2 * k + 1
    if n % 2 == 0 and k == n // 2:
        return [(n, n)]
    return [(ci, ci), (si, si), (ci, si), (si, ci)]


def compute_per_neuron_frequency_energy(
    coef_2d: torch.Tensor,
    p: int,
) -> tuple[torch.Tensor, list[int]]:
    """Sum each neuron's energy over the matched-bigram set for every frequency.

    Args:
        coef_2d: (p_basis, p_basis, d_mlp) — 2D multiplicative-Fourier coefficients
            of each neuron's (p, p) firing pattern.
        p: prime modulus.

    Returns:
        out: (n_freqs, d_mlp) — row k is the energy each neuron carries at
            frequency ``freq_indices[k]``.
        freq_indices: list of frequency indices (1..(n-1)//2 and n/2 if n even).
    """
    n = p - 1
    d_mlp = coef_2d.shape[-1]
    freq_indices = list(range(1, (n - 1) // 2 + 1))
    if n % 2 == 0:
        freq_indices.append(n // 2)

    out = torch.zeros(len(freq_indices), d_mlp, dtype=coef_2d.dtype, device=coef_2d.device)
    for i, k in enumerate(freq_indices):
        bigrams = matched_bigrams(k, n)
        e = sum((coef_2d[a, b] ** 2) for a, b in bigrams)
        out[i] = e
    return out, freq_indices
