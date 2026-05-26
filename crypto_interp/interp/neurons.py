"""Per-neuron analysis utilities.

Two clusters of helpers:

- Frequency-matched-bigram decomposition: :func:`matched_bigrams`,
  :func:`compute_per_neuron_frequency_energy`. Sums each neuron's 2D
  multiplicative-Fourier energy over the 4-bigram pattern of each frequency.
- Output-side cluster identification: :func:`per_neuron_dominant_char`,
  :func:`cluster_signal`, :func:`reference_cos_signal`. Projects each
  neuron's ``W_U @ W_out`` direction onto characters to identify each
  neuron's dominant frequency; reconstructs the cluster's contribution to
  the residual stream on the (a, b) input grid; provides the algebraic
  ground-truth ``cos(θ_k(a)+θ_k(b))`` reference for mechanism verification.

The output-side helpers live here (rather than in ``analysis/``) so the
intervention/cache layer can depend on them — analyses always depend on
``interp``, never the other way around.
"""

import numpy as np
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


# ---------- output-side cluster identification ----------

def per_neuron_dominant_char(W_U: torch.Tensor, W_out: torch.Tensor,
                             basis: torch.Tensor, ci, p: int):
    """Return ``(char_E[d_mlp, n_chars], dominant_char[d_mlp])`` (1-based char ids).

    Projects ``W_U @ W_out`` onto the character basis to identify each neuron's
    dominant output character. Used to find the cluster of neurons specialized
    to each character k (mechanism verification, free-rider economy analysis).
    """
    V = W_U[:, :p].double().T @ W_out.double()        # (p, d_mlp)
    coef = basis @ V                                   # (n_basis, d_mlp)
    E = coef ** 2
    nch = max(ci.freqs)
    char_E = np.zeros((W_out.shape[1], nch))
    for k, rs in ci.by_char.items():
        char_E[:, k - 1] = E[rs].sum(dim=0).cpu().numpy()
    return char_E, char_E.argmax(axis=1) + 1


def cluster_signal(model, ds, cluster_neurons, W_U_k: torch.Tensor) -> np.ndarray:
    """Cluster's projection onto the unembed's character-k direction over the
    full (a, b) grid. Returns ``(p-1, p-1)``.

    Reconstructs what the cluster of neurons contributes to the residual stream
    along the direction the unembed reads as χ_k. Comparing this to
    :func:`reference_cos_signal` is the mechanism-verification recipe.
    """
    p = ds.p
    aa, bb = torch.meshgrid(torch.arange(1, p), torch.arange(1, p), indexing="ij")
    eq = torch.full_like(aa, ds.eq_token)
    inputs = torch.stack([aa, bb, eq], dim=-1).reshape(-1, 3)
    cache: dict[str, torch.Tensor] = {}
    model.remove_all_hooks()
    model.cache_all(cache)
    try:
        with torch.no_grad():
            _ = model(inputs)
    finally:
        model.remove_all_hooks()
    h = cache["blocks.0.mlp.hook_post"][:, -1, :].double()
    W_out = model.blocks[0].mlp.W_out.detach().double()
    cluster_resid = h[:, cluster_neurons] @ W_out[:, cluster_neurons].T
    sig = cluster_resid @ W_U_k.double()
    return sig.reshape(p - 1, p - 1).cpu().numpy()


def reference_cos_signal(k: int, p: int) -> np.ndarray:
    """Algebraic ground truth ``cos(θ_k(a) + θ_k(b))`` over the (a, b) grid for
    a ∈ {1,...,p-1}, b ∈ {1,...,p-1}, with θ_k(a) = 2πk·dlog(a)/(p-1). The
    target that a primary χ_k neuron cluster should reproduce."""
    from .bases import discrete_log_table
    _, dlog = discrete_log_table(p)
    a_dlog = np.array([dlog[a] for a in range(1, p)])
    theta = 2 * np.pi * k * a_dlog / (p - 1)
    return np.cos(theta[:, None] + theta[None, :])
