"""Which characters a model uses, and how.

- :func:`energy_support`: **K**, the characters carrying at least ``threshold``
  (default 5%) of the largest per-character energy in the value-token
  embedding. This is the K every other measure uses by default.
- :func:`char_ablation`: what removing each character from W_E does to the
  test loss, classified as vestigial, load-bearing or essential by the jump
  in log10 loss. It is a different question from K: an essential character
  can carry little energy, and a member of K can be only load-bearing.
- :func:`helpers`: for each k in K, the frequency that ablating χ_k
  destroys. A primary destroys its own; a helper destroys another one, by a
  small power relation (2 = the Sylow-2 doubling via ReLU's squared term).
- :func:`neuron_clusters`: the MLP neurons dominated by each k, and how well
  each cluster's output reproduces cos(θ_k(a) + θ_k(b)).
- :func:`fourier2d`: the MLP output's energy over character pairs (k_a, k_b);
  a character-product algorithm puts it on the diagonal.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .core import measure


@dataclass
class EnergySupport:
    K: tuple[int, ...]
    orders: tuple[int, ...]  # multiplicative order of each χ_k in K
    threshold: float
    energy: np.ndarray  # (n_chars,) energy of W_E at character k, indexed k − 1
    char_order: np.ndarray  # (n_chars,) multiplicative order of χ_k, indexed k − 1


@measure("energy_support", version=1)
def energy_support(S, threshold: float = 0.05) -> EnergySupport:
    """K = {k : energy_k ≥ threshold · max energy} over the value-token embedding."""
    from ..interp.metrics import order_of

    p = S.ds.p
    E = S.char_energy()
    K = tuple(int(k) for k in S.ci.freqs if E[k - 1] >= threshold * E.max())
    orders = np.array([order_of(k, p) for k in range(1, len(E) + 1)])
    return EnergySupport(K, tuple(order_of(k, p) for k in K), threshold, E, orders)


def dlog_order(p: int) -> np.ndarray:
    """Indices that sort the units 1..p−1 by discrete log (for dlog-ordered plots)."""
    from ..interp.bases import discrete_log_table

    _, dlog = discrete_log_table(p)
    return np.argsort([dlog[a] for a in range(1, p)])


def support(S, K=None) -> tuple[int, ...]:
    """``K`` if given, else the session's energy support."""
    return tuple(sorted(int(k) for k in (energy_support(S).K if K is None else K)))


@dataclass
class CharAblation:
    base_loss: float  # test loss with nothing ablated
    essential: tuple[int, ...]  # Δlog10 ≥ loss_jump[1]
    load_bearing: tuple[int, ...]  # loss_jump[0] ≤ Δlog10 < loss_jump[1]
    control_k: int  # the highest-energy character outside K
    control_dlog10: float  # its ablation impact: what "not in K" looks like
    dlog10: np.ndarray  # (n_chars,) Δlog10 test loss after ablating χ_k, indexed k − 1
    ablated_loss: np.ndarray  # (n_chars,)


@measure("char_ablation", version=1)
def char_ablation(
    S, threshold: float = 0.05, loss_jump: tuple[float, float] = (0.5, 2.0)
) -> CharAblation:
    """Ablate each character from W_E in turn and record the jump in test loss."""
    from ..interp.ablate import essential_characters

    r = essential_characters(
        S.model, S.ds, S.ci, S.basis, threshold=threshold, loss_jump=tuple(loss_jump)
    )
    per = r["per_char"]
    ks = sorted(per)
    dlog10 = np.array([per[k]["dlog10"] for k in ks])
    outside = [k for k in ks if k not in set(r["K"])]
    control = max(outside, key=lambda k: per[k]["energy"])
    return CharAblation(
        base_loss=r["base_loss"],
        essential=tuple(k for k in ks if per[k]["cls"] == "essential"),
        load_bearing=tuple(k for k in ks if per[k]["cls"] == "load-bearing"),
        control_k=int(control),
        control_dlog10=float(per[control]["dlog10"]),
        dlog10=dlog10,
        ablated_loss=np.array([per[k]["ablated_loss"] for k in ks]),
    )


@dataclass
class Helpers:
    K: tuple[int, ...]
    primaries: tuple[int, ...]  # k whose ablation destroys frequency k itself
    helpers: tuple[int, ...]  # k whose ablation destroys another frequency
    helper_of: tuple[int, ...]  # for each helper, the frequency it destroys
    helper_mult: tuple[int, ...]  # j with χ_{j·primary} = χ_helper (fold-aware); 0 if none ≤ 8
    dominant: tuple[int, ...]  # for each k in K, the dominant frequency of Δ_k
    delta_corr: tuple[float, ...]  # corr of Δ_k(Δlog) with cos(2πk·Δlog/(p−1)), per k in K
    spectrum: np.ndarray  # (|K|, (p−1)/2) Fourier energy of Δ_k(Δlog)
    delta_diff: np.ndarray  # (|K|, p−1) Δ_k reduced to the dlog offset
    delta_ab: np.ndarray  # (|K|, p−1, p−1) Δ_k(ab, c), averaged over (a, b) with the same ab


@measure("helpers", version=1)
def helpers(S, K=None, max_mult: int = 8) -> Helpers:
    """Primary/helper classification of K by the Δ_k ablation spectrum."""
    from ..interp.grids import compute_logits_grid
    from ..interp.harmonic import delta_k, helper_multiplier
    from ..interp.metrics import correlate
    from ..interp.reductions import fourier_spectrum_1d, reduce_to_ab, reduce_to_diff

    p = S.ds.p
    K = support(S, K)
    full = compute_logits_grid(S.model, S.ds)
    dominant, spectra, diffs, abs_, corr = [], [], [], [], []
    for k in K:
        grid = delta_k(S.model, S.ds, S.ci, S.basis, k, logits_full=full)
        f = reduce_to_diff(grid, p, value_axis="full")
        spec = fourier_spectrum_1d(f)
        dominant.append(int(spec.argmax() + 1))
        spectra.append(spec)
        diffs.append(f)
        abs_.append(reduce_to_ab(grid, p)[:, 1:p])
        corr.append(correlate(f, np.cos(2 * np.pi * k * np.arange(p - 1) / (p - 1))))
    pairs = [(k, m) for k, m in zip(K, dominant, strict=True) if m != k]
    return Helpers(
        K=K,
        primaries=tuple(k for k, m in zip(K, dominant, strict=True) if m == k),
        helpers=tuple(k for k, _ in pairs),
        helper_of=tuple(m for _, m in pairs),
        helper_mult=tuple(helper_multiplier(k, m, p, max_mult=max_mult) or 0 for k, m in pairs),
        dominant=tuple(dominant),
        delta_corr=tuple(float(c) for c in corr),
        spectrum=np.array(spectra),
        delta_diff=np.array(diffs),
        delta_ab=np.array(abs_),
    )


@dataclass
class NeuronClusters:
    K: tuple[int, ...]
    sizes: tuple[int, ...]  # neurons whose dominant character is k, per k in K
    corr: tuple[float, ...]  # cluster output vs cos(θ_k(a) + θ_k(b)); nan for an empty cluster
    dominant_char: np.ndarray  # (d_mlp,) each neuron's dominant character
    signal: np.ndarray  # (|K|, p−1, p−1) cluster output along the unembed's χ_k read direction
    reference: np.ndarray  # (|K|, p−1, p−1) cos(θ_k(a) + θ_k(b))


@measure("neuron_clusters", version=1)
def neuron_clusters(S, K=None) -> NeuronClusters:
    """Group MLP neurons by dominant character and check each cluster's output."""
    from ..interp.metrics import correlate

    K = support(S, K)
    n = S.ds.p - 1
    _, dom = S.per_neuron_dominant_char()
    sizes, corr, sig, ref = [], [], [], []
    for k in K:
        size = int((dom == k).sum())
        r = S.reference_signal(k)
        s = S.cluster_signal(k) if size else np.zeros((n, n))
        sizes.append(size)
        corr.append(correlate(s, r) if size else float("nan"))
        sig.append(s)
        ref.append(r)
    return NeuronClusters(
        K, tuple(sizes), tuple(corr), np.asarray(dom), np.array(sig), np.array(ref)
    )


@dataclass
class Fourier2D:
    diagonal_fraction: float  # share of the energy with k_a = k_b
    energy: np.ndarray  # (n_chars, n_chars) E[k_a − 1, k_b − 1]


@measure("fourier2d", version=1)
def fourier2d(S, hook: str = "blocks.0.hook_mlp_out") -> Fourier2D:
    """Energy of the hook's final-position output over pairs of input characters."""
    import torch

    from ..interp.grids import compute_activation_grid

    p = S.ds.p
    f_ab = compute_activation_grid(S.model, S.ds, hook)  # (p−1, p−1, width)
    basis_v = S.basis[:, 1:p].double()
    coef = torch.einsum("ka,lb,abd->kld", basis_v, basis_v, f_ab)
    E_basis = (coef**2).sum(dim=-1).cpu().numpy()
    nch = max(S.ci.freqs)
    E = np.zeros((nch, nch))
    for ka, rows_a in S.ci.by_char.items():
        for kb, rows_b in S.ci.by_char.items():
            E[ka - 1, kb - 1] = E_basis[np.ix_(rows_a, rows_b)].sum()
    return Fourier2D(float(np.trace(E) / E.sum()), E)
