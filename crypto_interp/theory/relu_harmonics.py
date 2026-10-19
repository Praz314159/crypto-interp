"""Why doubling helpers are free: ReLU's second harmonic.

A ReLU applied to a cosine has a second-harmonic component,

    ReLU(cos θ) = 1/π + (1/2) cos θ + (2/(3π)) cos 2θ − (2/(15π)) cos 4θ + …,

so a neuron reading χ_k also produces χ_2k. (For |cos θ| = ReLU(z) + ReLU(−z) the
coefficient doubles to 4/(3π).) The capacity claim built on this: a small
ReLU MLP approximates the character product cos(k(φ_a + φ_b)) better when it
also gets the 2k features. That is what a helper χ_2k in the embedding
provides.

- :func:`relu_cos_coefficients`: the Fourier coefficients of ReLU(cos θ) and |cos θ|.
- :func:`product_fit_fvu`: the fraction of variance a width-W ReLU MLP leaves
  unexplained when fitting cos(k(φ_a + φ_b)) on the group Z_n, with and
  without the 2k features.
"""

from __future__ import annotations

import math

import numpy as np
import torch


def relu_cos_coefficients(max_m: int = 4, n_points: int = 200_000) -> dict[str, np.ndarray]:
    """Cosine coefficients a_0..a_max_m of ReLU(cos θ) and |cos θ| (dense quadrature)."""
    theta = np.linspace(0.0, 2 * np.pi, n_points, endpoint=False)
    out = {}
    for name, sig in [("relu", np.maximum(np.cos(theta), 0.0)), ("abs", np.abs(np.cos(theta)))]:
        a = [sig.mean()] + [2.0 * np.mean(sig * np.cos(m * theta)) for m in range(1, max_m + 1)]
        out[name] = np.array(a)
    return out


#: Closed forms for the coefficients of ReLU(cos θ).
RELU_COS_EXACT = {0: 1 / math.pi, 1: 0.5, 2: 2 / (3 * math.pi), 3: 0.0, 4: -2 / (15 * math.pi)}


def character_features(j: torch.Tensor, n: int, ks) -> torch.Tensor:
    """[cos kφ, sin kφ] for each k in ``ks``, φ = 2πj/n."""
    phi = 2 * math.pi * j / n
    return torch.stack([f(k * phi) for k in ks for f in (torch.cos, torch.sin)], dim=-1)


def fit_fvu(
    X: torch.Tensor, y: torch.Tensor, width: int, steps: int = 4000, lr: float = 5e-2, seed: int = 0
) -> float:
    """Fraction of variance unexplained after fitting a width-``width`` ReLU MLP."""
    torch.manual_seed(seed)
    net = torch.nn.Sequential(
        torch.nn.Linear(X.shape[-1], width), torch.nn.ReLU(), torch.nn.Linear(width, 1)
    ).double()
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    y = y.reshape(-1, 1)
    for _ in range(steps):
        opt.zero_grad()
        ((net(X) - y) ** 2).mean().backward()
        opt.step()
    with torch.no_grad():
        return ((net(X) - y) ** 2).mean().item() / y.var().item()


def product_fit_fvu(n: int, k: int, widths, n_seeds: int = 4, steps: int = 4000) -> dict:
    """Median FVU over seeds for each width, fitting cos(k(φ_a + φ_b)) on Z_n from
    the χ_k features alone (``"primary"``) and with the χ_2k features (``"helper"``)."""
    ja, jb = (
        g.reshape(-1).double()
        for g in torch.meshgrid(torch.arange(n), torch.arange(n), indexing="ij")
    )
    target = torch.cos(2 * math.pi * k * (ja + jb) / n)
    X = {
        "primary": torch.cat([character_features(ja, n, [k]), character_features(jb, n, [k])], -1),
        "helper": torch.cat(
            [character_features(ja, n, [k, 2 * k]), character_features(jb, n, [k, 2 * k])], -1
        ),
    }
    return {
        "widths": list(widths),
        **{
            name: [
                float(
                    np.median([fit_fvu(x, target, w, steps=steps, seed=s) for s in range(n_seeds)])
                )
                for w in widths
            ]
            for name, x in X.items()
        },
    }
