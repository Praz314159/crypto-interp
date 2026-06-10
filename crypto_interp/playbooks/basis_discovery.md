# Basis discovery — find the natural basis

**Strategy 1.** In what basis is the embedding sparse? Every other analysis
presupposes you know the representational basis the model is using.

## Algebraic anchor

The character group of the task's domain. For modular multiplication on
`(Z/p)*`, the natural basis is the multiplicative-Fourier basis — the
discrete-log pulls `(Z/p)*` over to `Z/(p−1)` and additive Fourier on
`Z/(p−1)` gives the Dirichlet characters `χ_k` of `(Z/p)*`. For modular
addition on `(Z/p, +)`, the natural basis is the additive Fourier basis on
`Z/p` directly.

## Verbs

- `S.basis` — the basis tensor, ``(p, p)`` orthonormal rows.
- `S.ci` — the `CharIndex` mapping character `k → (cos_row, sin_row)`.
- `S.char_energy(W=None)` — per-character energy in W (defaults to W_E),
  returns `np.ndarray` indexed by `k-1`.
- `S.essential(threshold=0.05)` — full per-character ablation: returns
  `{per_char: {k → {energy, ablated_loss, dlog10, cls}}, base_loss, K}`.
  `K` is the essential set (`energy ≥ threshold × energy.max()`).

## Recipe

```python
from crypto_interp.interp import Session

S = Session.from_run("experiments/003_dmodel_sweep_p113/runs/dmodel_24_dmlp_20_wd2_seed1")
ess = S.essential(threshold=0.05)
K = ess["K"]
print(f"K = {K}  orders = {[S.order(k) for k in K]}")
for k in K:
    info = ess["per_char"][k]
    print(f"  χ{k:>3}  energy={info['energy']:6.2f}  dlog10={info['dlog10']:+.2f}  ({info['cls']})")
```

Expected on the seed-1 fixture: ``K = [8, 10, 33, 46, 52]`` with orders
``[14, 56, 112, 56, 28]``, the three primaries (`10, 33, 52`) showing
Δlog10 ≈ 7 (essential), the two helpers (`8, 46`) showing Δlog10 ≈ 3–4
(also essential, but lower).

## Common variations

- **Sweep the threshold.** ``ess = S.essential(threshold=0.01)`` lets in
  more characters; ``0.10`` is stricter. Use to study which characters are
  *load-bearing* vs *vestigial*.
- **Energy without ablation.** ``S.char_energy()`` returns just the W_E
  energy distribution (no forward passes), useful for fast iteration
  during a training-time scan.
- **Project onto K and re-evaluate.** Combine with `ablate_embedding` to
  test sufficiency: how does the model perform restricted to K?
