# Measures

Every analysis in the library is a *measure*: a named, versioned function of
one checkpoint (or, for training dynamics, of one run) that returns a
dataclass. Evaluate one with

| call | on | caching | returns |
|---|---|---|---|
| `m(S, **params)` | a `Session` | none | the dataclass |
| `m.at(checkpoint, **params)` | a checkpoint file | read and write | the dataclass, arrays included |
| `m.over(target, checkpoints="latest" \| "all", **params)` | a run, a `RunSet` or a list of runs | read and write | a DataFrame: run config + scalar and tuple fields |
| `crypto-interp measure NAME --run … \| --experiment … --where …` | the selected runs | read and write | the same table, printed or `--out` CSV/Parquet |

Cached results live in `<run>/measures/<name>/v<version>/`, keyed by the
checkpoint's content hash and the parameters. Measures registered so far:
`crypto_interp.measures.REGISTRY`.

Most measures take an optional `K`, the set of characters to analyse. It
defaults to the energy support below, so every measure agrees on K unless
told otherwise.

The examples below run on the reference model:

```pycon
>>> from crypto_interp import measures as M, runs
>>> ck = runs.Run.open("tests/fixtures/mul_p113_d24_m32_seed1").latest()

```

## Background: characters

The multiplicative group (ℤ/p)* is cyclic of order p − 1. Pick a generator
g; every nonzero x is g^dlog(x). Its characters are χ_k(x) = e^{iθ_k(x)} with
θ_k(x) = 2πk·dlog(x)/(p−1), for k = 1, …, (p−1)/2 up to conjugation. They
turn multiplication into addition of angles, θ_k(ab) = θ_k(a) + θ_k(b), which
is what the network exploits. χ_k has multiplicative order (p−1)/gcd(k, p−1);
for p = 113, p − 1 = 112 = 2⁴·7.

## Characters

### `energy_support`

**K**: the characters carrying at least `threshold` (default 0.05) of the
largest per-character energy in the embedding W_E of the value tokens.

| field | |
|---|---|
| `K`, `orders` | the characters, and each one's multiplicative order |
| `energy`, `char_order` | arrays over all k (index k − 1) |

### `char_ablation`

Removes each character from W_E in turn and records the jump in test loss,
as Δlog₁₀. A jump over 2 makes it *essential*, and one between 0.5 and 2
makes it *load-bearing* (`loss_jump=(0.5, 2.0)`). This answers a different question from K: an
essential character can carry little energy, and a character in K can be
merely load-bearing. On the reference model, χ6 is essential but outside K.
It is also the control, so the control is not always inert:

```pycon
>>> a = M.char_ablation.at(ck)
>>> a.essential
(3, 6, 10, 20, 51)
>>> M.energy_support.at(ck).K
(3, 10, 20, 51)

```

| field | |
|---|---|
| `essential`, `load_bearing` | characters by size of the loss jump |
| `base_loss` | test loss before ablation |
| `control_k`, `control_dlog10` | the highest-energy character outside K and its jump: what "not in K" looks like |
| `dlog10`, `ablated_loss` | arrays over all k |

### `helpers`

For each k in K, ablates χ_k and finds which frequency of the output is
destroyed. A *primary* destroys its own. A *helper* destroys another
character's, related by a small multiplier: 2 is the second harmonic that a
ReLU produces from a primary's neurons.

```pycon
>>> h = M.helpers.at(ck)
>>> h.primaries, h.helpers, h.helper_of, h.helper_mult
((3, 10, 51), (20,), (10,), (2,))

```

| field | |
|---|---|
| `primaries`, `helpers`, `helper_of`, `helper_mult` | the classification; `helper_of[i]` is the primary of `helpers[i]` |
| `dominant`, `delta_corr` | per k in K: the frequency destroyed, and the correlation of the damage with it |
| `spectrum`, `delta_diff`, `delta_ab` | the Δ_k ablation spectra and maps |

### `neuron_clusters`

Assigns each MLP neuron to the character it writes to most strongly through
the unembedding. It then compares each cluster's summed output over (a, b)
with the reference cos(θ_k(a) + θ_k(b)), as a Pearson correlation.

| field | |
|---|---|
| `sizes`, `corr` | per k in K: neurons in the cluster, and correlation with the reference (NaN for an empty cluster) |
| `dominant_char`, `signal`, `reference` | arrays: per-neuron assignment, cluster outputs, references |

### `fourier2d`

The energy of the MLP output (`hook=` to choose another activation) over
pairs of input characters (k_a, k_b). A character-product algorithm puts it
on the diagonal k_a = k_b.

```pycon
>>> round(M.fourier2d.at(ck).diagonal_fraction, 2)
0.99

```

## Kernel and noise

### `kernel_noise`

If the model computes with the characters in K, the logit for candidate c
depends only on the offset d = dlog(ab) − dlog(c), through a kernel
κ(d) = Σ_{k∈K} α_k cos(2πkd/(p−1)) that peaks at d = 0. `kernel_noise`
measures how true that is.

| field | |
|---|---|
| `L_emp` | the model's cross-entropy over all (a, b) of units |
| `L_sym` | the loss of κ_obs, the mean logit at each offset: the model's symmetric part |
| `L_K` | the loss of κ_obs restricted to K |
| `kernel_margin` | κ_obs(0) minus its largest other value |
| `crt_gcd`, `crt_floor` | g = gcd(K ∪ {p−1}), and log g |
| `regime` | see below |
| `kappa_obs`, `kappa_K`, `alpha` | the observed kernel, its K-restriction, and the amplitudes α_k |

The **regime** is assigned in this order:

1. **CRT-fail** if K is non-empty and g > 1. The characters in K then cannot
   tell apart the g elements of a coset of the order-(p−1)/g subgroup. Any
   kernel on K has period (p−1)/g, so L_K ≥ log g.
2. Otherwise **grokked** if L_emp < 0.01,
3. **algorithmic-but-noisy** if only L_sym < 0.01,
4. and **intermediate** if neither.

> **Caveat.** L_sym averages over all pairs, the training pairs included. A
> model that only memorized its training pairs can therefore score a small
> L_sym and be labelled algorithmic-but-noisy. Read the regime together with
> `grok_status`.

### `noise_modes`

Writes the logits in discrete-log coordinates as κ_obs(x_a + x_b − x_c) plus
an asymmetric residual ε, and splits ε's 3-D Fourier power into classes:
`single_a`, `single_b` (the output depends on one input alone), `single_c`
(a bias toward some answers), `pair_ac`, `pair_bc` (shortcuts through one
input), `pair_ab` (an input interaction that is never read out),
`off_resonance` (the right product, read out at the wrong character) and
`generic`. `algorithm` is the kernel's own power. `top_classes` and
`top_modes` list the strongest modes.

## Errors and clocks

### `errors`

For every (a, b): margin = logit of the answer − largest other logit, which
is positive exactly when the model is right. Errors are counted per input
token and per answer. A token with an error rate over 1% in either position
is a *defect token*.

```pycon
>>> e = M.errors.at(ck)
>>> e.n_wrong, e.defect_tokens
(1, ())

```

| field | |
|---|---|
| `n_wrong`, `n_total`, `accuracy` | |
| `margin_min`, `margin_p1`, `margin_median` | the margin distribution |
| `defect_tokens` | tokens erring on more than 1% of their pairs |
| `margin`, `wrong`, `err_by_a`, `err_by_b`, `err_by_target`, … | arrays |

### `clock_residuals`, `clock_defect`, `clock_repair`

Each token's embedding should sit on a clock:
W_E[:, x] ≈ Σ_k C_k cos θ_k(x) + S_k sin θ_k(x). The coefficients are fitted
per run by least squares, so a clean clock scores zero, whatever its shape.

- `clock_residuals`: each token's deviation from the fit (`in_dev`), per
  character plane (`plane_dev`), and the share of its embedding outside the
  clock subspace (`leak`).
- `clock_defect`: the clock k\* on which the defect tokens are most
  displaced, and how tightly they cluster in angle on each clock (`arc_R_by_k`,
  against a null `arc_R_null`).
- `clock_repair`: a causal test. Move the defect tokens to their ideal
  positions on k\* and recount wrong pairs (`wrong_repair_defects`). Two
  controls do the same for as many random clean tokens
  (`wrong_patch_random_clean`) and for every token (`wrong_straighten_clock`).

The reference model has no defect tokens, so `kstar` is `None` there.
`measures.token_table(pop)` joins `errors` and `clock_residuals` into one
row per (run, token).

## Training dynamics

These are *run measures*: functions of a run's training record, not of one
checkpoint, and cheap enough not to cache. The reference run in
`tests/fixtures/` ships without its training record, so they need a run you
trained.

### `grok_status`

From the loss history: `memorized_epoch` (train loss first below
`mem_thresh`), `cliff_epoch` (test loss first below `grok_thresh`), and
`status`, one of `grokked`, `memorized-only` or `no-fit`.

### `dynamics_markers`

From the per-character embedding energy recorded during training
(`metrics.pt`), with K the final energy support:

- `bifurcation_epoch`: when K's mean energy, relative to the other
  characters', first exceeds `ratio` (default 1.5) times its initial value;
- `commit_epoch`: after which the top |K| characters always include K;
- `cliff_epoch`: as in `grok_status`.
