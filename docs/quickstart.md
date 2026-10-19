# Quickstart

This page analyses the trained model that ships with the repository, first
from the command line and then from Python. It needs no GPU and no training,
and takes about a minute on a laptop.

The model is `tests/fixtures/mul_p113_d24_m32_seed1`: a one-layer transformer
that learned multiplication mod 113. Its input is `a b =`, and it reads the
product `a·b mod 113` from the last position. It has `d_model=24`, `d_mlp=32`,
and was trained for 10,000 epochs on 30% of all pairs with seed 1.

```bash no-run
pip install -e .
```

## From the command line

A *run* is a directory of checkpoints. `runs ls` reads its config:

```bash
crypto-interp runs ls --run tests/fixtures/mul_p113_d24_m32_seed1
```

A *measure* is a named analysis of one checkpoint (the latest, by default).
`measure` prints its scalar fields as a table:

```bash
crypto-interp measure kernel_noise --run tests/fixtures/mul_p113_d24_m32_seed1
```

```text
                         run               K    L_emp        L_sym  L_emp_test   L_sym_test          L_K  kernel_margin  crt_gcd  crt_floor  regime
tests/mul_p113_d24_m32_seed1 (3, 10, 20, 51) 0.000125 2.264117e-08    0.000179 2.432381e-08 2.669832e-08      18.524425        1        0.0 grokked
```

How to read it:

- **K** is the set of characters the model uses: those χ_k carrying at least 5%
  of the largest per-character energy in the embedding. A character χ_k maps
  a token x to the angle θ_k(x) = 2πk·dlog(x)/112, so the model needs one
  "clock" per k.
- **L_emp** is the model's actual cross-entropy over all (a, b), while
  **L_sym** is the loss of its symmetric part alone: the logit as a function
  of how far the candidate is from the answer, in discrete-log steps. A tiny
  L_sym means the algorithm is exact, and the gap L_emp − L_sym is noise on
  top of it. The `_test` versions use only the pairs held out from training,
  which memorization cannot help with.
- **crt_gcd** = gcd(K ∪ {112}). When it is 1, the characters in K jointly
  separate every element of (ℤ/113)*, so the answer is pinned down uniquely
  (the CRT step). When it is g > 1, no model using only K can do better than
  loss log g. That is the *CRT-fail* regime.
- **regime** combines these: grokked, algorithmic-but-noisy, CRT-fail or
  intermediate. See [measures](measures.md#kernel-and-noise).

Two more measures show *how* the characters are implemented:

```bash
crypto-interp measure helpers --run tests/fixtures/mul_p113_d24_m32_seed1
crypto-interp measure neuron_clusters --run tests/fixtures/mul_p113_d24_m32_seed1
```

`helpers` finds that χ20 is not computed in its own right: ablating it
destroys frequency 10, so it is the second harmonic of χ10, a by-product of the
ReLU. `neuron_clusters` assigns each MLP neuron to the character it writes to
most. The cluster sizes are (11, 13, 0, 8): the helper χ20 has no neurons.

Results are cached in the run directory under `measures/`, keyed by the
checkpoint's contents and the measure's version, so asking again is instant.

### Figures

```bash
crypto-interp figure list
crypto-interp figure cluster --run tests/fixtures/mul_p113_d24_m32_seed1 --k 10 --out figs/cluster.png
crypto-interp figure walkthrough --run tests/fixtures/mul_p113_d24_m32_seed1 --a 7 --b 12 --out figs/walkthrough.png
```

`cluster` is the figure in the README. `walkthrough` follows one input,
7 × 12, through the model: each token's position on each clock, the angles
adding, the neuron clusters responding, and the per-character logit waves
summing to a peak at 84.

## From Python

The same measures are plain functions. A `Session` holds one model, its
dataset and its character basis, and runs the forward passes the measures
share:

```pycon
>>> from crypto_interp import measures, runs
>>> from crypto_interp.interp import Session
>>> run = runs.Run.open("tests/fixtures/mul_p113_d24_m32_seed1")
>>> S = Session.from_run(run)
>>> measures.energy_support(S).K
(3, 10, 20, 51)
>>> h = measures.helpers(S)
>>> h.primaries, h.helpers, h.helper_of, h.helper_mult
((3, 10, 51), (20,), (10,), (2,))

```

Calling a measure on a `Session` computes it directly. `.at(checkpoint)`
goes through the cache, and its result also carries the array fields:

```pycon
>>> kn = measures.kernel_noise.at(run.latest())
>>> kn.regime, kn.crt_gcd
('grokked', 1)
>>> kn.kappa_obs.shape  # the observed kernel, one value per dlog offset
(112,)
>>> e = measures.errors.at(run.latest())
>>> e.n_wrong, e.n_total  # one wrong (a, b) pair out of 112²
(1, 12544)

```

`.over(...)` takes a run or a set of runs and returns a pandas table, one row
per run (or per checkpoint, with `checkpoints="all"`), with the run's config
alongside:

```pycon
>>> df = measures.neuron_clusters.over(run)
>>> print(df[["run", "d_mlp", "K", "sizes"]].to_string(index=False))
                         run  d_mlp               K          sizes
tests/mul_p113_d24_m32_seed1     32 (3, 10, 20, 51) (11, 13, 0, 8)

```

## Next

- The [tutorial](tutorial.md) trains this model from scratch and analyses a
  population of runs.
- [Measures](measures.md) lists everything that can be measured.
