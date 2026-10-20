# Tutorial: train, measure, extend

This page trains the reference model from scratch, shows how runs are laid
out and selected, and adds a new measure. Training takes minutes, so those
blocks are not re-run when the docs are tested. Everything after them runs
against the reference run in `tests/fixtures/`, which was trained by exactly
the command below.

## 1. Experiments and configs

An *experiment* is a directory `experiments/<id>_<name>/` holding a
`config.py` that defines `CONFIG`, an `ExperimentConfig`. Runs override its
fields, so one experiment holds a family of related runs. For example,
`experiments/003_dmodel_sweep_p113/config.py` is multiplication mod 113:

```python no-run
CONFIG = ExperimentConfig(
    task="mul",
    p=113,
    frac_train=0.3,  # share of the p² pairs used for training
    d_model=128,
    d_mlp=512,
    weight_decay=1.0,
    num_epochs=20_000,
    save_every=500,  # checkpoint cadence
    metrics_every=50,  # per-character energy cadence, for training dynamics
    # ... the full file also sets the optimizer and attention shapes
)
```

`crypto_interp.data.available_tasks()` lists the tasks: `add`, `mul`,
`mul_units` (multiplication on {1..p−1}² only, without 0) and
`sqrt_of_product`.

## 2. Train one run

```bash no-run
crypto-interp train --experiment 003_dmodel_sweep_p113 --seed 1 \
    --set d_model=24 --set d_mlp=32 --num-epochs 10000 --tag reference_seed1
```

This takes about nine minutes on a laptop CPU, and a few on a GPU
(`--device cuda`). Every `ExperimentConfig` field can be set with `--set`.
The run directory, `experiments/003_dmodel_sweep_p113/runs/reference_seed1/`,
then holds:

```text
checkpoint_000000.pt … checkpoint_009999.pt   every save_every epochs: weights + full config
losses.pt                                     train/test loss per epoch
metrics.pt                                    per-character embedding energy every metrics_every epochs
manifest.json                                 task, model and training config, provenance (git sha, host)
measures/                                     cached measure results, written on demand
```

Each checkpoint is self-describing: it records the task, the model config
and the training config, so a model loads without any other file. The seed
fixes the train/test split and the initialization. On the machine this page
was written on, the command reproduces the reference run's mechanism
exactly: the same K, the same helper and the same cluster sizes.

To continue a run: `crypto-interp train --experiment … --resume <checkpoint> --num-epochs 5000`.

## 3. Sweep

`sweep` trains seeds × a grid of overrides, several processes at a time:

```bash no-run
crypto-interp sweep --experiment 003_dmodel_sweep_p113 --seeds 1..20 \
    --set d_model=24 --grid d_mlp=20,32 --grid weight_decay=1,2 \
    --tag-format 'dmodel_{d_model}_dmlp_{d_mlp}_wd{weight_decay}_seed{seed}' --jobs 4
```

That is 80 runs. Rerun with `--skip-existing` to train only the missing
ones. [`infra/runpod/`](../infra/runpod/README.md) runs sweeps on a rented
GPU.

Expect variety. At small primes, grokking is sensitive to the train
fraction, the weight decay and the seed: some runs grok, some only memorize
the training pairs, and some never fit them. `grok_status` tells these apart.

## 4. Select runs

The run index finds runs by their config. `runs.query` takes an
experiment-name prefix and any config fields, each as a value, a collection
or a predicate:

```python no-run
from crypto_interp import runs

pop = runs.query(experiment="003", d_model=24, d_mlp=[20, 32], seed=range(1, 11))
wide = runs.query(p=127, d_mlp=lambda d: d >= 32)
pop.table()  # one row per run: config, latest epoch, excluded
```

The CLI takes the same selection: `--experiment 003 --where d_mlp=32 --where seed=1..10`.
Runs listed as `excluded` in their manifest (for example, a run with a known
problem) are skipped unless asked for.

A `RunSet` can also be built from run directories directly. The rest of
this page uses one holding just the reference run:

```pycon
>>> from crypto_interp import measures, runs
>>> pop = runs.RunSet([runs.Run.open("tests/fixtures/mul_p113_d24_m32_seed1")])
>>> [(run.name, run.config()["d_mlp"]) for run in pop]
[('mul_p113_d24_m32_seed1', 32)]

```

## 5. Measure a population

`.over` evaluates a measure on every run and returns one table, with the
run's config as columns, so ordinary pandas does the rest:

```pycon
>>> kn = measures.kernel_noise.over(pop)
>>> h = measures.helpers.over(pop)[["run", "helpers", "helper_mult"]]
>>> print(
...     kn.merge(h, on="run")[["d_mlp", "seed", "K", "regime", "helpers", "helper_mult"]].to_string(
...         index=False
...     )
... )
 d_mlp  seed               K  regime helpers helper_mult
    32     1 (3, 10, 20, 51) grokked   (20,)        (2,)

```

A measure over a trajectory uses every checkpoint instead of the latest:
`measures.kernel_noise.over(pop, checkpoints="all")` gives one row per
(run, epoch). Results are cached per checkpoint, so a second call, or a
figure that needs the same table, only computes what is new.

Figures take these tables. `crypto-interp figure list` names them; the same
functions are importable:

```python
from crypto_interp.figures import kernel

fig = kernel.regimes(measures.kernel_noise.over(pop))
fig.savefig("regimes.png")
```

## 6. Write a measure

A measure is a function of a `Session` that returns a dataclass. Here is one
about attention. Averaged over all inputs, every head attends equally from
`=` to `a` and to `b`, which is forced by the symmetry a·b = b·a. On a single
input, though, a head can favour one side. The measure asks how lopsided
each head is:

```python
from dataclasses import dataclass

import numpy as np

from crypto_interp.measures import measure


@dataclass
class AttentionSplit:
    lopsided: tuple[float, ...]  # per head, mean |attn(= → a) − attn(= → b)|
    split: np.ndarray  # (p−1, p−1, heads): attn(= → a) − attn(= → b)


@measure("attention_split", version=1)
def attention_split(S) -> AttentionSplit:
    pattern = S.cache.grid("attn_attn", group=True)[..., -1, :]  # from '=', over (a, b, =)
    split = (pattern[..., 0] - pattern[..., 1]).numpy()
    return AttentionSplit(tuple(np.abs(split).mean(axis=(0, 1)).tolist()), split)
```

`S.cache` runs the model once over every (a, b) and keeps every activation;
`grid(..., group=True)` arranges them as an n × n grid over the group's
elements, here the nonzero inputs. Scalar and tuple fields become table columns, and arrays stay
in the cache, returned by `.at`. Once defined, the measure is registered and
has `.at`, `.over` and the CLI (`crypto-interp measure attention_split …`):

```pycon
>>> r = attention_split.at(pop[0].latest())
>>> [round(x, 2) for x in r.lopsided]
[0.5, 0.58, 0.51, 0.58]
>>> r.split.shape
(112, 112, 4)
>>> list(attention_split.over(pop).columns[-1:])
['lopsided']

```

So on a typical input each head puts three quarters or more of its
attention on one side, and which side depends on the input.

Bump `version` whenever the function's output changes, so stale cached
results are not reused. A measure of a whole run's training record, rather
than one checkpoint, uses `@run_measure` and receives the `Run`.

## Next

[Measures](measures.md) documents the built-in measures. [Library](library.md)
covers the primitives a measure is written with: grids, activation caches,
character bases, hooks and ablations.
