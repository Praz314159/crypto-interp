# Library

How `crypto_interp` is organised, what a run looks like on disk, and the
primitives that measures are written with.

## Layers

Each subpackage may import only from those below it, and
`tests/test_layering.py` enforces this:

```text
cli                      the crypto-interp command; thin, imports everything
figures  → measures      plots from measure tables, never from models
viz      → interp, measures, runs
measures → interp, runs  named, versioned, cached analyses
interp   → models, data, runs
training → models, data, runs
runs     → models, data  checkpoints, manifests, the run index
data     → reference     tasks as datasets
models, reference, theory   no internal imports
```

The split that matters most is that figures only see tables. Anything a plot
shows is a measure first, so it is cached, versioned, testable and available
to `measure --out` as data.

## Runs on disk

```text
experiments/003_dmodel_sweep_p113/
  config.py                       CONFIG = ExperimentConfig(...)
  runs/dmodel_24_dmlp_32_seed1/
    checkpoint_000000.pt …        weights + task + model and training config
    losses.pt                     loss history
    metrics.pt                    per-character embedding energy during training
    manifest.json                 task, configs, provenance; optional "excluded"
    measures/<name>/v<version>/   cached measure results (.json scalars, .npz arrays)
```

**Checkpoints** (schema 2) are self-describing. `runs.load_checkpoint(path)`
returns the model weights, the `TaskSpec`, the model and training configs,
the epoch and the optimizer state. `interp.load_run(path)` goes on to
rebuild the model and its dataset, which is regenerated from the `TaskSpec`
rather than stored.

**Manifests** describe a run without loading a checkpoint, which keeps the
run index fast. A manifest with `"excluded": "<reason>"` hides the run from
queries. That is how runs with a known problem are kept on disk but out of
results. `crypto-interp runs rebuild-manifests DIR…` regenerates manifests
from checkpoints.

**The run index** (`runs.query`, `runs.discover`, `RunSet`, `Run`) finds
`<root>/<experiment>/runs/<run>` directories and filters them by config. See
the [tutorial](tutorial.md#4-select-runs).

## Session

A `Session` is one model plus what every analysis needs: its dataset, the
character basis and index for p, and lazily computed forward passes that
are shared between measures.

```pycon
>>> from crypto_interp.interp import Session
>>> S = Session.from_run("tests/fixtures/mul_p113_d24_m32_seed1")
>>> S.ds.p, S.ds.n_train, S.ds.n_test
(113, 3830, 8939)
>>> tuple(S.logits_grid.shape)  # every (a, b) of units, final position, all tokens
(112, 112, 114)
>>> tuple(S.cache["mlp_post"].shape)  # every (a, b) of 0..p−1, all positions, all neurons
(12769, 3, 32)

```

There are two grids. The full grid covers all p² value-token pairs, 0
included; the activation cache uses it. The group grid covers the n² pairs of
the task's cyclic group, the domain of the characters, which `logits_grid` and
most measures use: for multiplication mod p it is the (p−1)² pairs with
a, b ≠ 0, and for addition mod n it is the whole full grid.
`S.cache.grid(name)` reshapes a cached activation onto the full grid, and
`S.cache.grid(name, group=True)` onto the group grid.
`crypto_interp.interp.grids` has the conversions.

Activation names follow TransformerLens (`blocks.0.mlp.hook_post`), and short
names work too: `mlp_post`, `mlp_out`, `attn_attn` (the attention pattern),
`resid_mid` and so on.

## The group and its characters

Every group task computes the operation of a cyclic group, and its dataset
carries that group as `ds.structure`, a `CyclicStructure`: the order n, the
tokens that are group elements, and each one's coordinate x in ℤ/n. For
addition mod n, token t has coordinate t. For multiplication mod p, token
t ≠ 0 has coordinate dlog(t), so n = p − 1. Multiplication mod p is addition
mod p − 1 with the tokens renamed, and every analysis is written against the
structure, so it applies to both:

```pycon
>>> from crypto_interp.data import TaskSpec
>>> mul, add = TaskSpec("mul", 113).structure(), TaskSpec("add", 112).structure()
>>> (mul.n, mul.generator, mul.tokens[:3], mul.coords[:3]), (add.n, add.coords[:3])
((112, 3, (1, 2, 3), (0, 12, 1)), (112, (0, 1, 2)))
>>> mul.order(56), add.order(56)  # χ_56 has order 2 in both: the same group
(2, 2)

```

`structure.char_basis()` returns the real character basis, the Fourier basis
of ℤ/n pulled back to the group tokens, with a cosine and a sine row per
character, plus a delta row for each value token outside the group (token 0
for multiplication). It comes with a `CharIndex` from k to its rows
(`ci.cos[k]`, `ci.sin[k]`). A Session holds these as `S.basis` and `S.ci`.
`interp.char_index(p)` and `interp.discrete_log_table(p)` remain for
multiplication mod p, and `interp.project_1d` and `project_2d` project
activations onto characters.

## Interventions

Weight edits are context managers that restore the model on exit.
`ablate_char_w` removes one character from the embedding. Here, removing χ10
destroys the model:

```pycon
>>> from crypto_interp.interp import ablate_char_w, patch_mlp_out
>>> f"{S.evaluate()[1]:.0e}"  # test loss
'2e-04'
>>> with ablate_char_w(S.model, 10, S.basis, S.ci):
...     print(round(S.evaluate()[1]))
30

```

Activation patching then locates where the lost information is needed. Run
the model with χ10 ablated, patch in the clean MLP output at one position at
a time, and record the test loss:

```pycon
>>> logits, clean = S.run_with_cache(S.ds.inputs)
>>> with ablate_char_w(S.model, 10, S.basis, S.ci):
...     loss = patch_mlp_out(S.model, S.ds.inputs, clean, S.test_loss_metric)
>>> [f"{x:.0e}" for x in loss.tolist()]  # patched at a, b, '='
['3e+01', '3e+01', '2e-05']

```

Patching at `=` restores everything, and patching at a or b restores nothing.
The MLP computes the product at the `=` position, from what attention moved
there. `patch_attn_out` and `patch_resid_pre` patch other sites, and
`act_patch` patches any hook. For arbitrary edits during a forward pass,
`hooks(model, [(name, fn), …])` attaches hook functions, such as
`zero_hook()` and `patch_hook(...)`, for the duration of a `with` block.

## Adding a task

A task is a function `build(p, frac_train, seed) -> Dataset` in
`crypto_interp/data/`, registered in `crypto_interp.data._TASKS`.
`data/add.py` is the smallest example. A task that computes a cyclic group's
operation sets the dataset's `structure` (and is listed in `GROUP_TASKS`), and
then every measure applies to it. For any other task, construct a `Session`
with `basis=` and `ci=` for its group.
