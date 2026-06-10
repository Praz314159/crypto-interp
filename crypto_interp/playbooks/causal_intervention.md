# Causal intervention — weight ablation + activation patching

**Strategy 3.** Move from "X correlates with Y" to "X *causes* Y" by
intervening. Two complementary flavors:

- **Weight-space ablation.** Remove a character's components from W_E
  (treat as a missing input feature). Tests "does the input direction
  matter."
- **Activation-space patching.** Replace an intermediate activation with
  one from a different forward pass. Tests "does this internal signal
  carry the information at this point in the forward pass."

Both are causal. Use both together — they probe different places.

## Algebraic anchor

Removing a structural element (a basis vector from the embedding, or an
activation at a specific position/component) maps to "project out a basis
element" — a classical algebraic operation. The mech-interp question is
*which* element matters and *where*.

## Verbs

- `with ablate_char_w(model, k, basis, ci): …` — weight-space context
  manager; restores W_E on exit.
- `with weight_patch(model, param_name, new_value): …` — generic
  weight-space patch (any parameter).
- `act_patch(model, corrupted_inputs, clean_cache, hook_name, metric_fn, positions=None)`
  — activation-space causal probe. Per-position by default; `positions="all"`
  patches the entire activation at once.
- `patch_mlp_out` / `patch_resid_pre` / `patch_attn_out` — thin wrappers
  fixing the hook name.
- `S.test_loss_metric(logits)` — canonical metric (test-set CE at the
  final answer position).

## Recipe — weight-space: which characters are causally essential?

```python
from crypto_interp.interp import Session, ablate_char_w
from crypto_interp.interp.ablate import evaluate_loss

S = Session.from_run("experiments/003_dmodel_sweep_p113/runs/dmodel_24_dmlp_20_wd2_seed1")
base_test = S.evaluate()[1]
for k in S.essential()["K"]:
    with ablate_char_w(S.model, k, S.basis, S.ci):
        ablated_test = evaluate_loss(S.model, S.ds)[1]
    print(f"χ{k}: test {base_test:.1e} → {ablated_test:.1e}  (Δlog10 {math.log10(ablated_test/base_test):+.2f})")
```

## Recipe — activation-space: where does the signal live?

```python
from crypto_interp.interp import Session, ablate_char_w, patch_mlp_out

S = Session.from_run(".../seed1")
# Clean run = full model. Corrupt run = with χ_10 ablated from W_E.
_, clean_cache = S.run_with_cache(S.ds.inputs)
clean_metric = S.test_loss_metric(_)

with ablate_char_w(S.model, k=10, basis=S.basis, ci=S.ci):
    # Patch the clean mlp_out into the corrupted forward, per position.
    per_pos = patch_mlp_out(S.model, S.ds.inputs, clean_cache, S.test_loss_metric)
print(f"per-position mlp_out patch CE: {[float(x) for x in per_pos]}")
# Expected: pos 0,1 ≈ corrupt CE; pos 2 ≈ clean CE — info is at '=' token.
```

## Common variations

- **Sweep hook names.** Combine `patch_mlp_out`, `patch_resid_pre`,
  `patch_attn_out` to localize which component of the block carries the
  signal.
- **Define a logit-difference metric.** For tasks with a known answer,
  `metric = lambda lg: (lg[..., correct] - lg[..., distractor]).mean()` is
  often more informative than CE.
- **Patch over a range of positions.** `act_patch(..., positions=[1, 2])`
  patches both b- and =-token in one call.
- **Compose with weight-space.** `with ablate_char_w(k=10): act_patch(...)`
  — patch an activation under a partial-W_E corruption to test joint
  necessity.
