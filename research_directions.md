# Research Directions: Mechanistic Interpretability of Cryptographic Algorithms

## Project framing

Mechanistic interpretability of cryptographically-important algorithms.
Inspired by Nanda et al. (ICLR 2023), "Progress Measures for Grokking via
Mechanistic Interpretability" (`papers/grokking.pdf`), which reverse-engineered
a one-layer transformer trained on modular addition and found it had
rediscovered a DFT + trigonometric-identity algorithm.

**Central question.** When ML models learn cryptographic primitives, do they
converge on the algorithms cryptographers have designed, find variants, or
invent something different? Each outcome is informative:

- *Convergence* validates the cryptographer's algorithm as the "natural" one.
- *Variant* suggests the same algorithmic skeleton with different
  factorizations or constants.
- *Divergence* points to potentially unexplored algorithmic territory.

---

## Thread 1 — Replication / algorithm-choice (primary first step)

Extend Nanda-style interpretability to modular arithmetic operations beyond
addition. The methodology (Fourier-basis activation analysis, frequency
ablations, restricted/excluded loss progress measures) is the template; we
adapt it to richer operations and compare what the model learns to what
cryptographers use.

**Candidate operations (single-token regime, small prime p):**

- **Square root mod p** — *first concrete experiment.* Two well-known
  cryptographer algorithms: Tonelli–Shanks and Cipolla. Having two reference
  algorithms makes the comparison particularly clean: does the model match
  one, the other, both depending on hyperparameters, or neither?
- **Modular multiplication mod p** — closest analogue of Nanda's setup; tests
  whether the DFT trick extends from addition to multiplication.
- **Legendre symbol / quadratic residue testing** — small output space (±1, 0)
  but rich algebraic structure; cryptographer's algorithm uses Euler's
  criterion or quadratic reciprocity.
- **Primality testing** — particularly rich landscape with multiple reference
  algorithms (Fermat, Solovay–Strassen, Miller–Rabin, Baillie–PSW, AKS) and
  a structural feature unique to this task: cryptographer's standard
  algorithms are *probabilistic* (witness-based), but a feedforward model is
  deterministic at inference. The natural prediction is that the model will
  converge on a witness-based circuit with a *fixed deterministic witness
  set* baked into weights — equivalent to deterministic Miller–Rabin. The
  empirical question becomes: **which witnesses does it pick?** Cryptographers
  have minimal deterministic witness sets (e.g., {2,3,5,7,11,13,17,19,23,29,31,37}
  is deterministic for n < 3.3 × 10¹⁴) — does the model rediscover these,
  pick a subset, or find different witnesses tuned to its training
  distribution? Three increasingly-informative experimental setups:
  *(a)* binary classification (prime / composite); *(b)* output a Miller–Rabin
  witness for composites — no memorization shortcut, forces the model to
  compute the witness; *(c)* output a primality certificate. Distribution
  sensitivity is also a clean experiment: include vs. exclude Carmichael
  numbers and check whether the model learns the strong (MR-style) check or
  settles for the weak (Fermat-style) one. Note that primality testing
  transitively requires modular exponentiation, so it provides a setting
  where mod-exp must be learned as a subroutine rather than studied in
  isolation.

**Deliverable.** For each operation: a fully reverse-engineered circuit and a
direct comparison to the cryptographer's reference algorithm(s).

---

## Thread 2 — Data complexity and minimal sufficient training sets

How much training data is required to drive convergence to the canonical
algorithm, and is there algebraic structure to the *minimal* sufficient
training set?

Best piloted on plain modular addition, where the ground-truth algorithm (DFT)
is already known from Nanda et al. Methodology developed there can transfer to
the operations in Thread 1.

**Sub-experiments:**

1. **Structured-subset phase diagram.** Reproduce Power et al. / Liu et al.
   data-fraction × weight-decay phase diagram, but with structured (not
   random) data subsets.
2. **Sparse-recovery scaling.** If the converged solution uses *s* key
   frequencies, does data complexity scale like *O(s · P log P)* rather than
   *O(P²)* — i.e., consistent with sparse signal recovery?
3. **Generating-set training.** Train only on (gⁱ, gʲ) for a generator *g*.
   Does the model still grok? If yes, data complexity scales with |⟨g⟩|·log P,
   not P².
4. **Influence-function reverse analysis.** Take a fully-grokked model; rank
   training samples by contribution to the DFT circuit; check whether
   high-influence samples have algebraic structure (orbits, subgroups,
   uniform coverage).
5. **Constructive curriculum.** Try to *design* a minimal training set that
   forces convergence to a specified algorithm. Success would identify a
   principle of dataset construction for algorithmic learning.

---

## Thread 3 — Representation regime: single-token vs. limb

A field of size 2ⁿ can be represented as bits (vocab=2), bytes (vocab=256),
or limbs (vocab=2³²). Vocabulary scales with the representation *base*, not
the field size. Field size determines sequence length.

This matters because **representation choice changes which algorithm the model
converges to.** Existing work on multi-digit integer arithmetic (Quirke 2023,
Lee et al.) finds that small transformers in the multi-token regime learn
*digit-wise / carry-circuit* algorithms — not the Fourier algorithm Nanda
found in the single-token regime.

**Two regimes, two purposes:**

- *Single-token regime* (small fields, one token per element): cleanest
  methodology. Right home for Threads 1 and 2.
- *Limb regime* (multi-token representations of cryptographic-sized fields):
  the regime where Montgomery reduction, Barrett reduction, Karatsuba,
  Toom–Cook, and other cryptographer tricks *actually live* — they only make
  sense when numbers are stored as multi-limb sequences. Likely where this
  project's most distinctive contribution sits, but harder: no canonical basis
  for activation analysis, more layers needed for sequential operations like
  carry propagation, more compute.

**Eventual goal in this thread:** train a model on limb-represented modular
multiplication for cryptographic-sized *p* and check whether it rediscovers
Montgomery- or Barrett-style reduction.

---

## Thread 5 — Research-agent harness (parallel infrastructure)

The methodology Nanda used is *almost* algorithmic: train → cache activations
→ project into a basis → look for sparsity → ablate to validate → if
unsuccessful, try another basis. A small number of decision points are where
human judgment enters. This is plausible territory for a research agent.

Three tiers of ambition:

- **Tier 1 — Recipe runner.** Agent takes a fully-specified experiment
  (primitive, prime, hyperparameters, basis) and runs the Nanda pipeline
  end-to-end, producing a structured report. No agentic decisions; mostly
  engineering. The substrate everything else builds on.
- **Tier 2 — Hypothesis loop.** Agent maintains a small set of hypotheses
  about the model's algorithm and a few primitive moves (project, ablate,
  re-evaluate). Updates confidence over hypotheses; terminates when one is
  well-supported or all are exhausted. Open design question: what space of
  bases is the agent allowed to try?
- **Tier 3 — Methodology-discovering agent.** When no hypothesized basis
  works, the agent designs new analysis tools. Several iterations away.

**Methodology for designing the harness:** instrument the hand-done
experiments (Threads 1–2) so every decision and its reasoning is recorded.
The transcript is the specification for what an agent would need to automate.
Concretely: weight ablation evidence over visual sparsity; avoid retraining
where possible by caching aggressively; bound the basis space the agent can
search before designing more sophisticated discovery.

Without a harness, applying interp methodology to the ~15 primitives in
Thread 4 is years of work. With even Tier 1 working, it becomes feasible.

---

## Thread 4 — Broader algorithmic landscape (longer-term)

Beyond modular arithmetic. Tagged by tractability:
🟢 tractable now, 🟡 plausible with effort, 🔴 aspirational / structural mismatch.

**Number-theoretic primitives**
- 🟢 Modular square root mod p (Thread 1)
- 🟢 Quadratic residue / Legendre symbol (Thread 1)
- 🟢 Discrete log in small cyclic groups (does the model find baby-step-giant-step? Pohlig–Hellman?)
- 🟢 CRT reconstruction
- 🟡 Modular exponentiation (square-and-multiply — needs depth)
- 🟡 Pollard rho factoring or discrete log (cycle detection — open question whether feedforward nets can learn it)
- 🟢/🟡 Primality testing (now promoted to Thread 1 — see above)

**Symmetric / finite-field**
- 🟢 GF(2ⁿ) multiplication for small n (log/antilog tables vs. carry-less mul vs. tower fields)
- 🟡 AES S-box (defined as inversion in GF(2⁸) plus an affine map — does the model factor it that way?)
- 🟡 ChaCha quarter-round
- 🔴 Full SHA-256 / AES round (mostly memorization at this scale)

**Lattice-based**
- 🟢 NTT / polynomial multiplication in small Z_q[x]/(xⁿ+1)
- 🟢 2D Gauss lattice reduction (n=2 case of LLL — clean entry point)
- 🟡 LLL — structural challenges: real-valued matrices, iterative with
  variable step count, Gram–Schmidt. Probably needs a recurrent / looped
  transformer to study honestly.
- 🟡 Babai's nearest-plane / rounding
- 🟡 Discrete Gaussian sampling

**Code-based**
- 🟢 Berlekamp–Massey for LFSR synthesis on short sequences over GF(q) — strong candidate; foundational for Reed–Solomon decoding; has Euclidean-algorithm variant for two-algorithm comparison
- 🟢 Reed–Solomon syndrome computation
- 🟡 Berlekamp–Welch decoding
- 🟡 Patterson's algorithm (Goppa decoding for McEliece)
- 🔴 Information-set decoding (Prange / Stern / BJMM — combinatorial search, wrong shape for transformer)

**Isogeny-based**
- 🟢 Scalar multiplication on a small elliptic curve over a small field (does the model find double-and-add? Montgomery ladder? windowed exponentiation?)
- 🟢 Vélu's formulas for small-degree isogenies on a small curve
- 🟡 Pairing computation via Miller's algorithm
- 🟡 CSIDH group action evaluation
- 🔴 SIDH walks / Kani's lemma

**Hash-based / signatures**
- 🟢 Merkle authentication path verification
- 🟢 Lamport / Winternitz one-time-signature verification

**Cross-cutting**
- 🟢 Lagrange interpolation / Shamir secret-sharing reconstruction
- 🟡 Constant-time vs. variable-time training pressure as a side-channel angle (looped architecture forced to do equal work per input). Genuinely novel framing.

---

## Constraints clarified

We initially framed interpretability as limited to "small moduli". The real
constraints are more specific:

1. **Tokenization choice**, not field size, sets the embedding-matrix size.
   Single-token framings cap out around |F| ≈ 10⁵–10⁶ on current hardware;
   multi-token framings have no such cap.
2. **Interpretability methodology** historically uses exhaustive activation
   probing (every (a, b) pair). This scales beyond enumeration only with
   sparse-recovery / sampling-based extensions, which require committing to a
   hypothesized basis.
3. **Algorithm relevance.** Some cryptographer tricks (Montgomery, Barrett,
   Karatsuba) only pay off at limb scale. They cannot appear in the
   single-token regime regardless of how well we interp it.

---

## Immediate next step

*(superseded — see "Updated immediate next step" at the bottom of the
2026-05-25 update.)*

Square root mod p (Thread 1, single-token regime). First milestone: train a
small transformer to grok square root mod a small prime p, then compare the
learned circuit to Tonelli–Shanks and Cipolla.

---

## 2026-05-25 — Update: paper consolidation and new research threads

The work has consolidated around modular **multiplication** as the focal task,
with a paper in preparation. This section captures the current state and the
threads not yet executed.

### Paper status

- **Scope:** one paper, top main-track (ICLR/NeurIPS), spine = **mechanism +
  economy** in modular multiplication; dynamics/de-grokking/methods as
  appendices.
- **Act I — the learned algorithm:** the multiplicative-Fourier / discrete-log
  algorithm; Pontryagin duality realized by GD; **the Legendre channel**
  (minimal 9-neuron order-2 circuit) as the showcase of multiplicative-specific
  structure (`notes/02_legendre_channel.tex`).
- **Act II — the capacity-constrained economy:** the empirical **cost atom**
  (neurons ≈ budget/#primaries, order-2 ≈ 6.6× cheaper, doublings cost 0);
  the **doubling helper as a free-rider** (ReLU 2nd harmonic at a₂=2/3π,
  a₃=0; helpers free on the neuron/output side, paid on the embedding/input
  side per the sufficiency test); ≈3-doubling-pair-cluster backbone across 21
  grokked seeds; cross-prime generalization (p=113/127/181 — in progress).
- **Contribution boundary (post Gromov scoop-check):** Gromov (2023) owns the
  harmonic fact (a squaring nonlinearity produces χ_{2k}); the
  **economy/allocation framing** in the *multiplicative* setting is ours —
  what a trained, budget-limited net spontaneously does with the harmonic
  (free-rider on shared clusters, load-bearing, the building block of
  capacity economization). Lead with allocation, cite Gromov as substrate.
- **Related-work map:** Nanda 2023 (additive Fourier basis methodology);
  Chughtai/Chan/Nanda 2023 (characters/irreps-as-features framework);
  Zhong et al. *Clock & Pizza* (nonlinearity-selects-algorithm — qualitative
  precedent to our quantitative spectral law); Marchetti et al. 2026
  (sequential group composition with polynomial activations — depth-as-
  associativity but not iterated squaring); Wu et al. ICLR 2025 (formal-
  verification standard — pre-empt with a compact proof of the Legendre
  circuit); McCracken et al. *aCRT* NeurIPS 2025 (additive CRT; opens the
  neurons-per-frequency question we answer); Mohamadi et al. (margin theory
  of grokking *onset* — our de-grokking is a counterexample to margin-
  monotonicity, outside their no-WD assumption); Gromov 2023 (analytic
  single-hidden-layer; explicitly leaves depth open).
- **Pending manuscript work:** reconcile `notes/` with this session's
  corrections — the Pareto frontier was falsified by the empirical cost atom
  (`cost_atom.png`) and is red-flagged in `notes/03,04`; `notes/06` should
  be reframed as *allocation-under-constraint* with Gromov cited as substrate;
  population numbers (21 seeds), the sufficiency result, and the de-grokking
  appendix all need folding in.

### Thread 6 — Library and methodology: TransformerLens transfer  *(v1 landed)*

**Status (2026-05-26):** harness v1 merged from branch `harness-v1`. The
gap described below is closed. See [`crypto_interp/playbooks/README.md`](crypto_interp/playbooks/README.md)
for the canonical recipes; [`crypto_interp/README.md`](crypto_interp/README.md)
is the library entry point.

What landed:

- `ActivationCache` class with short-name aliases (`mlp_post` ↔
  `blocks.0.mlp.hook_post`) and `.final` / `.grid` / `.decompose_resid`.
- `run_with_cache`, `run_with_hooks`, `hooks(...)` context manager.
- Weight-space interventions as context managers (`weight_patch`,
  `ablate_char_w`, `freeze_param`), replacing the inline try/finally
  pattern in `grids.py` / `harmonic.py` / `ablate.py`.
- Activation patching: `act_patch` + `patch_mlp_out` / `patch_resid_pre`
  / `patch_attn_out`.
- `Session` — the analysis bundle bundling `(model, ds, basis, ci)`
  with lazy `cache` and one-line passthroughs to all the domain verbs.
- Layering fix: per-neuron-cluster helpers moved from `analysis/` to
  `interp/neurons.py`; analyses now depend on interp, never the other way.
- A second task (`modular_add`) registered in the data layer, plus a
  stub `experiments/006_add_p113/config.py`.
- Seven markdown playbooks documenting the canonical recipes for each
  mech-interp strategy.
- Smoke tests (`tests/test_harness_smoke.py`); the existing 16-test
  suite still passes (20/20 total).

Deferred to v2 (after the lattice-variation experimental program):

- `Task` / `AlgebraicDomain` dataclass. The current data layer dispatches
  by string; that's fine until the lattice-variation experiments reveal
  what algebraic metadata is actually used by the analyses.
- `FactoredMatrix`. Dense math is fine at d_model=24; revisit when
  attention pathway analysis becomes load-bearing (multi-block / longer
  contexts).

**Original gap analysis kept below for historical context:**

`crypto_interp` has strong **domain primitives** (the `interp/` package —
multiplicative-Fourier bases, character indexing, essential-character
ablation, harmonic-helper detection, dynamics detectors) but **ad-hoc
scaffolding** (custom hook-less caches, no patching utilities, no factored-
matrix tools for attention analysis). TransformerLens
(`/Users/prashanth/Desktop/Research/cryptography/TransformerLens`) is the
gold-standard reference for that scaffolding.

**Abstractions to study and adopt:**

- **`HookedTransformer` + `hook_points`:** hook-at-every-internal-point
  architecture; clean separation of the model from the cache/intervention API.
- **`ActivationCache`:** cache abstraction with selector APIs (cache by name,
  by component) — replaces hand-rolled dict caches.
- **`patching.py` (activation and path patching):** the canonical *causal*
  intervention (run two models, swap an activation at a hook point, measure
  logit change). We have ad-hoc ablations (`ablate_character`,
  `ablate_embedding`) and exact Δ_k (`delta_k`); the general patching
  framework would unlock attention-head analysis, residual-stream-position
  interventions, and path patching for circuit isolation.
- **`FactoredMatrix`:** low-rank linear-algebra utilities for `W_Q @ W_K^T`
  (the QK circuit) and `W_V @ W_O` (the OV circuit) without materializing
  the full matrices — directly addresses the ad-hoc-only attention/path
  themes in `experiments/ANALYSES.md` (§8, §10).
- **`SVDInterpreter.py`, `head_detector.py`:** rank-decomposition and
  attention-pattern classifiers; templates we lack.

**Integration paths** (decision needed after exploration):
1. **Wrap:** use `HookedTransformer` directly; rewrite our cache/ablate to
   call into it. Pro: maximal technique transfer. Con: TL's model class may
   not match our grokking-friendly tiny no-LN architecture exactly.
2. **Adopt patterns:** keep our model class, port TL's API patterns
   (`HookPoint`, `ActivationCache`, `patching`) into `crypto_interp/interp`.
   Pro: preserves architectural control. Con: more porting work.

Entry points: `TransformerLens/demos/Main_Demo.ipynb`, then
`transformer_lens/patching.py` and `ActivationCache.py`.

### Thread 7 — Multilayer / depth as iterated squaring

**Hypothesis:** stacking transformer blocks lets the network *iterate* the
ReLU squaring step — block 1 produces χ_{2k} from χ_k, block 2 squares
χ_{2k} to χ_{4k} at full strength (rather than relying on a₄, which is
weak). So **max Sylow-2 doubling-chain depth ≈ number of stacked nonlinear
layers**, capped by v₂(p−1). Rehabilitates the retired Sylow-chain-depth
idea: chain depth comes from network depth, not from p−1's factorization
within a single layer.

**Experiment (depth-vs-width controlled):** matched total ReLU neurons
across (depth, width) configs at tight per-layer width — e.g., M=36 total:
(L=1, d_mlp=36), (L=2, d_mlp=18), (L=3, d_mlp=12). Measure helper-chain
depth + grok rate. Predict deeper-narrower nets exploit the tower while
shallower-wider don't.

**Blocked on:** `num_layers` plumbing fix (~3 lines in `training/loop.py`
and `interp/load.py`, plus `.get('num_layers', 1)` for backward-compat).

**Honest caveats:** depth-2 chains aren't necessary (1-layer groks), so the
model only exploits the tower if *forced* by tight per-layer width; deeper
nets are harder to interpret (more places computation can hide); added
attention rounds confound (cleanest version is MLP-only depth — invasive).

### Thread 8 — Size reduction: locating the hard grokking floor

At d_model=24, d_mlp=20, wd=2.0 the model groks ≈38% stable / ≈52%
ever-grokked. We have not found a configuration that **prevents grokking
outright**. Sweep d_mlp ∈ {16, 12, 10, 8, 6} (and/or d_model ∈ {20, 18,
16}), 10 seeds each at wd=2, locate the grok → no-grok transition.
Predicted: d_mlp ≤ 8 should fail (below the bilinear-fidelity floor for a
primary). The transition is the datum.

### Thread 9 — Activation harmonic-parity generalization (Act II supporting)

The doubling economy is a property of ReLU's harmonics (a₁=½, a₂=2/3π,
a₃=0). Sweep σ ∈ {ReLU, GELU, SiLU, tanh, cube, square} at constrained
d_mlp. Prediction (numerically grounded — see `cost_atom.png` and the
harmonic-coefficient table): even-harmonic σ (ReLU/GELU/SiLU) → χ_{2k}
doubling economy; **odd σ (tanh/cube) → χ_{3k} tripling economy**. Same
algorithm, different "free" character determined by the nonlinearity's
parity. Doubles as the second-architecture robustness axis required for
main-track.

### Thread 10 — Capacity-constrained economy as a multilayer-interpretability lever

A methodological extension of Thread 7: **the constraint is itself the
interpretability tool.** Hypothesis — starving multilayer nets forces them
into the minimal algebraic structure (Sylow tower spread across layers +
≈3-cluster backbone), making the multilayer computation legible group-
theoretically where the unconstrained version would be a mess. Tests whether
economy is just a phenomenon to study or a *lever for interpretability*.
Plausibly the strongest single methods contribution if it lands; depends on
Thread 7's outcome.

### Thread 11 — KAN as a decoupling probe

Kolmogorov–Arnold Networks replace fixed activations with learnable spline
edges → no fixed harmonic structure → the doubling economy is no longer
forced by σ. Test: does a KAN still economize (economy is an intrinsic
preference) or not (economy is a fingerprint of the fixed nonlinearity)?
Larger implementation lift; supporting result, not load-bearing.

### Thread 12 — The additive↔multiplicative bridge (next-paper-scale)

The deepest unexplored direction. Addition and multiplication live in
*different* algebraic homes — additive characters on (Z/p,+) vs
multiplicative/Dirichlet characters on ((Z/p)*,×). A task needing both
forces the model to bridge them via the **discrete log / antilog** — the
structural heart of a finite field. No paper in our collection studies this.

Tasks: minimal = **fused multiply-add `a·b + c mod p`** (the +c cannot be
done in dlog space → forces antilog); richer = polynomial evaluation /
Horner; **Euler's criterion** `a^((p−1)/2)` (ties straight back to the
Legendre result); the **Number-Theoretic Transform** (used in fast
multiplication and lattice crypto — connects to
`papers/salsa_lattice_attack.pdf`). Questions: (i) do learned + and ×
sub-circuits compose *modularly*, (ii) how is the multiplicative→additive
conversion implemented, (iii) does the economy survive across the bridge.
Aligns with the project-goals "generic engine" thread.

---

## Updated immediate next step

1. **In flight (Colab, Drive-resident with skip-resume):** cross-prime
   population sweep, n=20 at p=127 and p=181. Output completes the
   cross-prime row of Act II.
2. **Next (no compute):** TransformerLens exploration (Thread 6) — read
   `Main_Demo.ipynb` and `transformer_lens/patching.py`, then propose an
   integration plan for `crypto_interp` (Wrap vs Adopt-patterns).
3. **Queued (Colab, after cross-prime):** size-reduction sweep (Thread 8)
   and the activation-parity sweep (Thread 9); the latter needs the MLP
   activation made config-driven (~5-line change).
4. **Deferred:** manuscript reconciliation (Pareto out, cost-atom in,
   Gromov-bounded framing); multilayer / depth sweep (after `num_layers`
   plumbing fix); KAN; +/× compositional task.
