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

Square root mod p (Thread 1, single-token regime). First milestone: train a
small transformer to grok square root mod a small prime p, then compare the
learned circuit to Tonelli–Shanks and Cipolla.
