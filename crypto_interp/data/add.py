"""Modular addition: (a, b) -> (a + b) mod n, the cyclic group Z/n.

The Nanda baseline. Tokenization mirrors ``mul``:
  - Tokens 0..n-1 are numerical values.
  - Token n is the '=' separator.
  - vocab_size = n + 1, n_answer_tokens = n.

The modulus is the config field ``p``, but it need not be prime: any n gives
the cyclic group of that order. Multiplication mod a prime p is this task for
n = p − 1, with the tokens renamed by the discrete log (see
:mod:`crypto_interp.data.structure`).
"""

from .base import Dataset, assemble
from .structure import CyclicStructure


def build(p: int, frac_train: float = 0.3, seed: int = 0) -> Dataset:
    pairs = [(a, b) for a in range(p) for b in range(p)]
    labels = [(a + b) % p for a, b in pairs]
    return assemble(
        p=p,
        task="add",
        pairs=pairs,
        labels=labels,
        vocab_size=p + 1,
        n_answer_tokens=p,
        eq_token=p,
        frac_train=frac_train,
        seed=seed,
        structure=CyclicStructure.additive(p),
    )


if __name__ == "__main__":
    ds = build(p=113, frac_train=0.3, seed=0)
    print(f"task=add, p=113, vocab={ds.vocab_size}, n_answer={ds.n_answer_tokens}")
    print(f"Total {len(ds.inputs)}, train {ds.n_train}, test {ds.n_test}")
    a, b, _ = ds.inputs[5].tolist()
    print(f"Sample: ({a}, {b}, =) -> {ds.labels[5].item()};  ({a}+{b}) mod 113 = {(a + b) % 113}")
