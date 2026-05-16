"""Modular multiplication: (a, b) -> (a * b) mod p.

Tokenization mirrors Nanda's modular-addition setup exactly:
  - Tokens 0..p-1 are numerical values.
  - Token p is the '=' separator.
  - vocab_size = p + 1, n_answer_tokens = p.
"""

from .base import Dataset, assemble


def build(p: int, frac_train: float = 0.3, seed: int = 0) -> Dataset:
    pairs = [(a, b) for a in range(p) for b in range(p)]
    labels = [(a * b) % p for a, b in pairs]
    return assemble(
        p=p, task="mul",
        pairs=pairs, labels=labels,
        vocab_size=p + 1, n_answer_tokens=p, eq_token=p,
        frac_train=frac_train, seed=seed,
    )


if __name__ == "__main__":
    ds = build(p=113, frac_train=0.3, seed=0)
    print(f"task=mul, p=113, vocab={ds.vocab_size}, n_answer={ds.n_answer_tokens}")
    print(f"Total {len(ds.inputs)}, train {ds.n_train}, test {ds.n_test}")
    a, b, _ = ds.inputs[5].tolist()
    print(f"Sample: ({a}, {b}, =) -> {ds.labels[5].item()};  ({a}*{b}) mod 113 = {(a*b)%113}")
