"""Modular multiplication: (a, b) -> (a * b) mod p.

Tokenization mirrors Nanda's modular-addition setup exactly:
  - Tokens 0..p-1 are numerical values.
  - Token p is the '=' separator.
  - vocab_size = p + 1, n_answer_tokens = p.

``mul`` uses every pair in {0..p-1}². ``mul_units`` uses only the units,
{1..p-1}², the multiplicative group itself (the setup of Nguyen 2026, "The
Discrete Log Clock"). It keeps the same tokens, so token 0 never appears.
"""

from .base import Dataset, assemble


def build(p: int, frac_train: float = 0.3, seed: int = 0) -> Dataset:
    return _build(p, frac_train, seed, task="mul", values=range(p))


def build_units(p: int, frac_train: float = 0.3, seed: int = 0) -> Dataset:
    return _build(p, frac_train, seed, task="mul_units", values=range(1, p))


def _build(p: int, frac_train: float, seed: int, task: str, values: range) -> Dataset:
    pairs = [(a, b) for a in values for b in values]
    labels = [(a * b) % p for a, b in pairs]
    return assemble(
        p=p,
        task=task,
        pairs=pairs,
        labels=labels,
        vocab_size=p + 1,
        n_answer_tokens=p,
        eq_token=p,
        frac_train=frac_train,
        seed=seed,
    )


if __name__ == "__main__":
    ds = build(p=113, frac_train=0.3, seed=0)
    print(f"task=mul, p=113, vocab={ds.vocab_size}, n_answer={ds.n_answer_tokens}")
    print(f"Total {len(ds.inputs)}, train {ds.n_train}, test {ds.n_test}")
    a, b, _ = ds.inputs[5].tolist()
    print(f"Sample: ({a}, {b}, =) -> {ds.labels[5].item()};  ({a}*{b}) mod 113 = {(a * b) % 113}")
