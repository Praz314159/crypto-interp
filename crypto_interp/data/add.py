"""Modular addition: (a, b) -> (a + b) mod p.

The Nanda baseline. Tokenization mirrors ``mul``:
  - Tokens 0..p-1 are numerical values.
  - Token p is the '=' separator.
  - vocab_size = p + 1, n_answer_tokens = p.

Same architecture and tokenization as ``mul`` — only the label function
differs. This is the minimal extension that adds a second task and unlocks
the lattice-variation experimental program (basis dispatch by task).
"""

from .base import Dataset, assemble


def build(p: int, frac_train: float = 0.3, seed: int = 0) -> Dataset:
    pairs = [(a, b) for a in range(p) for b in range(p)]
    labels = [(a + b) % p for a, b in pairs]
    return assemble(
        p=p, task="add",
        pairs=pairs, labels=labels,
        vocab_size=p + 1, n_answer_tokens=p, eq_token=p,
        frac_train=frac_train, seed=seed,
    )


if __name__ == "__main__":
    ds = build(p=113, frac_train=0.3, seed=0)
    print(f"task=add, p=113, vocab={ds.vocab_size}, n_answer={ds.n_answer_tokens}")
    print(f"Total {len(ds.inputs)}, train {ds.n_train}, test {ds.n_test}")
    a, b, _ = ds.inputs[5].tolist()
    print(f"Sample: ({a}, {b}, =) -> {ds.labels[5].item()};  ({a}+{b}) mod 113 = {(a+b)%113}")
