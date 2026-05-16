"""Sqrt of product: (a, b) -> canonical_sqrt(a*b mod p) if QR else NoRoot.

Tokenization:
  - Tokens 0..p-1 are numerical sqrt values.
  - Token p is the NoRoot answer (signal that a*b is not a QR).
  - Token p+1 is the '=' separator.
  - vocab_size = p + 2, n_answer_tokens = p + 1.
"""

from reference import is_quadratic_residue, tonelli_shanks

from .base import Dataset, assemble


def build(p: int, frac_train: float = 0.3, seed: int = 0) -> Dataset:
    no_root = p
    eq = p + 1

    pairs = [(a, b) for a in range(p) for b in range(p)]
    labels = []
    for a, b in pairs:
        prod = (a * b) % p
        if is_quadratic_residue(prod, p):
            r = tonelli_shanks(prod, p)
            assert r is not None
            labels.append(r)
        else:
            labels.append(no_root)

    return assemble(
        p=p, task="sqrt_of_product",
        pairs=pairs, labels=labels,
        vocab_size=p + 2, n_answer_tokens=p + 1, eq_token=eq,
        frac_train=frac_train, seed=seed,
    )


if __name__ == "__main__":
    ds = build(p=113, frac_train=0.3, seed=0)
    no_root = ds.p
    n_no_root = (ds.labels == no_root).sum().item()
    print(f"task=sqrt_of_product, p=113, vocab={ds.vocab_size}, n_answer={ds.n_answer_tokens}")
    print(f"Total {len(ds.inputs)}, train {ds.n_train}, test {ds.n_test}")
    print(f"NoRoot labels: {n_no_root} ({100*n_no_root/len(ds.labels):.1f}%)")
