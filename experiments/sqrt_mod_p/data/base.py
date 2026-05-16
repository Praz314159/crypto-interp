"""Shared dataset dataclass and split logic for all modular-arithmetic tasks.

Each per-task module (e.g. data/mul.py) implements `build(p, frac_train, seed)`
returning a `Dataset`. The model and training loop are task-agnostic and use
the `Dataset` interface uniformly.
"""

import random
from dataclasses import dataclass

import torch


@dataclass
class Dataset:
    p: int
    task: str
    inputs: torch.Tensor       # (N, 3) long
    labels: torch.Tensor       # (N,) long
    train_mask: torch.Tensor   # (N,) bool
    test_mask: torch.Tensor    # (N,) bool
    vocab_size: int
    n_answer_tokens: int       # how many indices logits[:, -1, :n_answer_tokens] covers
    eq_token: int

    @property
    def n_train(self) -> int:
        return int(self.train_mask.sum())

    @property
    def n_test(self) -> int:
        return int(self.test_mask.sum())


def assemble(
    p: int,
    task: str,
    pairs: list[tuple[int, int]],
    labels: list[int],
    vocab_size: int,
    n_answer_tokens: int,
    eq_token: int,
    frac_train: float,
    seed: int,
) -> Dataset:
    """Common assembly: tokenize [a, b, =], make a random train/test split."""
    n = len(pairs)
    rng = random.Random(seed)
    indices = list(range(n))
    rng.shuffle(indices)
    train_idx = set(indices[: int(frac_train * n)])

    inputs = torch.zeros(n, 3, dtype=torch.long)
    labels_t = torch.zeros(n, dtype=torch.long)
    train_mask = torch.zeros(n, dtype=torch.bool)
    for i, ((a, b), y) in enumerate(zip(pairs, labels)):
        inputs[i, 0] = a
        inputs[i, 1] = b
        inputs[i, 2] = eq_token
        labels_t[i] = y
        if i in train_idx:
            train_mask[i] = True
    test_mask = ~train_mask

    return Dataset(
        p=p, task=task, inputs=inputs, labels=labels_t,
        train_mask=train_mask, test_mask=test_mask,
        vocab_size=vocab_size, n_answer_tokens=n_answer_tokens, eq_token=eq_token,
    )
