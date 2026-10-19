"""Training callbacks: measurements taken during training, without the loop
depending on what is measured.

A callback implements three methods. ``train()`` and ``resume()`` call them in
order: ``on_train_start`` once, ``on_epoch_end`` after every optimizer step,
``on_train_end`` once after the final checkpoint.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from ..data import TaskSpec


class TrainingCallback(Protocol):
    def on_train_start(self, run_dir: Path, model, task: TaskSpec, start_epoch: int) -> None:
        """``start_epoch`` is 0 for a fresh run and the first new epoch on resume."""

    def on_epoch_end(self, epoch: int, model) -> None: ...

    def on_train_end(self, run_dir: Path) -> None: ...
