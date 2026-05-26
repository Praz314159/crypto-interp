"""Dataset registry. Add new tasks by creating a module with a ``build`` fn."""

from .base import Dataset
from . import mul, add, sqrt as sqrt_mod  # avoid shadowing math.sqrt elsewhere
from .io import load_or_build, save, load, cache_path

_TASKS = {
    "mul": mul.build,
    "add": add.build,
    "sqrt_of_product": sqrt_mod.build,
}


def build(task: str, **kwargs) -> Dataset:
    if task not in _TASKS:
        raise ValueError(f"Unknown task '{task}'. Known: {sorted(_TASKS)}")
    return _TASKS[task](**kwargs)


def available_tasks() -> list[str]:
    return sorted(_TASKS)


__all__ = [
    "Dataset",
    "build",
    "available_tasks",
    "load_or_build",
    "save",
    "load",
    "cache_path",
]
