"""Dataset registry. Add new tasks by creating a module with a ``build`` fn."""

from . import add, mul  # avoid shadowing math.sqrt elsewhere
from . import sqrt as sqrt_mod
from .base import Dataset
from .spec import TaskSpec

_TASKS = {
    "mul": mul.build,
    "mul_units": mul.build_units,
    "add": add.build,
    "sqrt_of_product": sqrt_mod.build,
}

#: Tasks on the multiplicative group, analysed in its character basis.
MULTIPLICATIVE_TASKS = frozenset({"mul", "mul_units"})


def build(task: str, **kwargs) -> Dataset:
    if task not in _TASKS:
        raise ValueError(f"Unknown task '{task}'. Known: {sorted(_TASKS)}")
    return _TASKS[task](**kwargs)


def available_tasks() -> list[str]:
    return sorted(_TASKS)


__all__ = [
    "Dataset",
    "MULTIPLICATIVE_TASKS",
    "TaskSpec",
    "build",
    "available_tasks",
]
