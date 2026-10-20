"""Dataset registry. Add new tasks by creating a module with a ``build`` fn."""

from . import add, mul  # avoid shadowing math.sqrt elsewhere
from . import sqrt as sqrt_mod
from .base import Dataset
from .spec import TaskSpec
from .structure import CharIndex, CyclicStructure

_TASKS = {
    "mul": mul.build,
    "mul_units": mul.build_units,
    "add": add.build,
    "sqrt_of_product": sqrt_mod.build,
}

#: Tasks on the multiplicative group (Z/p)*.
MULTIPLICATIVE_TASKS = frozenset({"mul", "mul_units"})

#: Tasks that compute a cyclic group's operation, and so have a CyclicStructure.
GROUP_TASKS = MULTIPLICATIVE_TASKS | {"add"}


def group_order(task: str, p: int) -> int | None:
    """The order n of ``task``'s cyclic group at modulus ``p`` (None for other tasks)."""
    if task in MULTIPLICATIVE_TASKS:
        return p - 1
    return p if task == "add" else None


def structure(task: str, p: int) -> CyclicStructure:
    """The cyclic group ``task`` is defined on, for modulus ``p``."""
    if task in MULTIPLICATIVE_TASKS:
        return CyclicStructure.multiplicative(p)
    if task == "add":
        return CyclicStructure.additive(p)
    raise ValueError(f"task {task!r} is not a cyclic group task; known: {sorted(GROUP_TASKS)}")


def build(task: str, **kwargs) -> Dataset:
    if task not in _TASKS:
        raise ValueError(f"Unknown task '{task}'. Known: {sorted(_TASKS)}")
    return _TASKS[task](**kwargs)


def available_tasks() -> list[str]:
    return sorted(_TASKS)


__all__ = [
    "CharIndex",
    "CyclicStructure",
    "Dataset",
    "GROUP_TASKS",
    "MULTIPLICATIVE_TASKS",
    "group_order",
    "structure",
    "TaskSpec",
    "build",
    "available_tasks",
]
