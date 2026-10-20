"""Training dynamics: when a run memorized, committed to its characters, and grokked.

These are run measures (see :func:`.core.run_measure`): they read a run's
training record, not one checkpoint's weights.

- :func:`grok_status`: from the loss history. When train loss first fell below
  ``mem_thresh`` (memorization), when test loss first fell below
  ``grok_thresh`` (the cliff), and the outcome: grokked, memorized-only (the
  interesting failure), or no-fit.
- :func:`dynamics_markers`: from the per-character embedding energy that
  training recorded (``metrics.pt``, and the per-step ``fine_grained.pt``
  where present). With K the final energy support:

  * **bifurcation**: first epoch the mean energy ratio of K to non-K exceeds
    ``ratio`` times its initial value (K starts to outgrow the rest);
  * **commitment**: first epoch after which the top-|K| characters always
    contain K;
  * **cliff**: as in grok_status.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from .core import run_measure


def _loss_history(run):
    ck = run.load()
    return np.asarray(ck.train_losses, dtype=np.float64), np.asarray(
        ck.test_losses, dtype=np.float64
    )


@dataclass
class GrokStatus:
    status: str  # "grokked" | "memorized-only" | "no-fit"
    memorized_epoch: int | None  # first epoch with train loss < mem_thresh
    cliff_epoch: int | None  # first epoch with test loss < grok_thresh
    final_train_loss: float
    final_test_loss: float


@run_measure("grok_status", version=1)
def grok_status(run, mem_thresh: float = 0.1, grok_thresh: float = 0.1) -> GrokStatus:
    from ..interp.dynamics import grokking_status

    train, test = _loss_history(run)
    if len(train) == 0 or len(test) == 0:
        raise ValueError(f"{run.id} has no loss history")
    st = grokking_status(train, test, mem_thresh=mem_thresh, grok_thresh=grok_thresh)
    if st["grokked"]:
        status = "grokked"
    elif st["memorized_step"] is not None:
        status = "memorized-only"
    else:
        status = "no-fit"
    return GrokStatus(
        status, st["memorized_step"], st["cliff_step"], float(train[-1]), st["final_test_loss"]
    )


def k_class(K, n: int) -> str:
    """Coarse label of K by its characters' orders in the group of order n."""
    orders = [n // math.gcd(int(k), n) for k in K]
    if any(o == n for o in orders):
        return "primitive"
    if all(o % 2 == 1 for o in orders):
        return "odd"
    if all(o & (o - 1) == 0 for o in orders):
        return "two_power"
    return "mixed"


@dataclass
class DynamicsMarkers:
    K: tuple[int, ...]  # energy support of the final recorded embedding
    K_class: str  # see k_class
    cliff_epoch: int | None
    bifurcation_epoch: int | None  # from metrics.pt; None if never or not recorded
    commit_epoch: int | None
    bifurcation_epoch_fine: int | None  # from per-step fine_grained.pt, when present
    ratio_at: float | None  # K / non-K mean energy ratio at the epoch nearest ``at_epoch``
    epochs: np.ndarray  # epochs recorded in metrics.pt
    ratio: np.ndarray  # K / non-K mean energy ratio at those epochs


def _ratio(E: np.ndarray, mask: np.ndarray) -> np.ndarray:
    non_k = E[:, ~mask].mean(axis=1)
    return E[:, mask].mean(axis=1) / np.where(non_k > 0, non_k, 1e-30)


@run_measure("dynamics_markers", version=1)
def dynamics_markers(
    run, threshold: float = 0.05, ratio: float = 1.5, at_epoch: int = 1500
) -> DynamicsMarkers:
    from ..interp.dynamics import bifurcation_step, cliff_step, commit_step
    from ..interp.metrics import char_energy_batch

    structure = run.task.structure()
    n = structure.n
    _, test = _loss_history(run)
    cliff = cliff_step(test)
    empty = np.zeros(0)
    metrics_path = run.path / "metrics.pt"
    if not metrics_path.exists():
        return DynamicsMarkers((), "", cliff, None, None, None, None, empty, empty)

    m = torch.load(metrics_path, weights_only=False)
    epochs = np.asarray(m["epochs"])
    E = np.asarray(m["freq_energies"], dtype=np.float64)  # (T, n_chars), column k − 1
    final = E[-1]
    K = tuple(int(k + 1) for k in np.flatnonzero(final >= threshold * final.max()))
    mask = np.zeros(E.shape[1], dtype=bool)
    mask[[k - 1 for k in K]] = True

    if mask.all():  # every character is in K: there is no "rest" to outgrow
        return DynamicsMarkers(
            K, k_class(K, n), cliff, None, None, None, None, epochs, np.full(len(epochs), np.nan)
        )
    bif = bifurcation_step(E, mask, ratio=ratio)
    com = commit_step(E, list(K), mode="subset")
    r = _ratio(E, mask)
    i = int(np.argmin(np.abs(epochs - at_epoch)))

    bif_fine = None
    fine_path = run.path / "fine_grained.pt"
    if fine_path.exists():
        fg = torch.load(fine_path, weights_only=False)
        basis, ci = structure.char_basis()
        E_fine = char_energy_batch(fg["W_E"][:, :, : structure.n_values], basis, ci)
        step = bifurcation_step(E_fine, mask, ratio=ratio)
        bif_fine = None if step is None else int(fg["epochs"][step])

    return DynamicsMarkers(
        K=K,
        K_class=k_class(K, n),
        cliff_epoch=cliff,
        bifurcation_epoch=None if bif is None else int(epochs[bif]),
        commit_epoch=None if com is None else int(epochs[com]),
        bifurcation_epoch_fine=bif_fine,
        ratio_at=float(r[i]),
        epochs=epochs,
        ratio=r,
    )
