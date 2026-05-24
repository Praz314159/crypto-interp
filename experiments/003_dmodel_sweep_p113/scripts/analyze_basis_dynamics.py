"""Empirical analysis of basis-selection dynamics across seeds.

Implements three measurements on existing seed checkpoints:

  G1: Per-character W_E energy trajectory over training (every 500 steps).
      For each character k in 1..56, energy_k(t) = ||proj_k(W_E_values(t))||^2.

  G2: Eigenspectrum of W_E_values^T W_E_values over training, plus the
      character-composition of each top eigenvector. Tracks when the
      eigen-identity stabilizes.

  B4: NTK-style measurement at t=0. Compute gradient of training loss
      w.r.t. W_E at init, project onto character basis. Rank characters by
      |grad|, |init|, and |grad|*|init| (the first-step amplification heuristic).
      Test whether top-ranked predicts observed K.

Outputs go to ``experiments/003_dmodel_sweep_p113/data/basis_dynamics/``:
  - per-seed npz with G1+G2 time series
  - one summary csv with B4 ranks + observed K + cliff time

Usage:
    python experiments/003_dmodel_sweep_p113/scripts/analyze_basis_dynamics.py
"""
from __future__ import annotations

import math
import pickle
import re
import time
from pathlib import Path

import numpy as np
import torch

from crypto_interp.interp.bases import multiplicative_fourier_basis
from crypto_interp.interp.load import load_run

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
OUT = ROOT / "data" / "basis_dynamics"
OUT.mkdir(parents=True, exist_ok=True)


# ---------- helpers ----------

def list_seed_dirs() -> list[Path]:
    """Find all dmodel_24_seedN directories (no wd suffix)."""
    pat = re.compile(r"^dmodel_24_seed(\d+)$")
    out = []
    for d in sorted(RUNS.iterdir()):
        m = pat.match(d.name)
        if m and d.is_dir():
            out.append((int(m.group(1)), d))
    out.sort(key=lambda x: x[0])
    return out


def list_ckpts(seed_dir: Path) -> list[tuple[int, Path]]:
    out = []
    for p in sorted(seed_dir.glob("checkpoint_*.pt")):
        m = re.match(r"checkpoint_(\d+)\.pt", p.name)
        if m:
            out.append((int(m.group(1)), p))
    return out


def build_char_basis_p113():
    """Return (basis_pp, char_index_map) where char_index_map[k] = list of basis row
    indices that together compose multiplicative-Fourier character k. For k=1..55,
    each character has a (cos_k, sin_k) pair; for k=56 only cos_56."""
    p = 113
    basis, names, g = multiplicative_fourier_basis(p)
    # names = ["delta_0", "mul const", "mul cos 1", "mul sin 1", ..., "mul cos 56"]
    char_index = {}  # k -> list of row indices
    for i, nm in enumerate(names):
        m = re.match(r"mul (cos|sin) (\d+)", nm)
        if m:
            kk = int(m.group(2))
            char_index.setdefault(kk, []).append(i)
    return basis, names, char_index, g


def project_W_E(W_E_values: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """W_E_values: (d_model, p). Returns coef (n_basis, d_model)."""
    return torch.einsum("kp,dp->kd", basis.to(W_E_values.dtype), W_E_values)


def char_energies(coef: torch.Tensor, char_index: dict[int, list[int]]) -> np.ndarray:
    """coef: (n_basis, d_model). Returns array length 56 of per-character total energy."""
    out = np.zeros(56)
    for k, rows in char_index.items():
        out[k - 1] = float((coef[rows] ** 2).sum())
    return out


def topk_eigvecs(W_E_values: torch.Tensor, k: int = 24):
    """Top-k eigenvectors of W_E_values^T W_E_values (token-space). Returns (eigvals, V).
    V has shape (p, k)."""
    M = W_E_values.T @ W_E_values  # (p, p)
    eigvals, eigvecs = torch.linalg.eigh(M)  # ascending
    # take top-k
    eigvals = eigvals[-k:].flip(0)
    eigvecs = eigvecs[:, -k:].flip(1)
    return eigvals, eigvecs


def eigvec_char_composition(V: torch.Tensor, basis: torch.Tensor,
                            char_index: dict[int, list[int]]) -> np.ndarray:
    """For each eigenvector (column of V), compute per-character energy via the basis.
    V: (p, k). Returns (k, 56)."""
    # coef[j, m] = sum_a basis[j,a] V[a,m]   shape (n_basis, k_eig)
    coef = torch.einsum("jp,pm->jm", basis.to(V.dtype), V)
    n_eig = V.shape[1]
    out = np.zeros((n_eig, 56))
    for k_char, rows in char_index.items():
        out[:, k_char - 1] = (coef[rows] ** 2).sum(dim=0).cpu().numpy()
    return out


# ---------- G1 + G2 per-seed trajectory ----------

def analyze_seed_trajectory(seed: int, seed_dir: Path, basis, char_index) -> dict:
    ckpts = list_ckpts(seed_dir)
    if not ckpts:
        return None
    n_ck = len(ckpts)
    print(f"  seed {seed}: {n_ck} checkpoints, range {ckpts[0][0]}-{ckpts[-1][0]}")

    epochs = np.zeros(n_ck, dtype=np.int64)
    char_E = np.zeros((n_ck, 56), dtype=np.float64)
    eigvals = np.zeros((n_ck, 24), dtype=np.float64)
    eigvec_comp = np.zeros((n_ck, 24, 56), dtype=np.float64)

    t0 = time.time()
    for i, (ep, path) in enumerate(ckpts):
        epochs[i] = ep
        state = torch.load(path, weights_only=False, map_location="cpu")["model_state"]
        W_E = state["embed.W_E"].double()  # (d_model, vocab)
        W_E_v = W_E[:, :113]
        coef = project_W_E(W_E_v, basis)
        char_E[i] = char_energies(coef, char_index)
        ev, V = topk_eigvecs(W_E_v, k=24)
        eigvals[i] = ev.cpu().numpy()
        eigvec_comp[i] = eigvec_char_composition(V, basis, char_index)
    print(f"    done in {time.time() - t0:.1f}s")

    return dict(
        seed=seed,
        epochs=epochs,
        char_energy=char_E,
        eigvals=eigvals,
        eigvec_char_composition=eigvec_comp,
    )


# ---------- B4: NTK at init ----------

def b4_init_gradient(seed: int, seed_dir: Path, basis, char_index) -> dict:
    """Load model at checkpoint_000000.pt, compute grad of train loss w.r.t. W_E,
    project onto character basis. Return per-character |init|, |grad|, |grad|*|init|."""
    init_ckpt = seed_dir / "checkpoint_000000.pt"
    model, ds, ckpt = load_run(init_ckpt, device="cpu")
    model.train()

    # Training batch: all (a, b, =) inputs for which train_mask is True.
    mask = ds.train_mask.bool()
    inputs = ds.inputs[mask]
    labels = ds.labels[mask]

    model.zero_grad()
    logits = model(inputs)[:, -1, :]  # (N, vocab) at final position
    loss = torch.nn.functional.cross_entropy(logits, labels)
    loss.backward()

    grad_WE = model.embed.W_E.grad.detach().double()[:, :113]  # (d_model, p)
    init_WE = model.embed.W_E.detach().double()[:, :113]

    init_coef = project_W_E(init_WE, basis)
    grad_coef = project_W_E(grad_WE, basis)

    init_E = char_energies(init_coef, char_index)
    grad_E = char_energies(grad_coef, char_index)
    # First-step amplification heuristic: |grad|*|init|
    # (Equivalent in spirit to gradient flow growth dx/dt ∝ -∂L/∂x ∝ x in a margin/linear regime.)
    amp = np.sqrt(init_E) * np.sqrt(grad_E)

    return dict(
        seed=seed,
        init_energy=init_E,
        grad_energy=grad_E,
        amp=amp,
        train_loss_at_init=float(loss.item()),
    )


# ---------- main ----------

def main():
    print("Building multiplicative-Fourier character basis ...")
    basis, names, char_index, g = build_char_basis_p113()
    print(f"  primitive root g={g}, {len(char_index)} characters indexed")

    seed_dirs = list_seed_dirs()
    print(f"\nFound {len(seed_dirs)} seed directories")

    # G1 + G2: trajectory analysis per seed.
    trajectory_out = OUT / "trajectories.pkl"
    trajectories = {}
    for seed, sdir in seed_dirs:
        try:
            res = analyze_seed_trajectory(seed, sdir, basis, char_index)
            if res:
                trajectories[seed] = res
        except Exception as e:
            print(f"  seed {seed}: ERROR {e}")
    with open(trajectory_out, "wb") as f:
        pickle.dump(trajectories, f)
    print(f"\nSaved G1/G2 trajectories: {trajectory_out}")

    # B4: init-gradient per seed.
    b4_out = OUT / "b4_init_gradient.pkl"
    b4 = {}
    print("\nComputing B4 (init-gradient projection)...")
    for seed, sdir in seed_dirs:
        try:
            res = b4_init_gradient(seed, sdir, basis, char_index)
            b4[seed] = res
            top_amp = np.argsort(res["amp"])[::-1][:8] + 1
            print(f"  seed {seed}: top-8 amp characters = {list(top_amp)}")
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  seed {seed}: ERROR {e}")
    with open(b4_out, "wb") as f:
        pickle.dump(b4, f)
    print(f"\nSaved B4: {b4_out}")


if __name__ == "__main__":
    main()
