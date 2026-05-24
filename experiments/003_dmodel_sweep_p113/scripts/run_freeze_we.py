"""Freeze W_E at init; train the rest of the network. Does the model still
find a multiplicative-character basis, and how fast?

Per-step records (full window):
  - W_U character energy (W_U columns over the c-axis projected onto chars)
  - W_E character energy (frozen — sanity check that it doesn't move)
  - train loss
  - test loss

Outputs:
  data/freeze_we/seed{N}_freeze_we.pt   (full per-step trajectory)

Usage:
    python experiments/003_dmodel_sweep_p113/scripts/run_freeze_we.py --seed 1
"""
from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim

from crypto_interp import data
from crypto_interp.interp.bases import multiplicative_fourier_basis
from crypto_interp.models import Transformer, TransformerConfig
from crypto_interp.training.loop import cross_entropy_high_precision

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ROOT / "datasets"
OUT_DIR = ROOT / "data" / "freeze_we"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def build_char_basis(device):
    basis, names, _ = multiplicative_fourier_basis(113, device=device)
    char_index = {}
    for i, nm in enumerate(names):
        m = re.match(r"mul (cos|sin) (\d+)", nm)
        if m:
            kk = int(m.group(2))
            char_index.setdefault(kk, []).append(i)
    return basis.double(), char_index


def char_energy_dp(tensor_dp, basis, char_index, n_chars=56):
    """tensor_dp: (d_model, p) — energy of each character across d_model rows."""
    coef = torch.einsum("kp,dp->kd", basis, tensor_dp.double())
    E = (coef ** 2).sum(dim=1).cpu().numpy()
    out = np.zeros(n_chars)
    for k, rows in char_index.items():
        out[k - 1] = float(E[rows].sum())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--num-epochs", type=int, default=10000)
    ap.add_argument("--freeze-we", action="store_true", default=True)
    ap.add_argument("--no-freeze-we", dest="freeze_we", action="store_false")
    args = ap.parse_args()

    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cpu")

    ds = data.load_or_build(cache_dir=DATASETS, task="mul", p=113,
                            frac_train=0.3, seed=seed)
    train_in = ds.inputs[ds.train_mask]
    train_lab = ds.labels[ds.train_mask]
    test_in = ds.inputs[ds.test_mask]
    test_lab = ds.labels[ds.test_mask]

    cfg = TransformerConfig(d_vocab=ds.vocab_size, d_model=24, d_mlp=512,
                            num_heads=4, d_head=32, n_ctx=3, num_layers=1)
    model = Transformer(cfg).to(device)

    # ---- freeze W_E ----
    if args.freeze_we:
        model.embed.W_E.requires_grad_(False)
        tag = "freeze_we"
    else:
        tag = "control"

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = optim.AdamW(trainable, lr=1e-3, weight_decay=1.0, betas=(0.9, 0.98))
    warmup = 10
    sched = optim.lr_scheduler.LambdaLR(opt, lambda s: min(s / warmup, 1.0))

    basis, char_index = build_char_basis(device)
    p = 113
    T = args.num_epochs

    char_E_WE = np.zeros((T, 56))  # should be constant if frozen
    char_E_WU = np.zeros((T, 56))
    train_losses = np.zeros(T)
    test_losses = np.zeros(T)

    t0 = time.time()
    for ep in range(T):
        logits = model(train_in)[:, -1, :ds.n_answer_tokens]
        loss = cross_entropy_high_precision(logits, train_lab, True)
        with torch.no_grad():
            tlog = model(test_in)[:, -1, :ds.n_answer_tokens]
            tloss = cross_entropy_high_precision(tlog, test_lab, True)
        train_losses[ep] = float(loss.item())
        test_losses[ep] = float(tloss.item())

        loss.backward()
        # Record signals.
        W_E = model.embed.W_E.detach()[:, :p]
        W_U = model.unembed.W_U.detach()[:, :p]   # (d_model, vocab) — same dims as W_E
        char_E_WE[ep] = char_energy_dp(W_E, basis, char_index)
        char_E_WU[ep] = char_energy_dp(W_U, basis, char_index)

        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=False)

        if ep % 500 == 0:
            print(f"  [{tag}] step {ep:5d}  log10 train {np.log10(loss.item()):7.3f}  "
                  f"log10 test {np.log10(tloss.item()):7.3f}  {time.time() - t0:.1f}s")

    out_path = OUT_DIR / f"seed{seed:02d}_{tag}.pt"
    torch.save({
        "seed": seed, "tag": tag,
        "char_E_WE": torch.tensor(char_E_WE),
        "char_E_WU": torch.tensor(char_E_WU),
        "train_losses": torch.tensor(train_losses),
        "test_losses": torch.tensor(test_losses),
    }, out_path)
    print(f"Saved {out_path}  ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
