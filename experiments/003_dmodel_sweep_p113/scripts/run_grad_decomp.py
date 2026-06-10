"""Decompose the gradient at W_E by which path through the network produced it.

The 1-layer transformer has the residual structure
    resid_pre  = embed(x)
    resid_mid  = resid_pre + attn(resid_pre)
    resid_post = resid_mid + mlp(resid_mid)

So the gradient at W_E flows back along four paths from logits:
  A. skip-skip            logits → resid_post → resid_mid → resid_pre → W_E
  B. skip-MLP             logits → resid_post → mlp_out → resid_mid → resid_pre → W_E
  C. attn-skip            logits → resid_post → resid_mid → attn_out → resid_pre → W_E
  D. attn-MLP             logits → resid_post → mlp_out → resid_mid → attn_out → resid_pre → W_E

We isolate them with four forward passes per step using detach() to cut paths:
  full     g_full   = A + B + C + D
  no_attn  g_full - C - D    (attn_out detached → no gradient through C, D)
  no_mlp   g_full - B - D    (mlp_out detached → no gradient through B, D)
  bare     g_full - B - C - D = A   (both detached)

We never actually combine; we just record all four characterizations:
  full, no_attn (= A+B), no_mlp (= A+C), bare (= A)

From these you can recover each path: A=bare, B = no_mlp_contrib = (no_attn - A),
C = (no_mlp - A), D = full - A - B - C.

Records per-step:
  - full grad energy per character
  - bare (skip-only) grad
  - skip+MLP grad      (attn ablated)
  - skip+attn grad     (mlp ablated)

Training itself uses the full gradient (the decompositions are read-only
auxiliary backward passes).

Usage:
    python experiments/003_dmodel_sweep_p113/scripts/run_grad_decomp.py --seed 1
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
OUT_DIR = ROOT / "data" / "grad_decomp"
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


def char_energy(tensor_dp, basis, char_index, n_chars=56):
    coef = torch.einsum("kp,dp->kd", basis, tensor_dp.double())
    E = (coef ** 2).sum(dim=1).cpu().numpy()
    out = np.zeros(n_chars)
    for k, rows in char_index.items():
        out[k - 1] = float(E[rows].sum())
    return out


def forward_with_ablation(model, x, detach_attn=False, detach_mlp=False):
    """Run the model with optional detach() on attn_out and/or mlp_out.
    Returns logits at the final position.

    This relies on Transformer's exact structure (1 layer, 1 block)."""
    block = model.blocks[0]
    # embed + pos
    e = model.embed(x)
    e = model.pos_embed(e)
    # attn
    attn_out = block.attn(e)
    if detach_attn:
        attn_out = attn_out.detach()
    resid_mid = e + attn_out
    # mlp
    mlp_out = block.mlp(resid_mid)
    if detach_mlp:
        mlp_out = mlp_out.detach()
    resid_post = resid_mid + mlp_out
    return model.unembed(resid_post)


def compute_WE_grad(model, x, labels, n_ans, use_f64, detach_attn, detach_mlp):
    """Return the gradient of W_E for one forward+backward, with optional path
    ablation. Does not update model parameters."""
    model.zero_grad(set_to_none=False)
    logits = forward_with_ablation(model, x, detach_attn, detach_mlp)
    last = logits[:, -1, :n_ans]
    loss = cross_entropy_high_precision(last, labels, use_f64)
    loss.backward()
    g = model.embed.W_E.grad.detach().clone()
    model.zero_grad(set_to_none=False)
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--num-epochs", type=int, default=200)
    args = ap.parse_args()

    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cpu")

    ds = data.load_or_build(cache_dir=DATASETS, task="mul", p=113,
                            frac_train=0.3, seed=seed)
    train_in = ds.inputs[ds.train_mask]
    train_lab = ds.labels[ds.train_mask]

    cfg = TransformerConfig(d_vocab=ds.vocab_size, d_model=24, d_mlp=512,
                            num_heads=4, d_head=32, n_ctx=3, num_layers=1)
    model = Transformer(cfg).to(device)
    opt = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1.0,
                      betas=(0.9, 0.98))
    warmup = 10
    sched = optim.lr_scheduler.LambdaLR(opt, lambda s: min(s / warmup, 1.0))

    basis, char_index = build_char_basis(device)
    p = 113
    T = args.num_epochs
    char_G_full = np.zeros((T, 56))      # all paths (A+B+C+D)
    char_G_no_attn = np.zeros((T, 56))   # A+B (mlp path only, plus skip)
    char_G_no_mlp = np.zeros((T, 56))    # A+C (attn path only, plus skip)
    char_G_bare = np.zeros((T, 56))      # A only (pure skip)
    train_losses = np.zeros(T)

    t0 = time.time()
    for ep in range(T):
        # 1) full forward + backward → also record loss + train step
        opt.zero_grad(set_to_none=False)
        logits = model(train_in)[:, -1, :ds.n_answer_tokens]
        loss = cross_entropy_high_precision(logits, train_lab, True)
        train_losses[ep] = float(loss.item())
        loss.backward()
        g_full = model.embed.W_E.grad.detach()[:, :p].clone()
        char_G_full[ep] = char_energy(g_full, basis, char_index)
        # 2) Take the train step (uses g_full which is currently in .grad).
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=False)

        # 3) Compute the three ablated gradients at the *just-stepped* weights.
        # NOTE: ablated gradients are read-only; they don't drive training.
        for label, det_a, det_m, dest in [
            ("no_attn", True,  False, char_G_no_attn),
            ("no_mlp",  False, True,  char_G_no_mlp),
            ("bare",    True,  True,  char_G_bare),
        ]:
            g = compute_WE_grad(model, train_in, train_lab,
                                ds.n_answer_tokens, True, det_a, det_m)
            dest[ep] = char_energy(g[:, :p], basis, char_index)

        if ep % 50 == 0:
            print(f"  step {ep:4d}  log10 train {np.log10(loss.item()):.3f}  "
                  f"{time.time() - t0:.1f}s")

    out = OUT_DIR / f"seed{seed:02d}_grad_decomp.pt"
    torch.save({
        "seed": seed,
        "char_G_full":    torch.tensor(char_G_full),
        "char_G_no_attn": torch.tensor(char_G_no_attn),
        "char_G_no_mlp":  torch.tensor(char_G_no_mlp),
        "char_G_bare":    torch.tensor(char_G_bare),
        "train_losses":   torch.tensor(train_losses),
    }, out)
    print(f"Saved {out}  ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
