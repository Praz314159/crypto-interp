"""Training loop for the sqrt-mod-p experiment.

Mirrors Nanda's setup:
  - AdamW, lr=1e-3, weight_decay=1.0, betas=(0.9, 0.98), warmup over 10 steps.
  - Full-batch training (no minibatches).
  - Cross-entropy in float64 to avoid float32 underflow on confident logits.
  - Save checkpoints periodically so we can study training dynamics.
"""

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

import data
from model import Transformer, TransformerConfig


def cross_entropy_high_precision(
    logits: torch.Tensor,
    labels: torch.Tensor,
    use_float64: bool,
) -> torch.Tensor:
    """Cross entropy with optional float64 log-softmax.

    Float64 avoids float32 underflow on confident predictions which Nanda
    reports causes grokking-relevant loss spikes. MPS doesn't support float64,
    so we fall back to float32 there.
    """
    if use_float64:
        logits = logits.to(torch.float64)
    logprobs = F.log_softmax(logits, dim=-1)
    sel = torch.gather(logprobs, index=labels[:, None], dim=-1)
    return -sel.mean()


def loss_fn(model, inputs, labels, n_answer_tokens, use_float64):
    # Take logits at the final position; restrict to answer tokens (sqrt values
    # 0..p-1 plus NoRoot at index p), excluding the '=' separator at p+1.
    logits = model(inputs)[:, -1, :n_answer_tokens]
    return cross_entropy_high_precision(logits, labels, use_float64)


def pick_device(arg: str) -> torch.device:
    if arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        # Default to CPU because MPS lacks float64. User can override with --device mps.
        return torch.device("cpu")
    return torch.device(arg)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="mul",
                        choices=data.available_tasks(),
                        help="Which modular-arithmetic task to train on.")
    parser.add_argument("--p", type=int, default=113)
    parser.add_argument("--frac-train", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0)
    parser.add_argument("--num-epochs", type=int, default=50000)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--out", type=str, default="runs")
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto",
                        help="auto|cpu|mps|cuda. Auto avoids MPS because MPS lacks float64.")
    parser.add_argument("--metrics-every", type=int, default=0,
                        help="If >0, save per-frequency embedding energy every N epochs "
                             "to metrics.pt for trajectory analysis.")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to a checkpoint to resume from. Continues training "
                             "in the same run dir, appending to losses/metrics. "
                             "--num-epochs is interpreted as additional epochs.")
    parser.add_argument("--early-stop-loss", type=float, default=0.0,
                        help="Stop training when test_loss < this value for "
                             "--early-stop-patience consecutive epochs. 0 = disabled.")
    parser.add_argument("--early-stop-patience", type=int, default=500)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = pick_device(args.device)
    use_float64 = device.type != "mps"
    print(f"Device: {device}, high-precision (float64) loss: {use_float64}")
    if not use_float64:
        print("WARNING: MPS doesn't support float64. Float32 cross-entropy may "
              "cause grokking-relevant loss spikes (see Nanda et al. appendix).")

    ds = data.load_or_build(task=args.task, p=args.p,
                            frac_train=args.frac_train, seed=args.seed)
    inputs = ds.inputs.to(device)
    labels = ds.labels.to(device)
    train_mask = ds.train_mask.to(device)
    test_mask = ds.test_mask.to(device)
    print(f"task={ds.task}, p={ds.p}, total={len(ds.inputs)}, "
          f"train={ds.n_train}, test={ds.n_test}, vocab={ds.vocab_size}")

    cfg = TransformerConfig(d_vocab=ds.vocab_size)
    model = Transformer(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.98),
    )
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda s: min(s / 10, 1.0))

    start_epoch = 0
    if args.resume:
        resume_path = Path(args.resume)
        print(f"Resuming from {resume_path}")
        rckpt = torch.load(resume_path, weights_only=False)
        model.load_state_dict(rckpt["model_state"])
        optimizer.load_state_dict(rckpt["optimizer_state"])
        if "scheduler_state" in rckpt:
            scheduler.load_state_dict(rckpt["scheduler_state"])
        else:
            # Old checkpoint without scheduler state; warmup is long done by 6500 epochs
            for _ in range(rckpt["epoch"] + 1):
                scheduler.step()
        start_epoch = rckpt["epoch"] + 1
        train_losses = list(rckpt["train_losses"])
        test_losses = list(rckpt["test_losses"])
        run_dir = resume_path.parent
        tag = run_dir.name
        print(f"  Resumed at epoch {start_epoch}. Run dir: {run_dir}")
    else:
        tag = args.tag or f"sqrt_p{args.p}_{int(time.time())}"
        run_dir = Path(args.out) / tag
        run_dir.mkdir(parents=True, exist_ok=True)
        train_losses, test_losses = [], []
        print(f"Saving to: {run_dir}")

    # Per-frequency embedding-energy tracking (multiplicative Fourier basis).
    track_metrics = args.metrics_every > 0
    metrics = {"epochs": [], "freq_energies": []}
    if args.resume and (run_dir / "metrics.pt").exists():
        old = torch.load(run_dir / "metrics.pt", weights_only=False)
        metrics["epochs"] = list(old["epochs"])
        metrics["freq_energies"] = list(old["freq_energies"])
    if track_metrics:
        from interp.bases import multiplicative_fourier_basis
        mul_basis, _, _ = multiplicative_fourier_basis(ds.p, device="cpu")
        mul_basis = mul_basis.to(device)
        print(f"Tracking metrics every {args.metrics_every} epochs.")
    start = time.time()
    end_epoch = start_epoch + args.num_epochs
    early_stop_streak = 0
    last_epoch_done = start_epoch
    for epoch in range(start_epoch, end_epoch):
        last_epoch_done = epoch
        train_loss = loss_fn(model, inputs[train_mask], labels[train_mask], ds.n_answer_tokens, use_float64)
        with torch.no_grad():
            test_loss = loss_fn(model, inputs[test_mask], labels[test_mask], ds.n_answer_tokens, use_float64)
        train_losses.append(train_loss.item())
        test_losses.append(test_loss.item())

        train_loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        if track_metrics and epoch % args.metrics_every == 0:
            with torch.no_grad():
                W_E_vals = model.embed.W_E[:, : ds.p].detach().to(torch.float64)
                coef = torch.einsum("kp,dp->kd", mul_basis.to(torch.float64), W_E_vals)
                per_basis_energy = (coef ** 2).sum(dim=1)  # (p,)
                # Compress to per-frequency (sum cos + sin pair)
                p = ds.p
                n = p - 1
                freq_e = []
                for k in range(1, (n - 1) // 2 + 1):
                    freq_e.append((per_basis_energy[2 * k] + per_basis_energy[2 * k + 1]).item())
                if n % 2 == 0:
                    freq_e.append(per_basis_energy[n].item())
                metrics["epochs"].append(epoch)
                metrics["freq_energies"].append(freq_e)

        if epoch % args.log_every == 0:
            elapsed = time.time() - start
            print(f"epoch {epoch:6d} | log10 train {np.log10(train_loss.item()):7.3f} | "
                  f"log10 test {np.log10(test_loss.item()):7.3f} | {elapsed:.1f}s")

        # Early stop check
        if args.early_stop_loss > 0:
            if test_loss.item() < args.early_stop_loss:
                early_stop_streak += 1
                if early_stop_streak >= args.early_stop_patience:
                    print(f"Early stop at epoch {epoch}: test_loss < "
                          f"{args.early_stop_loss} for {args.early_stop_patience} epochs.")
                    # Save final checkpoint before breaking
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state": model.state_dict(),
                            "optimizer_state": optimizer.state_dict(),
                            "scheduler_state": scheduler.state_dict(),
                            "train_losses": train_losses,
                            "test_losses": test_losses,
                            "config": {
                                "task": ds.task, "p": ds.p,
                                "frac_train": args.frac_train, "seed": args.seed,
                                "lr": args.lr, "weight_decay": args.weight_decay,
                                "d_model": cfg.d_model, "d_mlp": cfg.d_mlp,
                                "num_heads": cfg.num_heads, "d_head": cfg.d_head,
                            },
                        },
                        run_dir / f"checkpoint_{epoch:06d}.pt",
                    )
                    last_epoch_done = epoch
                    break
            else:
                early_stop_streak = 0

        if (epoch % args.save_every == 0) or (epoch == end_epoch - 1):
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "train_losses": train_losses,
                    "test_losses": test_losses,
                    "config": {
                        "task": ds.task,
                        "p": ds.p,
                        "frac_train": args.frac_train,
                        "seed": args.seed,
                        "lr": args.lr,
                        "weight_decay": args.weight_decay,
                        "d_model": cfg.d_model,
                        "d_mlp": cfg.d_mlp,
                        "num_heads": cfg.num_heads,
                        "d_head": cfg.d_head,
                    },
                },
                run_dir / f"checkpoint_{epoch:06d}.pt",
            )

    torch.save(
        {"train_losses": train_losses, "test_losses": test_losses},
        run_dir / "losses.pt",
    )
    if track_metrics:
        torch.save(
            {
                "epochs": metrics["epochs"],
                "freq_energies": np.array(metrics["freq_energies"]),  # (n_steps, n_freqs)
                "config": {"task": ds.task, "p": ds.p, "seed": args.seed,
                           "frac_train": args.frac_train, "metrics_every": args.metrics_every},
            },
            run_dir / "metrics.pt",
        )
        print(f"Saved metrics: {run_dir / 'metrics.pt'}")
    print(f"Done. Final train loss {train_losses[-1]:.6f}, test loss {test_losses[-1]:.6f}")


if __name__ == "__main__":
    main()
