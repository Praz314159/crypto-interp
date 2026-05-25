"""Single-run training loop.

Faithful port of the original ``train.py`` main() with the CLI surface stripped
out — it now takes an :class:`ExperimentConfig` plus the paths it should write
to. Behavior is otherwise identical: Nanda's AdamW + warmup + full-batch +
float64 cross-entropy.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from .. import data
from ..interp.metrics import EmbeddingEnergyTracker
from ..models import Transformer, TransformerConfig
from .config import ExperimentConfig


# ----------------------------- helpers -----------------------------

def cross_entropy_high_precision(
    logits: torch.Tensor,
    labels: torch.Tensor,
    use_float64: bool,
) -> torch.Tensor:
    """CE with optional float64 log-softmax.

    Float64 avoids float32 underflow on confident predictions which Nanda reports
    causes grokking-relevant loss spikes. MPS doesn't support float64, so we
    fall back to float32 there.
    """
    if use_float64:
        logits = logits.to(torch.float64)
    logprobs = F.log_softmax(logits, dim=-1)
    sel = torch.gather(logprobs, index=labels[:, None], dim=-1)
    return -sel.mean()


def _loss_fn(model, inputs, labels, n_answer_tokens, use_float64):
    logits = model(inputs)[:, -1, :n_answer_tokens]
    return cross_entropy_high_precision(logits, labels, use_float64)


def pick_device(arg: str) -> torch.device:
    if arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        # Default to CPU because MPS lacks float64. User can override with device="mps".
        return torch.device("cpu")
    return torch.device(arg)


def _stamp_checkpoint(cfg: ExperimentConfig, datasets_dir: Path) -> dict:
    """Pack the config into the dict we attach to every checkpoint."""
    out = cfg.to_checkpoint_dict()
    out["datasets_dir"] = str(datasets_dir)
    return out


# Fields a resume() invocation is permitted to override; everything else must
# carry over from the loaded checkpoint so the stamp keeps describing the model
# that is actually being trained.
_RESUME_OVERRIDE_KEYS = (
    "log_every", "save_every", "metrics_every",
    "early_stop_loss", "early_stop_patience", "device",
)


def _stamp_resumed_checkpoint(
    saved_cfg: dict,
    cfg: ExperimentConfig,
    datasets_dir: Path,
) -> dict:
    """Build a checkpoint stamp for resumed training.

    Architecture, optimizer, and dataset fields are locked-in by the loaded
    checkpoint (saved_cfg). Only runtime knobs may be overridden by the
    resume invocation's cfg.
    """
    out = dict(saved_cfg)
    for k in _RESUME_OVERRIDE_KEYS:
        out[k] = getattr(cfg, k)
    out["datasets_dir"] = str(datasets_dir)
    return out


def _save_checkpoint(
    run_dir: Path,
    epoch: int,
    model,
    optimizer,
    scheduler,
    train_losses,
    test_losses,
    cfg_stamp: dict,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "train_losses": train_losses,
            "test_losses": test_losses,
            "config": cfg_stamp,
        },
        run_dir / f"checkpoint_{epoch:06d}.pt",
    )


# ----------------------------- public API -----------------------------

def train(
    cfg: ExperimentConfig,
    run_dir: Path,
    datasets_dir: Path,
) -> dict:
    """Train a fresh model under ``cfg`` and write artifacts to ``run_dir``.

    Returns the final (train_loss, test_loss) as a dict for convenience.
    """
    run_dir = Path(run_dir)
    datasets_dir = Path(datasets_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    device = pick_device(cfg.device)
    use_float64 = device.type != "mps"
    print(f"Device: {device}, high-precision (float64) loss: {use_float64}")
    if not use_float64:
        print("WARNING: MPS doesn't support float64. Float32 cross-entropy may "
              "cause grokking-relevant loss spikes (see Nanda et al. appendix).")

    ds = data.load_or_build(
        cache_dir=datasets_dir,
        task=cfg.task, p=cfg.p, frac_train=cfg.frac_train, seed=cfg.seed,
    )
    inputs = ds.inputs.to(device)
    labels = ds.labels.to(device)
    train_mask = ds.train_mask.to(device)
    test_mask = ds.test_mask.to(device)
    print(f"task={ds.task}, p={ds.p}, total={len(ds.inputs)}, "
          f"train={ds.n_train}, test={ds.n_test}, vocab={ds.vocab_size}")

    model_cfg = TransformerConfig(
        d_vocab=ds.vocab_size,
        d_model=cfg.d_model, d_mlp=cfg.d_mlp,
        num_heads=cfg.num_heads, d_head=cfg.d_head,
        n_ctx=cfg.n_ctx, num_layers=cfg.num_layers,
    )
    model = Transformer(model_cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    optimizer = optim.AdamW(
        model.parameters(), lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        betas=(cfg.beta1, cfg.beta2),
    )
    warmup = max(cfg.warmup_steps, 1)
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min(s / warmup, 1.0)
    )

    cfg_stamp = _stamp_checkpoint(cfg, datasets_dir)
    train_losses: list[float] = []
    test_losses: list[float] = []
    fg_W_E: list[torch.Tensor] = []   # per-step W_E for the first fine_grained_until epochs
    fg_epochs: list[int] = []

    tracker = (
        EmbeddingEnergyTracker(p=ds.p, device=device)
        if cfg.metrics_every > 0 else None
    )
    if tracker is not None:
        print(f"Tracking metrics every {cfg.metrics_every} epochs.")

    start = time.time()
    early_stop_streak = 0
    last_epoch_done = -1

    for epoch in range(cfg.num_epochs):
        last_epoch_done = epoch
        # Per-step W_E snapshot for the early fine-grained window (epoch 0 = init).
        if cfg.fine_grained_until and epoch < cfg.fine_grained_until:
            fg_W_E.append(model.embed.W_E.detach().cpu().clone())
            fg_epochs.append(epoch)
        train_loss = _loss_fn(model, inputs[train_mask], labels[train_mask],
                              ds.n_answer_tokens, use_float64)
        with torch.no_grad():
            test_loss = _loss_fn(model, inputs[test_mask], labels[test_mask],
                                 ds.n_answer_tokens, use_float64)
        train_losses.append(train_loss.item())
        test_losses.append(test_loss.item())

        train_loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        if tracker is not None and epoch % cfg.metrics_every == 0:
            tracker.record(epoch, model.embed.W_E[:, : ds.p].detach())

        if epoch % cfg.log_every == 0:
            elapsed = time.time() - start
            print(f"epoch {epoch:6d} | log10 train {np.log10(train_loss.item()):7.3f} | "
                  f"log10 test {np.log10(test_loss.item()):7.3f} | {elapsed:.1f}s")

        if cfg.early_stop_loss > 0:
            if test_loss.item() < cfg.early_stop_loss:
                early_stop_streak += 1
                if early_stop_streak >= cfg.early_stop_patience:
                    print(f"Early stop at epoch {epoch}: test_loss < "
                          f"{cfg.early_stop_loss} for {cfg.early_stop_patience} epochs.")
                    _save_checkpoint(run_dir, epoch, model, optimizer, scheduler,
                                     train_losses, test_losses, cfg_stamp)
                    break
            else:
                early_stop_streak = 0

        if (epoch % cfg.save_every == 0) or (epoch == cfg.num_epochs - 1):
            _save_checkpoint(run_dir, epoch, model, optimizer, scheduler,
                             train_losses, test_losses, cfg_stamp)

    torch.save({"train_losses": train_losses, "test_losses": test_losses},
               run_dir / "losses.pt")
    if fg_W_E:
        torch.save({"W_E": torch.stack(fg_W_E), "epochs": torch.tensor(fg_epochs)},
                   run_dir / "fine_grained.pt")
        print(f"Saved fine-grained W_E ({len(fg_W_E)} steps): {run_dir / 'fine_grained.pt'}")
    if tracker is not None:
        tracker.save(
            run_dir / "metrics.pt",
            config={"task": ds.task, "p": ds.p, "seed": cfg.seed,
                    "frac_train": cfg.frac_train, "metrics_every": cfg.metrics_every},
        )
        print(f"Saved metrics: {run_dir / 'metrics.pt'}")
    print(f"Done. Final train loss {train_losses[-1]:.6f}, "
          f"test loss {test_losses[-1]:.6f}")
    return {
        "epoch": last_epoch_done,
        "train_loss": train_losses[-1],
        "test_loss": test_losses[-1],
        "run_dir": str(run_dir),
    }


def resume(
    cfg: ExperimentConfig,
    resume_ckpt: Path,
    additional_epochs: int,
    datasets_dir: Path,
) -> dict:
    """Continue training from a checkpoint, appending to losses/metrics in place.

    The run_dir is inferred from the checkpoint's parent directory. ``cfg`` is
    only used for the knobs that aren't already locked-in by the checkpoint
    (logging cadence, early-stop config, metrics_every).
    """
    resume_ckpt = Path(resume_ckpt)
    run_dir = resume_ckpt.parent
    datasets_dir = Path(datasets_dir)

    print(f"Resuming from {resume_ckpt}")
    rckpt = torch.load(resume_ckpt, weights_only=False)
    saved_cfg = rckpt["config"]

    device = pick_device(cfg.device)
    use_float64 = device.type != "mps"
    print(f"Device: {device}, high-precision (float64) loss: {use_float64}")

    ds = data.load_or_build(
        cache_dir=datasets_dir,
        task=saved_cfg["task"], p=saved_cfg["p"],
        frac_train=saved_cfg["frac_train"], seed=saved_cfg["seed"],
    )
    inputs = ds.inputs.to(device)
    labels = ds.labels.to(device)
    train_mask = ds.train_mask.to(device)
    test_mask = ds.test_mask.to(device)

    model_cfg = TransformerConfig(
        d_vocab=ds.vocab_size,
        d_model=saved_cfg["d_model"], d_mlp=saved_cfg["d_mlp"],
        num_heads=saved_cfg["num_heads"], d_head=saved_cfg["d_head"],
        n_ctx=saved_cfg.get("n_ctx", 3),
        num_layers=saved_cfg.get("num_layers", 1),
    )
    model = Transformer(model_cfg).to(device)
    model.load_state_dict(rckpt["model_state"])

    optimizer = optim.AdamW(
        model.parameters(), lr=saved_cfg["lr"],
        weight_decay=saved_cfg["weight_decay"],
        betas=(saved_cfg.get("beta1", 0.9), saved_cfg.get("beta2", 0.98)),
    )
    optimizer.load_state_dict(rckpt["optimizer_state"])

    warmup = max(saved_cfg.get("warmup_steps", 10), 1)
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min(s / warmup, 1.0)
    )
    if "scheduler_state" in rckpt:
        scheduler.load_state_dict(rckpt["scheduler_state"])
    else:
        for _ in range(rckpt["epoch"] + 1):
            scheduler.step()

    start_epoch = rckpt["epoch"] + 1
    train_losses = list(rckpt["train_losses"])
    test_losses = list(rckpt["test_losses"])

    tracker = (
        EmbeddingEnergyTracker(p=ds.p, device=device)
        if cfg.metrics_every > 0 else None
    )
    if tracker is not None and (run_dir / "metrics.pt").exists():
        tracker.restore(torch.load(run_dir / "metrics.pt", weights_only=False))

    end_epoch = start_epoch + additional_epochs
    early_stop_streak = 0
    last_epoch_done = start_epoch
    start = time.time()
    print(f"Resuming at epoch {start_epoch}, training {additional_epochs} more.")

    cfg_stamp = _stamp_resumed_checkpoint(saved_cfg, cfg, datasets_dir)

    for epoch in range(start_epoch, end_epoch):
        last_epoch_done = epoch
        train_loss = _loss_fn(model, inputs[train_mask], labels[train_mask],
                              ds.n_answer_tokens, use_float64)
        with torch.no_grad():
            test_loss = _loss_fn(model, inputs[test_mask], labels[test_mask],
                                 ds.n_answer_tokens, use_float64)
        train_losses.append(train_loss.item())
        test_losses.append(test_loss.item())

        train_loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        if tracker is not None and epoch % cfg.metrics_every == 0:
            tracker.record(epoch, model.embed.W_E[:, : ds.p].detach())

        if epoch % cfg.log_every == 0:
            elapsed = time.time() - start
            print(f"epoch {epoch:6d} | log10 train {np.log10(train_loss.item()):7.3f} | "
                  f"log10 test {np.log10(test_loss.item()):7.3f} | {elapsed:.1f}s")

        if cfg.early_stop_loss > 0:
            if test_loss.item() < cfg.early_stop_loss:
                early_stop_streak += 1
                if early_stop_streak >= cfg.early_stop_patience:
                    print(f"Early stop at epoch {epoch}.")
                    _save_checkpoint(run_dir, epoch, model, optimizer, scheduler,
                                     train_losses, test_losses, cfg_stamp)
                    break
            else:
                early_stop_streak = 0

        if (epoch % cfg.save_every == 0) or (epoch == end_epoch - 1):
            _save_checkpoint(run_dir, epoch, model, optimizer, scheduler,
                             train_losses, test_losses, cfg_stamp)

    torch.save({"train_losses": train_losses, "test_losses": test_losses},
               run_dir / "losses.pt")
    if tracker is not None:
        tracker.save(
            run_dir / "metrics.pt",
            config={"task": ds.task, "p": ds.p, "seed": saved_cfg["seed"],
                    "frac_train": saved_cfg["frac_train"], "metrics_every": cfg.metrics_every},
        )
    print(f"Done. Final train {train_losses[-1]:.6f}, test {test_losses[-1]:.6f}")
    return {
        "epoch": last_epoch_done,
        "train_loss": train_losses[-1],
        "test_loss": test_losses[-1],
        "run_dir": str(run_dir),
    }
