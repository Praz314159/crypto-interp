"""Train a model for a given experiment.

Usage:
    python scripts/train.py --experiment 001_mul_p113 --tag main_run
    python scripts/train.py --experiment 001_mul_p113 --seed-override 7 --tag seed7
    python scripts/train.py --experiment 001_mul_p113 --resume experiments/001_mul_p113/runs/main_run/checkpoint_006500.pt --num-epochs 2000

Loads ``experiments/<id>/config.py:CONFIG`` and hands it to the library's
training loop. CLI flags can override the high-traffic knobs (seed, epochs,
tag, logging cadence) without editing the config file — useful for sweeps.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from dataclasses import replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
# Make `crypto_interp` importable even without `pip install -e .`.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def load_experiment_config(experiment_id: str):
    """Import ``experiments/<id>/config.py`` and return its CONFIG attribute."""
    cfg_path = REPO_ROOT / "experiments" / experiment_id / "config.py"
    if not cfg_path.exists():
        raise FileNotFoundError(f"No config at {cfg_path}")
    spec = importlib.util.spec_from_file_location(
        f"experiments.{experiment_id}.config", cfg_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "CONFIG"):
        raise AttributeError(f"{cfg_path} does not export CONFIG")
    return module.CONFIG


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True,
                        help="Experiment id (directory under experiments/, e.g. 001_mul_p113).")
    parser.add_argument("--tag", default=None,
                        help="Sub-directory name under <experiment>/runs/. "
                             "Defaults to a timestamp.")
    # Overrides (omit → use CONFIG's value)
    parser.add_argument("--seed-override", type=int, default=None,
                        help="Override CONFIG.seed for this run.")
    parser.add_argument("--num-epochs", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=None)
    parser.add_argument("--save-every", type=int, default=None)
    parser.add_argument("--metrics-every", type=int, default=None)
    parser.add_argument("--device", type=str, default=None,
                        help="Override CONFIG.device. 'auto'|'cpu'|'mps'|'cuda'.")
    parser.add_argument("--early-stop-loss", type=float, default=None)
    parser.add_argument("--early-stop-patience", type=int, default=None)
    # Resume
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from this checkpoint path. --num-epochs is "
                             "interpreted as additional epochs.")
    args = parser.parse_args()

    cfg = load_experiment_config(args.experiment)

    overrides = {}
    if args.seed_override is not None: overrides["seed"] = args.seed_override
    if args.num_epochs is not None: overrides["num_epochs"] = args.num_epochs
    if args.log_every is not None: overrides["log_every"] = args.log_every
    if args.save_every is not None: overrides["save_every"] = args.save_every
    if args.metrics_every is not None: overrides["metrics_every"] = args.metrics_every
    if args.device is not None: overrides["device"] = args.device
    if args.early_stop_loss is not None: overrides["early_stop_loss"] = args.early_stop_loss
    if args.early_stop_patience is not None: overrides["early_stop_patience"] = args.early_stop_patience
    if overrides:
        cfg = replace(cfg, **overrides)

    exp_dir = REPO_ROOT / "experiments" / args.experiment
    datasets_dir = exp_dir / "datasets"

    # Import inside main so --help doesn't pull in torch.
    from crypto_interp.training import train, resume

    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.is_absolute():
            resume_path = (REPO_ROOT / resume_path).resolve()
        result = resume(cfg, resume_path,
                        additional_epochs=cfg.num_epochs,
                        datasets_dir=datasets_dir)
    else:
        import time
        tag = args.tag or f"{cfg.task}_{int(time.time())}"
        run_dir = exp_dir / "runs" / tag
        result = train(cfg, run_dir=run_dir, datasets_dir=datasets_dir)

    print(f"\nrun_dir={result['run_dir']}")
    return result


if __name__ == "__main__":
    main()
