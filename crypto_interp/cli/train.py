"""``crypto-interp train`` and ``crypto-interp sweep``."""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import os
import subprocess
import sys
import time
from dataclasses import fields, replace
from pathlib import Path

from ..runs import DEFAULT_ROOT, checkpoint_paths
from .select import parse_assignments, parse_value

# Fields that change how often things are logged, not what is trained; they
# are left out of default run names.
_CADENCE = {
    "log_every",
    "save_every",
    "metrics_every",
    "device",
    "notes",
    "early_stop_loss",
    "early_stop_patience",
}


def load_experiment_config(root: Path, experiment: str):
    """``CONFIG`` from ``<root>/<experiment>/config.py``."""
    path = Path(root) / experiment / "config.py"
    if not path.exists():
        raise SystemExit(f"no config at {path}")
    spec = importlib.util.spec_from_file_location(f"experiment_{experiment}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.CONFIG


def with_overrides(cfg, overrides: dict):
    """``cfg`` with ``overrides`` applied, each coerced to its field's type."""
    types = {f.name: f.type for f in fields(cfg)}
    coerced = {}
    for key, value in overrides.items():
        if key not in types:
            raise SystemExit(f"unknown config field {key!r}; known: {sorted(types)}")
        t = types[key] if isinstance(types[key], str) else types[key].__name__
        if t == "bool":
            coerced[key] = str(value).lower() in {"1", "true", "yes"}
        elif t in ("int", "float"):
            coerced[key] = {"int": int, "float": float}[t](value)
        else:
            coerced[key] = value
    return replace(cfg, **coerced)


def default_tag(overrides: dict, seed: int) -> str:
    """``d_model24_d_mlp32_seed3`` from the run-defining overrides and the seed."""
    parts = [
        f"{k}{v:g}" if isinstance(v, float) else f"{k}{v}"
        for k, v in sorted(overrides.items())
        if k not in _CADENCE | {"seed"}
    ]
    return "_".join(parts + [f"seed{seed}"])


def add_train_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--experiment", required=True, help="directory under the experiments root")
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    p.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override a config field, repeatable (d_model=24, weight_decay=2)",
    )
    p.add_argument("--num-epochs", type=int)
    p.add_argument("--save-every", type=int)
    p.add_argument("--metrics-every", type=int)
    p.add_argument("--device")


def _cli_overrides(args) -> dict:
    out = parse_assignments(args.set)
    for key in ("num_epochs", "save_every", "metrics_every", "device"):
        if getattr(args, key) is not None:
            out[key] = getattr(args, key)
    return out


def train_main(args) -> None:
    from ..data import TaskSpec
    from ..interp.metrics import EmbeddingEnergyCallback
    from ..training import resume, train

    overrides = _cli_overrides(args)
    if args.seed is not None:
        overrides["seed"] = args.seed
    cfg = with_overrides(load_experiment_config(args.root, args.experiment), overrides)
    callbacks = []
    if cfg.metrics_every > 0:
        task = TaskSpec(cfg.task, cfg.p, cfg.frac_train, cfg.seed)
        callbacks.append(EmbeddingEnergyCallback.for_task(task, every=cfg.metrics_every))

    if args.resume:
        result = resume(
            cfg, Path(args.resume), additional_epochs=cfg.num_epochs, callbacks=callbacks
        )
    else:
        run_dir = (
            args.root / args.experiment / "runs" / (args.tag or default_tag(overrides, cfg.seed))
        )
        if checkpoint_paths(run_dir):
            raise SystemExit(
                f"{run_dir} already has checkpoints; choose another --tag, or "
                "--resume one of them (a run directory holds one run)"
            )
        result = train(cfg, run_dir, callbacks=callbacks)
    print(f"run_dir={result['run_dir']}")


def sweep_main(args) -> None:
    """Train every (grid cell × seed) in parallel subprocesses."""
    seeds = parse_value(args.seeds)
    seeds = list(seeds) if isinstance(seeds, (range, list)) else [seeds]
    grid = parse_assignments(args.grid)
    keys = sorted(grid)
    values = [v if isinstance(v, (range, list)) else [v] for v in (grid[k] for k in keys)]
    fixed = _cli_overrides(args)
    jobs = []
    for combo in itertools.product(*values):
        cell = dict(zip(keys, combo, strict=True))
        for seed in seeds:
            tag = (
                args.tag_format.format(**fixed, **cell, seed=seed)
                if args.tag_format
                else default_tag({**fixed, **cell}, seed)
            )
            run_dir = args.root / args.experiment / "runs" / tag
            if args.skip_existing and checkpoint_paths(run_dir):
                state = (
                    "finished"
                    if _finished(run_dir, {**fixed, **cell}, args)
                    else ("unfinished: resume it with `train --resume`, or remove it, then rerun")
                )
                print(f"skip {tag} ({state})")
                continue
            cmd = [
                sys.executable,
                "-u",
                "-m",
                "crypto_interp.cli",
                "train",
                "--experiment",
                args.experiment,
                "--root",
                str(args.root),
                "--seed",
                str(seed),
                "--tag",
                tag,
            ]
            for k, v in {**fixed, **cell}.items():
                cmd += ["--set", f"{k}={v}"]
            jobs.append((tag, run_dir, cmd))

    print(f"{len(jobs)} runs, {args.jobs} at a time")
    env = os.environ.copy()
    if args.threads_per_job:
        env["OMP_NUM_THREADS"] = env["MKL_NUM_THREADS"] = str(args.threads_per_job)
    running, failed, start = [], [], time.time()
    while jobs or running:
        while jobs and len(running) < args.jobs:
            tag, run_dir, cmd = jobs.pop(0)
            run_dir.parent.mkdir(parents=True, exist_ok=True)
            log = open(run_dir.parent / f"{tag}.log", "w")
            running.append(
                (tag, subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env), log)
            )
            print(f"  started {tag}")
        time.sleep(1.0)
        for item in list(running):
            tag, proc, log = item
            if proc.poll() is None:
                continue
            log.close()
            running.remove(item)
            status = "done" if proc.returncode == 0 else f"FAILED (rc {proc.returncode})"
            if proc.returncode:
                failed.append(tag)
            print(f"  [{time.time() - start:6.0f}s] {tag} {status}")
    if failed:
        raise SystemExit(f"{len(failed)} runs failed: {failed}")


def _finished(run_dir: Path, overrides: dict, args) -> bool:
    """A run is finished if its last checkpoint is at the final epoch."""
    ckpts = checkpoint_paths(run_dir)
    cfg = with_overrides(load_experiment_config(args.root, args.experiment), overrides)
    from ..runs import checkpoint_epoch

    return checkpoint_epoch(ckpts[-1]) >= cfg.num_epochs - 1
