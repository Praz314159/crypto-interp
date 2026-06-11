"""Backfill manifest.json for run dirs that predate manifest writing.

Records what is reconstructable from the files themselves (checkpoint
steps/count, config from the latest checkpoint, file-date range) plus a
user-supplied --source label for provenance that is not inferable
(where the run was trained). Skips runs that already have a manifest.

Usage:
    python -m scripts.backfill_manifests \\
        --runs 'experiments/004_p127/runs/dmodel_24_dmlp_32_seed*' \\
        --source runpod-a100
"""
from __future__ import annotations

import argparse
import datetime
import glob
import json
import re
from pathlib import Path

import torch

CKPT_RE = re.compile(r"checkpoint_(\d+)\.pt$")


def backfill_one(run_dir: Path, source: str, force: bool) -> str:
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists() and not force:
        return "skip (exists)"

    ckpts = sorted(
        (int(m.group(1)), p)
        for p in run_dir.glob("checkpoint_*.pt")
        if (m := CKPT_RE.search(p.name))
    )
    if not ckpts:
        return "skip (no checkpoints)"

    config = None
    try:
        payload = torch.load(ckpts[-1][1], map_location="cpu", weights_only=False)
        config = payload.get("config") if isinstance(payload, dict) else None
    except Exception:
        pass

    mtimes = [p.stat().st_mtime for _, p in ckpts]
    fmt = lambda t: datetime.datetime.fromtimestamp(
        t, tz=datetime.timezone.utc).isoformat(timespec="seconds")

    manifest = {
        "backfilled": True,
        "source": source,
        "config": config,
        "n_checkpoints": len(ckpts),
        "checkpoint_steps": [s for s, _ in ckpts],
        "file_dates": {"first": fmt(min(mtimes)), "last": fmt(max(mtimes))},
        "has_metrics": (run_dir / "metrics.pt").exists(),
        "has_losses": (run_dir / "losses.pt").exists(),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str))
    return f"wrote ({len(ckpts)} ckpts, source={source})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True,
                    help="Glob of run dirs, e.g. 'experiments/004_p127/runs/*'")
    ap.add_argument("--source", default="unknown",
                    help="Provenance label: colab | local-cpu | runpod-a100 | unknown")
    ap.add_argument("--force", action="store_true",
                    help="Overwrite existing manifests.")
    args = ap.parse_args()

    dirs = sorted(Path(d) for d in glob.glob(args.runs) if Path(d).is_dir())
    if not dirs:
        raise SystemExit(f"no run dirs match {args.runs!r}")
    for d in dirs:
        print(f"{d}: {backfill_one(d, args.source, args.force)}")


if __name__ == "__main__":
    main()
