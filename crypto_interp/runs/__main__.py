"""Inspect and maintain runs on disk.

python -m crypto_interp.runs ls [--experiment 003] [--root experiments] [--all]
python -m crypto_interp.runs rebuild-manifests RUN_DIR [RUN_DIR ...]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .index import DEFAULT_ROOT, query
from .manifest import rebuild_manifest

LS_COLUMNS = ["task", "p", "seed", "d_model", "d_mlp", "weight_decay", "latest_epoch", "excluded"]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m crypto_interp.runs")
    sub = ap.add_subparsers(dest="cmd", required=True)
    ls = sub.add_parser("ls", help="list runs")
    ls.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ls.add_argument("--experiment", help="experiment directory prefix, e.g. 003")
    ls.add_argument("--all", action="store_true", help="include excluded runs")
    rb = sub.add_parser("rebuild-manifests", help="upgrade manifests from latest checkpoints")
    rb.add_argument("runs", nargs="+", type=Path)
    args = ap.parse_args(argv)

    if args.cmd == "ls":
        table = query(args.root, experiment=args.experiment, include_excluded=args.all).table()
        if table.empty:
            print("no runs")
            return
        print(table[[c for c in LS_COLUMNS if c in table]].to_string())
        print(f"\n{len(table)} runs")
    else:
        for run in args.runs:
            if run.is_dir():
                rebuild_manifest(run)
                print(f"rebuilt {run}")


if __name__ == "__main__":
    main()
