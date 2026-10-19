"""Run selection shared by the ``measure``, ``figure`` and ``runs ls`` commands."""

from __future__ import annotations

import argparse
from pathlib import Path

from ..runs import DEFAULT_ROOT, Run, RunSet, query


def add_selection_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("run selection")
    g.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="experiments directory")
    g.add_argument("--experiment", help="experiment directory prefix, e.g. 003")
    g.add_argument(
        "--where",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="config filter, repeatable: d_mlp=20, seed=1..10, p=127,181",
    )
    g.add_argument(
        "--select",
        action="append",
        default=[],
        metavar="QUERY",
        help="a whole query, repeatable; the selection is their union, e.g. "
        "--select 'experiment=003 d_mlp=20 weight_decay=2' --select experiment=004",
    )
    g.add_argument(
        "--run",
        action="append",
        default=[],
        type=Path,
        metavar="DIR",
        help="an explicit run directory, repeatable (overrides the queries)",
    )
    g.add_argument("--all", action="store_true", help="include runs marked excluded")


def parse_value(text: str):
    """'3' → 3, '0.5' → 0.5, '1..10' → range(1, 11), 'a,b' → ['a', 'b'] (each parsed)."""
    if ".." in text:
        lo, hi = text.split("..")
        return range(int(lo), int(hi) + 1)
    if "," in text:
        return [parse_value(t) for t in text.split(",")]
    for kind in (int, float):
        try:
            return kind(text)
        except ValueError:
            pass
    return text


def parse_assignments(items: list[str]) -> dict:
    out = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"expected KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        out[key.strip()] = parse_value(value.strip())
    return out


def _query(args, fields: dict) -> RunSet:
    experiment = fields.pop("experiment", None)
    return query(
        args.root,
        experiment=experiment,
        include_excluded=args.all,
        **fields,
    )


def select(args) -> RunSet:
    if args.run:
        return RunSet([Run.open(p) for p in args.run])
    if args.select:
        found = {}
        for q in args.select:
            fields = parse_assignments([kv for kv in q.split() if not kv.startswith("experiment=")])
            # The experiment is a directory prefix ("003"), never a number.
            fields["experiment"] = next(
                (kv.split("=", 1)[1] for kv in q.split() if kv.startswith("experiment=")), None
            )
            for run in _query(args, fields):
                found[run.id] = run
        return RunSet(list(found.values()))
    return _query(args, {"experiment": args.experiment, **parse_assignments(args.where)})
