"""The ``crypto-interp`` command.

::

    crypto-interp train   --experiment 003_dmodel_sweep_p113 --seed 3 --set d_mlp=20
    crypto-interp sweep   --experiment 004_p127 --seeds 1..20 --set d_model=24 --grid d_mlp=32,128
    crypto-interp runs ls --experiment 003
    crypto-interp measure kernel_noise --experiment 003 --where d_mlp=20 --out table.csv
    crypto-interp figure  regimes --experiment 00 --where d_model=24 --out regimes.png
    crypto-interp figure  list
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .select import add_selection_args, parse_assignments, select
from .train import add_train_args, sweep_main, train_main


def _measure_main(args) -> None:
    from .. import measures

    if args.name not in measures.REGISTRY:
        raise SystemExit(f"unknown measure {args.name!r}; known: {sorted(measures.REGISTRY)}")
    m = measures.REGISTRY[args.name]
    params = parse_assignments(args.param)
    pop = select(args)
    if isinstance(m, measures.RunMeasure):
        table = m.over(pop, **params)
    else:
        table = m.over(pop, checkpoints=args.checkpoints, **params)
    if args.out is None:
        print(table.to_string(index=False) if args.wide else _results_only(table, pop, args))
        return
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.suffix == ".parquet":
        table.to_parquet(args.out)
    else:
        table.to_csv(args.out, index=False)
    print(f"{len(table)} rows" + (f" → {args.out}" if args.out else ""))


def _results_only(table, pop, args) -> str:
    """The table without the run config columns, for reading in a terminal."""
    config = {"experiment", "epoch"}.union(*(r.config() for r in pop))
    keep = ["run"] + (["epoch"] if args.checkpoints == "all" else [])
    keep += [c for c in table.columns if c not in config and c != "run"]
    return table[keep].to_string(index=False)


def _figure_main(args) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from .figures import NO_RUNS, RECIPES

    if args.name == "list":
        width = max(map(len, RECIPES))
        for name, (help_, _) in RECIPES.items():
            print(f"{name:<{width}}  {help_}")
        return
    if args.name not in RECIPES:
        raise SystemExit(f"unknown figure {args.name!r}; `crypto-interp figure list`")
    if args.out is None:
        raise SystemExit("--out is required")
    pop = None if args.name in NO_RUNS else select(args)
    if pop is not None and not len(pop):
        raise SystemExit("the selection matches no runs")
    fig = RECIPES[args.name][1](pop, args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    print(f"wrote {args.out}")


def _runs_main(args) -> None:
    from ..runs import rebuild_manifest

    if args.runs_cmd == "ls":
        table = select(args).table()
        cols = ["task", "p", "seed", "d_model", "d_mlp", "weight_decay", "latest_epoch", "excluded"]
        print(table[[c for c in cols if c in table]].to_string() if len(table) else "no runs")
        print(f"\n{len(table)} runs")
    else:
        for run in args.dirs:
            rebuild_manifest(run)
            print(f"rebuilt {run}")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="crypto-interp", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="train one run (or resume one)")
    add_train_args(t)
    t.add_argument("--seed", type=int)
    t.add_argument("--tag", help="run directory name (default: from the overrides and seed)")
    t.add_argument("--resume", help="checkpoint to continue; --num-epochs counts extra epochs")
    t.set_defaults(func=train_main)

    s = sub.add_parser("sweep", help="train seeds × a grid of overrides in parallel")
    add_train_args(s)
    s.add_argument("--seeds", required=True, help="1..20 or 1,2,5")
    s.add_argument(
        "--grid",
        action="append",
        default=[],
        metavar="KEY=V1,V2",
        help="sweep a config field over values, repeatable",
    )
    s.add_argument("--jobs", type=int, default=4, help="concurrent training processes")
    s.add_argument("--threads-per-job", type=int, default=1)
    s.add_argument(
        "--skip-existing",
        action="store_true",
        help="skip runs whose directory already has checkpoints",
    )
    s.add_argument(
        "--tag-format",
        help="run name template over the overrides and seed, "
        "e.g. 'dmodel_{d_model}_dmlp_{d_mlp}_seed{seed}'",
    )
    s.set_defaults(func=sweep_main)

    r = sub.add_parser("runs", help="list runs or rebuild their manifests")
    rsub = r.add_subparsers(dest="runs_cmd", required=True)
    ls = rsub.add_parser("ls")
    add_selection_args(ls)
    rb = rsub.add_parser("rebuild-manifests")
    rb.add_argument("dirs", nargs="+", type=Path)
    r.set_defaults(func=_runs_main)

    m = sub.add_parser("measure", help="compute a measure over runs into a table")
    m.add_argument("name")
    add_selection_args(m)
    m.add_argument("--checkpoints", choices=["latest", "all"], default="latest")
    m.add_argument("--param", action="append", default=[], metavar="KEY=VALUE")
    m.add_argument("--out", type=Path, help=".csv or .parquet (default: print)")
    m.add_argument("--wide", action="store_true", help="also print the run config columns")
    m.set_defaults(func=_measure_main)

    f = sub.add_parser("figure", help="draw a figure (`figure list` shows them)")
    f.add_argument("name")
    add_selection_args(f)
    f.add_argument("--out", type=Path)
    f.add_argument("--k", type=int, help="character, for single-character figures")
    f.add_argument("--a", type=int, help="first input, for the walkthrough")
    f.add_argument("--b", type=int, help="second input, for the walkthrough")
    f.add_argument("--title")
    f.add_argument("--dpi", type=int, default=130)
    f.set_defaults(func=_figure_main)
    return ap


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)
