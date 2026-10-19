"""Migrate run directories to schema-2 checkpoints, quarantining strays.

Usage::

    python -m crypto_interp.runs.migrate experiments/*/runs/*            # dry run: report only
    python -m crypto_interp.runs.migrate experiments/*/runs/* --apply \\
        --v1-default task=mul --report migration_report.md

For each run directory:

1. **Lineages.** Checkpoints written by one training run share their loss
   history: an earlier checkpoint's ``train_losses`` is a prefix of a later
   one's (a resumed run continues the same history). Checkpoints are grouped
   into lineages on that test.
2. **Primary lineage.** The lineage whose history matches ``losses.pt``, or
   failing that the one with the most checkpoints.
3. **Quarantine.** Checkpoints from other lineages, and side files
   (``losses.pt``, ``metrics.pt``, ``fine_grained.pt``) that belong to a
   different run, move to ``<run>/_strays/``. Nothing is deleted.
4. **Exclusion.** If the directory name says ``seed<N>`` but the primary
   lineage was trained with another seed, the run is marked excluded in its
   manifest, with the reason, so populations skip it.
5. **Rewrite.** Schema-1 checkpoints in the primary lineage are rewritten as
   schema 2, replacing a file only after the rewritten model's forward pass
   matches the original exactly.
6. **Record.** ``manifest.json`` gets the primary lineage's training config
   and a ``migration`` entry. Re-running is a no-op.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F

from .checkpoint import Checkpoint, checkpoint_paths, load_checkpoint, save_checkpoint

STRAYS = "_strays"
SIDE_FILES = ("losses.pt", "metrics.pt", "fine_grained.pt")


def _is_prefix(a: list, b: list) -> bool:
    n = min(len(a), len(b))
    return n > 0 and a[:n] == b[:n]


@dataclass
class Lineage:
    paths: list[Path] = field(default_factory=list)
    history: list[float] = field(default_factory=list)  # longest train_losses seen
    seed: int | None = None


@dataclass
class RunReport:
    run: Path
    n_checkpoints: int
    lineages: list[Lineage]
    primary: int
    strays: list[Path]
    excluded: str | None
    excluded_recorded: bool
    latest_before: Path
    latest_after: Path
    to_convert: list[Path]
    notes: list[str] = field(default_factory=list)
    test_loss_before: float | None = None
    test_loss_after: float | None = None


def _group(ckpts: dict[Path, Checkpoint]) -> list[Lineage]:
    lineages: list[Lineage] = []
    for path, ck in ckpts.items():
        for lin in lineages:
            if _is_prefix(lin.history, ck.train_losses):
                lin.paths.append(path)
                if len(ck.train_losses) > len(lin.history):
                    lin.history = ck.train_losses
                break
        else:
            lineages.append(Lineage([path], list(ck.train_losses), ck.task.seed))
    return lineages


def _test_loss(ck: Checkpoint) -> float:
    ds = ck.task.build()
    with torch.no_grad():
        logits = ck.build_model()(ds.inputs[ds.test_mask])[:, -1, : ds.n_answer_tokens]
    return F.cross_entropy(logits.double(), ds.labels[ds.test_mask]).item()


def _side_file_strays(run: Path, primary: Lineage, ckpts: dict[Path, Checkpoint]) -> list[Path]:
    out = []
    losses = run / "losses.pt"
    if losses.exists():
        hist = list(torch.load(losses, weights_only=False)["train_losses"])
        if not _is_prefix(hist, primary.history):
            out.append(losses)
    metrics = run / "metrics.pt"
    if metrics.exists():
        seed = torch.load(metrics, weights_only=False).get("config", {}).get("seed")
        if seed is not None and seed != primary.seed:
            out.append(metrics)
    fine = run / "fine_grained.pt"
    first = min(primary.paths, key=lambda p: ckpts[p].epoch)
    if fine.exists() and ckpts[first].epoch == 0:
        # fine_grained W_E at epoch e is taken before step e; checkpoint_000000 is
        # saved after step 0, so it must equal the epoch-1 snapshot.
        fg = torch.load(fine, weights_only=False)
        at_1 = (fg["epochs"] == 1).nonzero()
        if len(at_1) and not torch.equal(
            fg["W_E"][int(at_1[0])], ckpts[first].model_state["embed.W_E"].cpu()
        ):
            out.append(fine)
    return out


def survey(run: Path, v1_defaults: dict | None = None) -> RunReport | None:
    paths = checkpoint_paths(run)
    if not paths:
        return None
    ckpts = {p: load_checkpoint(p, v1_defaults=v1_defaults) for p in paths}
    lineages = _group(ckpts)

    primary = max(range(len(lineages)), key=lambda i: len(lineages[i].paths))
    notes = []
    if (run / "losses.pt").exists():
        hist = list(torch.load(run / "losses.pt", weights_only=False)["train_losses"])
        matching = [i for i, lin in enumerate(lineages) if _is_prefix(hist, lin.history)]
        if matching:
            primary = matching[0]
        else:
            notes.append("losses.pt matches no checkpoint lineage")
    lin = lineages[primary]

    strays = [p for i, other in enumerate(lineages) if i != primary for p in other.paths]
    strays += _side_file_strays(run, lin, ckpts)

    manifest_path = run / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    excluded = manifest.get("excluded")
    excluded_recorded = excluded is not None
    m = re.search(r"seed(\d+)$", run.name)
    if excluded is None and m and int(m.group(1)) != lin.seed:
        excluded = (
            f"directory is named seed{m.group(1)} but its run was trained with seed {lin.seed}"
        )
    first_epoch = min(ckpts[p].epoch for p in lin.paths)
    if first_epoch != 0:
        notes.append(f"earliest checkpoint is epoch {first_epoch}")
    if ckpts[lin.paths[0]].source_schema == 1 and "task" not in _raw_config(lin.paths[0]):
        notes.append(f"task taken from v1 default {v1_defaults}")

    latest_after = max(lin.paths, key=lambda p: ckpts[p].epoch)
    report = RunReport(
        run=run,
        n_checkpoints=len(paths),
        lineages=lineages,
        primary=primary,
        strays=strays,
        excluded=excluded,
        excluded_recorded=excluded_recorded,
        latest_before=paths[-1],
        latest_after=latest_after,
        to_convert=[p for p in lin.paths if ckpts[p].source_schema == 1],
        notes=notes,
    )
    if report.latest_before != report.latest_after:
        report.test_loss_before = _test_loss(ckpts[report.latest_before])
        report.test_loss_after = _test_loss(ckpts[report.latest_after])
    return report


def _raw_config(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False).get("config", {})


def _convert(path: Path, v1_defaults: dict | None) -> None:
    """Rewrite one schema-1 checkpoint as schema 2, verified by forward pass."""
    ck = load_checkpoint(path, v1_defaults=v1_defaults)
    inputs = ck.task.build().inputs
    with torch.no_grad():
        before = ck.build_model()(inputs)
    tmp = path.with_suffix(".pt.tmp")
    save_checkpoint(ck, tmp)
    rewritten = load_checkpoint(tmp)
    with torch.no_grad():
        after = rewritten.build_model()(inputs)
    if rewritten.source_schema != 2 or not torch.equal(before, after):
        tmp.unlink()
        raise RuntimeError(f"rewritten checkpoint does not reproduce {path}")
    os.replace(tmp, path)


def apply(report: RunReport, v1_defaults: dict | None = None) -> None:
    run = report.run
    if report.strays:
        (run / STRAYS).mkdir(exist_ok=True)
        for p in report.strays:
            shutil.move(str(p), str(run / STRAYS / p.name))
    for p in report.to_convert:
        _convert(p, v1_defaults)

    manifest_path = run / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    latest = load_checkpoint(report.latest_after)
    manifest["config"] = latest.train_config
    if report.excluded:
        manifest["excluded"] = report.excluded
    manifest.setdefault("migrations", []).append(
        {
            "tool": "crypto_interp.runs.migrate",
            "date": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
            "quarantined": [p.name for p in report.strays],
            "converted_to_schema_2": len(report.to_convert),
            "notes": report.notes,
        }
    )
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str))


def needs_action(r: RunReport) -> bool:
    return bool(r.strays or r.to_convert or (r.excluded and not r.excluded_recorded))


def render(reports: list[RunReport]) -> str:
    lines = [
        "# Run migration report",
        "",
        f"{len(reports)} runs; {sum(len(r.to_convert) for r in reports)} schema-1 checkpoints "
        f"to rewrite; {sum(1 for r in reports if r.strays)} runs with strays; "
        f"{sum(1 for r in reports if r.excluded)} excluded.",
        "",
        "## Runs with strays, exclusions or notes",
        "",
        "| run | ckpts | lineages | quarantined | latest before → after | test loss before → after "
        "| excluded / notes |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in reports:
        if not (r.strays or r.excluded or r.notes):
            continue
        moved = ", ".join(p.name for p in r.strays) or "–"
        latest = (
            f"{r.latest_before.name} → {r.latest_after.name}"
            if r.latest_before != r.latest_after
            else "unchanged"
        )
        loss = (
            f"{r.test_loss_before:.3g} → {r.test_loss_after:.3g}"
            if r.test_loss_before is not None
            else "–"
        )
        extra = "; ".join(([r.excluded] if r.excluded else []) + r.notes) or "–"
        name = f"{r.run.parent.parent.name}/{r.run.name}"
        lines.append(
            f"| {name} | {r.n_checkpoints} | {len(r.lineages)} | {moved} | {latest} | {loss} "
            f"| {extra} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("runs", nargs="+", type=Path, help="run directories")
    ap.add_argument("--apply", action="store_true", help="move strays and rewrite checkpoints")
    ap.add_argument(
        "--v1-default",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="config value for schema-1 checkpoints that lack it (e.g. task=mul)",
    )
    ap.add_argument("--report", type=Path, help="write the markdown report here")
    args = ap.parse_args(argv)

    v1_defaults = dict(kv.split("=", 1) for kv in args.v1_default) or None
    reports = []
    for run in sorted(p for p in args.runs if p.is_dir()):
        r = survey(run, v1_defaults)
        if r is None:
            continue
        reports.append(r)
        status = "ok" if not needs_action(r) else "needs migration"
        print(f"{run}: {r.n_checkpoints} ckpts, {len(r.lineages)} lineage(s), {status}")
        if args.apply and needs_action(r):
            apply(r, v1_defaults)

    text = render(reports)
    if args.report:
        args.report.write_text(text)
    print()
    print(text)


if __name__ == "__main__":
    main()
