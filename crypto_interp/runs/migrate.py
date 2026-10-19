"""Migrate run directories to schema-2 checkpoints, quarantining strays.

Usage::

    python -m crypto_interp.runs.migrate experiments/*/runs/*            # dry run: report only
    python -m crypto_interp.runs.migrate experiments/*/runs/* --apply --report report.md

For each run directory:

1. **Lineages.** Checkpoints written by one training run share their loss
   history: an earlier checkpoint's ``train_losses`` is a prefix of a later
   one's (a resumed run continues the same history). Checkpoints are grouped
   into lineages on that test.
2. **Primary lineage.** The lineage whose history matches ``losses.pt``, or
   failing that the one with the most checkpoints.
3. **Task check.** The dataset rebuilt from the stored task spec must
   reproduce the losses the run recorded. If it doesn't, or the earliest
   schema-1 files never stored a task, every task with the model's vocabulary
   size is tried; the one that reproduces the recorded losses is used. A run
   for which no task does is blocked and left untouched.
4. **Quarantine.** Checkpoints from other lineages, and side files
   (``losses.pt``, ``metrics.pt``, ``fine_grained.pt``) that belong to a
   different run, move to ``<run>/_strays/``. Nothing is deleted.
5. **Exclusion.** If the directory name says ``seed<N>`` but the primary
   lineage was trained with another seed, the run is marked excluded in its
   manifest, with the reason, so populations skip it.
6. **Rewrite.** Primary-lineage checkpoints that are schema 1, or whose task
   was corrected, are rewritten as schema 2. A file is replaced only after the
   rewritten model's forward pass matches the original exactly.
7. **Record.** ``manifest.json`` gets the primary lineage's training config
   and a ``migrations`` entry. Re-running is a no-op.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import json
import math
import os
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F

from .. import data
from .checkpoint import Checkpoint, checkpoint_paths, load_checkpoint, save_checkpoint

STRAYS = "_strays"

# Placeholder task for schema-1 files that never recorded one; always replaced
# by a task verified against the recorded losses (step 3), or the run is blocked.
_UNKNOWN_TASK = {"task": "<unknown>"}

# Max |log(recomputed / recorded)| over train and test loss. Losses are recorded
# before each epoch's step and checkpoints saved after it, which leaves real runs
# within ~0.2; a wrong task or split is off by orders of magnitude.
LOSS_TOLERANCE = 1.0


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
    to_rewrite: list[Path]
    task: str | None  # verified task for the primary lineage; None = blocked
    task_changed: bool
    notes: list[str] = field(default_factory=list)
    test_loss_before: float | None = None
    test_loss_after: float | None = None

    @property
    def blocked(self) -> bool:
        return self.task is None


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


def _losses(ck: Checkpoint, task: str) -> tuple[float, float] | None:
    """(train, test) cross-entropy of ``ck`` on ``task``'s dataset, or None if the
    task's vocabulary doesn't fit the model."""
    ds = dataclasses.replace(ck.task, task=task).build()
    if ds.vocab_size != ck.model_config.d_vocab:
        return None
    with torch.no_grad():
        logits = ck.build_model()(ds.inputs)[:, -1, : ds.n_answer_tokens].double()
    return tuple(
        F.cross_entropy(logits[m], ds.labels[m]).item() for m in (ds.train_mask, ds.test_mask)
    )


def _reproduces(ck: Checkpoint, task: str) -> bool:
    got = _losses(ck, task)
    if got is None or not ck.train_losses:
        return False
    # Entry epoch+1 is computed on exactly the weights saved at `epoch`, when it exists.
    i = min(ck.epoch + 1, len(ck.train_losses) - 1)
    recorded = (ck.train_losses[i], ck.test_losses[i])
    return all(abs(math.log(g / r)) < LOSS_TOLERANCE for g, r in zip(got, recorded, strict=True))


def _verified_task(ck: Checkpoint, stored_known: bool) -> str | None:
    if stored_known and _reproduces(ck, ck.task.task):
        return ck.task.task
    matches = [t for t in data.available_tasks() if _reproduces(ck, t)]
    return matches[0] if len(matches) == 1 else None


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


def survey(run: Path) -> RunReport | None:
    paths = checkpoint_paths(run)
    if not paths:
        return None
    ckpts = {p: load_checkpoint(p, v1_defaults=_UNKNOWN_TASK) for p in paths}
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
    latest_after = max(lin.paths, key=lambda p: ckpts[p].epoch)

    latest = ckpts[latest_after]
    stored = latest.task.task
    task = _verified_task(latest, stored_known=stored != _UNKNOWN_TASK["task"])
    task_changed = task is not None and task != stored
    if task is None:
        notes.append(f"no task reproduces the recorded losses (stored: {stored}); not migrated")
    elif task_changed:
        notes.append(f"task {stored} → {task} (reproduces the recorded losses)")

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
        to_rewrite=[p for p in lin.paths if ckpts[p].source_schema == 1 or task_changed],
        task=task,
        task_changed=task_changed,
        notes=notes,
    )
    if task is not None and report.latest_before != report.latest_after:
        report.test_loss_before = _losses(ckpts[report.latest_before], task)[1]
        report.test_loss_after = _losses(ckpts[report.latest_after], task)[1]
    return report


def _rewrite(path: Path, task: str) -> None:
    """Rewrite one checkpoint as schema 2 with ``task``, verified by forward pass."""
    ck = load_checkpoint(path, v1_defaults=_UNKNOWN_TASK)
    ck.task = dataclasses.replace(ck.task, task=task)
    ck.train_config = {**ck.train_config, "task": task}
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


def needs_action(r: RunReport) -> bool:
    if r.blocked:
        return False
    return bool(r.strays or r.to_rewrite or (r.excluded and not r.excluded_recorded))


def apply(report: RunReport) -> None:
    if report.blocked:
        raise ValueError(f"{report.run}: blocked, not migrating ({report.notes})")
    run = report.run
    if report.strays:
        (run / STRAYS).mkdir(exist_ok=True)
        for p in report.strays:
            shutil.move(str(p), str(run / STRAYS / p.name))
    for p in report.to_rewrite:
        _rewrite(p, report.task)

    manifest_path = run / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest["config"] = load_checkpoint(report.latest_after).train_config
    if report.excluded:
        manifest["excluded"] = report.excluded
    manifest.setdefault("migrations", []).append(
        {
            "tool": "crypto_interp.runs.migrate",
            "date": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
            "quarantined": [p.name for p in report.strays],
            "rewritten": len(report.to_rewrite),
            "notes": report.notes,
        }
    )
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str))


def render(reports: list[RunReport]) -> str:
    lines = [
        "# Run migration report",
        "",
        f"{len(reports)} runs; {sum(len(r.to_rewrite) for r in reports)} checkpoints to rewrite; "
        f"{sum(1 for r in reports if r.strays)} runs with strays; "
        f"{sum(1 for r in reports if r.excluded)} excluded; "
        f"{sum(1 for r in reports if r.blocked)} blocked.",
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
    ap.add_argument("--report", type=Path, help="write the markdown report here")
    args = ap.parse_args(argv)

    reports = []
    for run in sorted(p for p in args.runs if p.is_dir()):
        r = survey(run)
        if r is None:
            continue
        reports.append(r)
        status = "blocked" if r.blocked else "needs migration" if needs_action(r) else "ok"
        print(f"{run}: {r.n_checkpoints} ckpts, {len(r.lineages)} lineage(s), {status}")
        if args.apply and needs_action(r):
            apply(r)

    text = render(reports)
    if args.report:
        args.report.write_text(text)
    print()
    print(text)


if __name__ == "__main__":
    main()
