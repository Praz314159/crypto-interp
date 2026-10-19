"""Every code block in README.md and docs/ runs, and its shown output is right.

Fenced blocks run in document order, one fresh process per file, in a scratch
directory holding a copy of the reference run at ``tests/fixtures/``:

- ``bash``: run as one script, stopping at the first failing command
  (``crypto-interp`` is on the PATH);
- ``python``: executed in the file's shared namespace;
- ``pycon``: a doctest against that same namespace, so shown output is checked.

A block whose info string contains ``no-run`` (installation, long training)
is skipped. Other languages (``text``, ...) are not run.
"""

from __future__ import annotations

import doctest
import os
import re
import shutil
import subprocess
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
DOCS = [REPO / "README.md", *sorted((REPO / "docs").glob("*.md"))]
FENCE = re.compile(r"^```([^\n`]*)\n(.*?)^```", re.M | re.S)
RUNNABLE = {"bash", "python", "pycon"}


def blocks(text: str):
    """``(language, code, line)`` for each runnable fenced block."""
    for m in FENCE.finditer(text):
        info = m.group(1).split()
        if info and info[0] in RUNNABLE and "no-run" not in info:
            yield info[0], m.group(2), text.count("\n", 0, m.start()) + 2


def run_doc(path: Path) -> None:
    """Run ``path``'s blocks in order in the current directory; raise on failure."""
    import matplotlib

    matplotlib.use("Agg")
    module = types.ModuleType("doc")  # a real module, so dataclasses defined in a doc work
    sys.modules["doc"] = module
    namespace = module.__dict__
    parser = doctest.DocTestParser()
    runner = doctest.DocTestRunner(optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE)
    for lang, code, line in blocks(path.read_text()):
        where = f"{path.name}:{line}"
        if lang == "bash":
            subprocess.run(["bash", "-e", "-c", code], check=True, stdout=subprocess.DEVNULL)
        elif lang == "python":
            exec(compile(code, where, "exec"), namespace)
        else:
            test = parser.get_doctest(code, namespace, where, str(path), line)
            if runner.run(test, clear_globs=False).failed:
                raise SystemExit(f"doctest failed at {where}")
            namespace.update(test.globs)  # DocTest runs on a copy


@pytest.mark.parametrize("doc", DOCS, ids=lambda p: p.relative_to(REPO).as_posix())
def test_doc_runs(doc, tmp_path):
    run = REPO / "tests" / "fixtures" / "mul_p113_d24_m32_seed1"
    shutil.copytree(run, tmp_path / "tests" / "fixtures" / run.name, ignore=_no_cache)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    shim = bin_dir / "crypto-interp"
    shim.write_text(f'#!/bin/sh\nexec "{sys.executable}" -m crypto_interp.cli "$@"\n')
    shim.chmod(0o755)
    env = {
        **os.environ,
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "PYTHONPATH": f"{REPO}{os.pathsep}{os.environ.get('PYTHONPATH', '')}",
        "MPLBACKEND": "Agg",
    }
    proc = subprocess.run(
        [sys.executable, __file__, str(doc)], cwd=tmp_path, env=env, capture_output=True, text=True
    )
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-4000:]


def test_every_measure_is_documented():
    from crypto_interp import measures

    headings = [
        line
        for line in (REPO / "docs" / "measures.md").read_text().splitlines()
        if line.startswith("### ")
    ]
    ours = [n for n, m in measures.REGISTRY.items() if m.fn.__module__.startswith("crypto_interp.")]
    missing = [n for n in ours if not any(f"`{n}`" in h for h in headings)]
    assert not missing, f"add these to docs/measures.md: {missing}"


def _no_cache(directory, names):
    return [n for n in names if n == "measures"]


if __name__ == "__main__":
    run_doc(Path(sys.argv[1]))
