"""Import sweep results from Drive layout into the experiments/ layout.

The Colab sweep script writes runs to ``crypto_sweep/p${p}_seed${s}/`` containing
``checkpoint_*.pt``, ``losses.pt``, ``metrics.pt``. ``Session.from_run`` and
``crypto_interp/analysis/`` expect the canonical
``experiments/00X_p${p}/runs/dmodel_24_dmlp_32_seed${s}/`` layout. This script
bridges them — idempotent, safe to rerun.

Usage:
    # From an unzipped Drive folder:
    python -m scripts.import_drive_sweep --drive ~/Downloads/crypto_sweep

    # Or from a zip downloaded from Drive:
    python -m scripts.import_drive_sweep --drive ~/Downloads/crypto_sweep.zip

    # Dry run (show what would happen):
    python -m scripts.import_drive_sweep --drive ~/Downloads/crypto_sweep --dry-run
"""
from __future__ import annotations

import argparse
import re
import shutil
import tempfile
import zipfile
from pathlib import Path

# Maps prime -> experiment directory name.
PRIME_TO_EXPERIMENT = {
    127: "004_p127",
    181: "005_p181",
    113: "003_dmodel_sweep_p113",
}

# Canonical per-run filename inside the experiments layout.
TAG = "dmodel_24_dmlp_32_seed{seed}"

# Match Drive-layout directory names like "p127_seed5".
DRIVE_DIR_RE = re.compile(r"^p(?P<p>\d+)_seed(?P<seed>\d+)$")


def expand_zip_if_needed(path: Path) -> tuple[Path, tempfile.TemporaryDirectory | None]:
    """If ``path`` is a zip, extract to a temp dir and return that. Caller must
    keep the returned TemporaryDirectory alive (use try/finally)."""
    if path.is_dir():
        return path, None
    if path.suffix == ".zip":
        tmp = tempfile.TemporaryDirectory(prefix="drive_sweep_")
        with zipfile.ZipFile(path) as zf:
            zf.extractall(tmp.name)
        # The zip may contain a single top-level dir (e.g. "crypto_sweep/"); descend.
        contents = list(Path(tmp.name).iterdir())
        root = contents[0] if len(contents) == 1 and contents[0].is_dir() else Path(tmp.name)
        return root, tmp
    raise ValueError(f"--drive must be a directory or .zip, got {path}")


def import_one(src: Path, p: int, seed: int, repo_root: Path, *, dry_run: bool) -> str:
    """Copy one Drive-layout dir into the experiments layout. Returns a status string."""
    if p not in PRIME_TO_EXPERIMENT:
        return f"  skip {src.name}: no experiment dir registered for p={p}"
    exp_dir = repo_root / "experiments" / PRIME_TO_EXPERIMENT[p]
    if not exp_dir.exists():
        return f"  skip {src.name}: experiment dir missing ({exp_dir})"
    dst = exp_dir / "runs" / TAG.format(seed=seed)

    has_metrics = (src / "metrics.pt").exists()
    ckpts = sorted(src.glob("checkpoint_*.pt"))
    if not has_metrics or not ckpts:
        return f"  skip {src.name}: incomplete (metrics={has_metrics}, ckpts={len(ckpts)})"

    if dst.exists() and (dst / "metrics.pt").exists():
        return f"  exists {dst.relative_to(repo_root)} (skip)"

    if dry_run:
        return f"  would copy {src.name} -> {dst.relative_to(repo_root)} ({len(ckpts)} ckpts)"

    dst.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        if f.is_file():
            shutil.copy2(f, dst / f.name)
    return f"  copy {src.name} -> {dst.relative_to(repo_root)} ({len(ckpts)} ckpts)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drive", required=True,
                    help="Path to crypto_sweep directory (or zip of it) from Drive.")
    ap.add_argument("--repo-root", default=".",
                    help="Repo root containing experiments/ (default: cwd).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Show what would happen without copying.")
    args = ap.parse_args()

    drive_path = Path(args.drive).expanduser().resolve()
    repo_root = Path(args.repo_root).resolve()

    if not drive_path.exists():
        raise SystemExit(f"--drive path does not exist: {drive_path}")

    src_root, _tmp = expand_zip_if_needed(drive_path)
    try:
        per_prime: dict[int, list[str]] = {}
        for d in sorted(src_root.iterdir()):
            if not d.is_dir():
                continue
            m = DRIVE_DIR_RE.match(d.name)
            if not m:
                continue
            p = int(m.group("p"))
            seed = int(m.group("seed"))
            status = import_one(d, p, seed, repo_root, dry_run=args.dry_run)
            per_prime.setdefault(p, []).append(status)

        for p in sorted(per_prime):
            print(f"\np = {p}  ({len(per_prime[p])} candidate seeds)")
            for s in per_prime[p]:
                print(s)
        if args.dry_run:
            print("\n(dry run — no files copied; remove --dry-run to apply)")
    finally:
        if _tmp is not None:
            _tmp.cleanup()


if __name__ == "__main__":
    main()
