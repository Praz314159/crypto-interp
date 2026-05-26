#!/usr/bin/env bash
# One-command lattice-variation pipeline: import Drive results + run analysis.
#
# Usage:
#     scripts/run_lattice_analysis.sh ~/Downloads/crypto_sweep
#     scripts/run_lattice_analysis.sh ~/Downloads/crypto_sweep.zip
#
# Idempotent: re-importing already-imported seeds is a no-op.
set -euo pipefail

if [ $# -eq 0 ]; then
    echo "Usage: $0 <path-to-crypto_sweep-dir-or-zip>"
    exit 1
fi
DRIVE_PATH="$1"

# Resolve repo root from script location.
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

echo "=== [1/2] importing Drive sweep into experiments/ ==="
python -m scripts.import_drive_sweep --drive "$DRIVE_PATH" --repo-root "$REPO_ROOT"

echo
echo "=== [2/2] running lattice_variation analysis ==="
python -m crypto_interp.analysis.lattice_variation \
    --out-dir "$REPO_ROOT/experiments/lattice_variation"

echo
echo "Done. Results in experiments/lattice_variation/"
ls -la "$REPO_ROOT/experiments/lattice_variation/"
