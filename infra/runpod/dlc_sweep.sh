#!/usr/bin/env bash
# Experiment 007: the Discrete Log Clock configuration (d_model 128, d_mlp 512,
# 40k epochs) over 30 seeds, on the units (mul_units) and, as a control, on
# the full grid with 0 (mul). Finished runs are skipped; unfinished ones are
# reported, not overwritten.
#
#     infra/runpod/dlc_sweep.sh [JOBS] [SEEDS]     # DEVICE=cuda on a GPU pod
set -euo pipefail
JOBS="${1:-4}"
SEEDS="${2:-1..30}"
DEVICE="${DEVICE:-cpu}"
crypto-interp sweep --experiment 007_dlc_p113 --seeds "$SEEDS" \
    --grid task=mul_units,mul --device "$DEVICE" \
    --tag-format '{task}_seed{seed}' --skip-existing --jobs "$JOBS"
