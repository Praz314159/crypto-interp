#!/usr/bin/env bash
# Train the p=127 and p=181 seeds with full trajectories (a checkpoint every
# 500 epochs, character energies every 100). Runs that already finished are
# skipped; unfinished ones are reported, not overwritten.
#
#     infra/runpod/trajectory_sweep.sh [JOBS]      # DEVICE=cuda on a GPU pod
set -euo pipefail
JOBS="${1:-4}"
DEVICE="${DEVICE:-cpu}"
for exp in 004_p127 005_p181; do
    crypto-interp sweep --experiment "$exp" --seeds 1..20 \
        --set d_model=24 --set d_mlp=32 \
        --num-epochs 20000 --save-every 500 --metrics-every 100 --device "$DEVICE" \
        --tag-format 'dmodel_{d_model}_dmlp_{d_mlp}_seed{seed}' \
        --skip-existing --jobs "$JOBS"
done
