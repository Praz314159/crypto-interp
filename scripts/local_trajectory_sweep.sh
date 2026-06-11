#!/usr/bin/env bash
# Retrain seeds locally with full checkpoint saving so we have trajectory data.
#
# The original Colab sweep at p=127 and p=181 stripped intermediate checkpoints
# to keep Drive bandwidth down — only the last checkpoint per seed reached
# disk. This script re-trains the missing-trajectory seeds locally on CPU
# (MPS doesn't support the float64 basis tensor), saving every 500 epochs to
# match the cadence of the seeds we already have.
#
# Idempotent: a seed with >10 checkpoints already on disk is skipped.
#
# Usage:
#     scripts/local_trajectory_sweep.sh [N_PARALLEL]
# Default N_PARALLEL=4. Set DEVICE=cuda to train on GPU (e.g. a cloud pod);
# multiple concurrent runs share one GPU fine at this model size.

set -u
N_PARALLEL="${1:-4}"
DEVICE="${DEVICE:-cpu}"
export DEVICE
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# Each entry is "exp seed"
TASKS=()
for s in $(seq 1 20); do
    name="experiments/004_p127/runs/dmodel_24_dmlp_32_seed${s}"
    if [ ! -d "$name" ] || [ "$(ls "$name"/checkpoint_*.pt 2>/dev/null | wc -l)" -le 10 ]; then
        TASKS+=("004_p127 $s")
    fi
done
for s in $(seq 1 20); do
    name="experiments/005_p181/runs/dmodel_24_dmlp_32_seed${s}"
    if [ ! -d "$name" ] || [ "$(ls "$name"/checkpoint_*.pt 2>/dev/null | wc -l)" -le 10 ]; then
        TASKS+=("005_p181 $s")
    fi
done

echo "[$(date +%H:%M:%S)] ${#TASKS[@]} retraining tasks pending, $N_PARALLEL in parallel"

train_one() {
    local exp="$1" seed="$2"
    local tag="dmodel_24_dmlp_32_seed${seed}"
    local logf="/tmp/traj_sweep_${exp}_seed${seed}.log"
    local t0=$(date +%s)
    # A partial dir is disposable by policy: clear any leftover checkpoints
    # so the retrain never merges with a previous run's files.
    rm -f "experiments/${exp}/runs/${tag}"/checkpoint_*.pt \
          "experiments/${exp}/runs/${tag}"/manifest.json
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    python -m scripts.train \
        --experiment "$exp" \
        --tag "$tag" \
        --seed-override "$seed" \
        --num-epochs 20000 \
        --save-every 500 \
        --metrics-every 100 \
        --device "$DEVICE" \
        --override d_model=24 --override d_mlp=32 \
        > "$logf" 2>&1
    local rc=$?
    local dt=$(( $(date +%s) - t0 ))
    if [ $rc -eq 0 ]; then
        echo "[$(date +%H:%M:%S)] done  $exp seed=$seed  (${dt}s)"
    else
        echo "[$(date +%H:%M:%S)] FAIL  $exp seed=$seed  (${dt}s, see $logf)"
    fi
}

export -f train_one
# Run in parallel via xargs.
printf '%s\n' "${TASKS[@]}" | xargs -P "$N_PARALLEL" -n 1 -I {} bash -c 'train_one $@' _ {}

echo "[$(date +%H:%M:%S)] sweep complete"
