#!/usr/bin/env bash
# Pod-side setup: clone the repo, install, and launch a sweep script in a
# detached tmux session. Run this once on a fresh RunPod GPU pod:
#
#     curl -fsSL https://raw.githubusercontent.com/Praz314159/crypto-interp/main/infra/runpod/bootstrap.sh | bash
#
# or copy it over and run it. Idempotent: re-running attaches to the
# existing sweep if one is already going.
#
# Env overrides:
#     REPO_URL    git URL to clone        (default: the public repo)
#     WORKDIR     where to clone          (default: /workspace)
#     N_PARALLEL  concurrent training processes sharing the GPU (default: 4)
#     SWEEP       sweep script under infra/runpod/ (default: trajectory_sweep.sh)
#     SWEEP_ARGS  extra arguments for it, after N_PARALLEL (e.g. a seed range)
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/Praz314159/crypto-interp.git}"
WORKDIR="${WORKDIR:-/workspace}"
N_PARALLEL="${N_PARALLEL:-4}"
SWEEP="${SWEEP:-trajectory_sweep.sh}"
SWEEP_ARGS="${SWEEP_ARGS:-}"

command -v tmux >/dev/null || apt-get install -y -q tmux
mkdir -p "$WORKDIR" && cd "$WORKDIR"
[ -d crypto-interp ] || git clone "$REPO_URL"
cd crypto-interp
git pull --ff-only
pip install -q -e .

echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none — sweep will run on CPU')"
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'" \
    && DEVICE=cuda || DEVICE=cpu

if tmux has-session -t sweep 2>/dev/null; then
    echo "sweep already running; attach with: tmux attach -t sweep"
    exit 0
fi

tmux new-session -d -s sweep \
    "DEVICE=$DEVICE infra/runpod/$SWEEP $N_PARALLEL $SWEEP_ARGS 2>&1 | tee $WORKDIR/sweep.log"
echo "sweep launched ($SWEEP, DEVICE=$DEVICE, N_PARALLEL=$N_PARALLEL)"
echo "follow progress:  tmux attach -t sweep    or    tail -f $WORKDIR/sweep.log"
