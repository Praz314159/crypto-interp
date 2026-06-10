#!/usr/bin/env bash
# Create an on-demand Secure Cloud GPU pod for the training sweep.
#
# Requires runpodctl configured with your API key:
#     runpodctl config --apiKey "$RUNPOD_API_KEY"
#
# Flag names vary slightly across runpodctl versions; if this errors,
# launch from the web console instead (see README.md).
set -euo pipefail

GPU_TYPE="${GPU_TYPE:-NVIDIA A100 80GB PCIe}"
IMAGE="${IMAGE:-runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04}"
NAME="${NAME:-crypto-interp-sweep}"

runpodctl create pod \
    --name "$NAME" \
    --gpuType "$GPU_TYPE" \
    --imageName "$IMAGE" \
    --containerDiskSize 40 \
    --secureCloud \
    --ports "22/tcp"

echo "Pod requested. Find the SSH command under Pods → $NAME → Connect."
