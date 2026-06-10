# RunPod sweep infrastructure

Run training sweeps on a rented GPU pod instead of Colab. Pods have no
session timeout, real SSH, and a persistent filesystem for the lifetime of
the pod — checkpoints just land on disk and get rsynced back.

Nothing here is account-specific; API keys live in your environment or
`.env` (gitignored), never in the repo.

## One-time setup

1. Create a RunPod account and add billing credit.
2. Settings → API Keys → create a key; export it as `RUNPOD_API_KEY`.
3. Settings → SSH Keys → add your `~/.ssh/id_ed25519.pub`.
4. (Optional, for `launch.sh`) install the CLI: `brew install runpod/runpodctl/runpodctl`
   and run `runpodctl config --apiKey "$RUNPOD_API_KEY"`.

## Pod spec

- **GPU:** one A100 PCIe (Secure Cloud, on-demand — not spot/interruptible).
  Full-rate fp64 covers the float64 basis tensors; the models themselves are
  tiny, so the win is running several training processes concurrently.
- **Template:** `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`
  (any recent PyTorch+CUDA template works).
- **Disk:** 40 GB container disk. No network volume needed — results are
  rsynced back and the pod is terminated.

Launch from the web console (Pods → Deploy) with the spec above, or run
`infra/runpod/launch.sh`.

## Running the trajectory sweep

```bash
# 0. Get the pod's SSH command from the console (Pods → Connect), e.g.:
POD="root@<pod-id>.runpod.io"   # adjust to the real host/port

# 1. (Optional) upload existing run dirs so finished seeds are skipped —
#    the sweep script is idempotent on >10-checkpoint runs:
rsync -az --relative experiments/./004_p127/runs experiments/./005_p181/runs \
    "$POD":/workspace/crypto-interp/experiments/

# 2. Bootstrap + launch (clones repo, installs, starts tmux session 'sweep'):
ssh "$POD" 'bash -s' < infra/runpod/bootstrap.sh

# 3. Monitor:
ssh "$POD" tail -f /workspace/sweep.log

# 4. When done, pull results back:
rsync -az "$POD":/workspace/crypto-interp/experiments/004_p127/runs/ experiments/004_p127/runs/
rsync -az "$POD":/workspace/crypto-interp/experiments/005_p181/runs/ experiments/005_p181/runs/

# 5. Terminate the pod in the console (or runpodctl remove pod <id>).
#    A stopped-but-not-terminated pod still bills for disk.
```

## Cost expectations

The d_model=24 models are kernel-launch-bound, not FLOPs-bound: ~4–10 min
per 20k-epoch seed on GPU, several processes sharing one card. A 40-seed
sweep is ~1–3 h ≈ **$2–7** at A100 on-demand rates. Benchmark one seed
first (`N_PARALLEL=1`, watch the log) before trusting a larger sweep's ETA.
