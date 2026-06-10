#!/bin/bash
# Sequentially resume training for the 12 seeds that didn't fully grok by epoch 6500.
# Each gets up to 13500 more epochs (total ~20000), with early stopping once test
# loss has been below 1e-5 for 500 consecutive epochs.

set -e
ADDITIONAL_EPOCHS=13500
EARLY_STOP=1e-5
PATIENCE=500

LAGGARDS=(2 3 4 5 6 7 8 9 10 12 13 15)

for seed in "${LAGGARDS[@]}"; do
    ckpt="runs/mul_sweep_seed${seed}/checkpoint_006499.pt"
    if [[ ! -f "$ckpt" ]]; then
        echo "MISSING checkpoint for seed $seed: $ckpt"; continue
    fi
    echo
    echo "=============================================="
    echo "Resuming seed $seed (laggard)"
    echo "=============================================="
    python3 -u train.py \
        --task mul --p 113 --frac-train 0.3 --seed "$seed" \
        --num-epochs "$ADDITIONAL_EPOCHS" \
        --resume "$ckpt" \
        --early-stop-loss "$EARLY_STOP" \
        --early-stop-patience "$PATIENCE" \
        --log-every 500 --save-every 2000 --metrics-every 50
done

echo
echo "All laggards extended. Run analysis with:"
echo "  python3 -m interp.analyze_order_classes --runs runs/mul_sweep_seed{0..15}"
