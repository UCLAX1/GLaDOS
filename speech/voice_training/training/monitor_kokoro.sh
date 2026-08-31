#!/usr/bin/env bash
# monitor_kokoro.sh — Check training status on gpu_zflow
# Usage: bash monitor_kokoro.sh [lines]
LINES="${1:-40}"
REMOTE="gpu_zflow"

echo "=== GPU utilization ==="
ssh "$REMOTE" "nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader"

echo ""
echo "=== Latest log (last $LINES lines) ==="
ssh "$REMOTE" "tail -n $LINES /tmp/train_kokoro.log 2>/dev/null || echo '(log not found)'"

echo ""
echo "=== Checkpoints saved ==="
ssh "$REMOTE" "ls -lh ~/voice_training/checkpoints/kokoro_robotic/ 2>/dev/null || echo '(none yet)'"

echo ""
echo "=== Training process ==="
ssh "$REMOTE" "pgrep -a -f 'train_finetune' 2>/dev/null || echo '(no training process found)'"
