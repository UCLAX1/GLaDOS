#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# deploy_to_gpu.sh  —  Sync the full training pipeline to a GPU server and
#                      launch fine-tuning. Run from your Mac terminal.
#
#   bash deploy_to_gpu.sh [remote_host]   # default: gpu_zflow
#
# Audio data is NOT synced here — copy it separately before training:
#   rsync -avz sorted/robotic/ gpu_zflow:~/voice_training/data/wavs_robotic/
#   rsync -avz sorted/human/   gpu_zflow:~/voice_training/data/wavs_human/
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

REMOTE="${1:-gpu_zflow}"
REMOTE_DIR="~/voice_training"
D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== [1/4] Syncing scripts to $REMOTE:$REMOTE_DIR ==="
ssh "$REMOTE" "mkdir -p $REMOTE_DIR/prepared_data"

scp "$D/training/setup_kokoro.sh" \
    "$D/training/train_kokoro.py" \
    "$D/training/prepare_glados.py" \
    "$D/training/extract_voicepack.py" \
    "$D/training/config_glados.yml" \
    "$D/training/test_glados_inference.py" \
    "$REMOTE:$REMOTE_DIR/"

scp "$D/prepared_data/train_list.txt" \
    "$D/prepared_data/val_list.txt" \
    "$D/prepared_data/OOD_texts.txt" \
    "$REMOTE:$REMOTE_DIR/prepared_data/"

echo "=== [2/4] Running one-time setup on $REMOTE ==="
echo "      (Downloads ~1.1 GB of base model weights — expect 5-20 min)"
ssh "$REMOTE" "cd $REMOTE_DIR && bash setup_kokoro.sh 2>&1 | tee /tmp/kokoro_setup.log"

echo "=== [3/4] Launching training (nohup → /tmp/train_kokoro.log) ==="
ssh "$REMOTE" "
  cd $REMOTE_DIR
  source kokoro_training/.venv/bin/activate 2>/dev/null || source venv/bin/activate 2>/dev/null || true
  nohup python train_kokoro.py > /tmp/train_kokoro.log 2>&1 &
  echo \"Training PID: \$!\"
"

echo "=== [4/4] Done. Monitor with: ==="
echo "  Live log:  ssh $REMOTE 'tail -f /tmp/train_kokoro.log'"
echo "  GPU util:  ssh $REMOTE 'watch -n5 nvidia-smi'"
echo "  Checkpts:  ssh $REMOTE 'ls -lh ~/voice_training/kokoro_training/logs/glados/'"
