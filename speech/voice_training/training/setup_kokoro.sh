#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# setup_kokoro.sh  —  One-time setup on gpu_zflow
#
# What it does:
#   1. Installs Python deps into the existing venv
#   2. Clones StyleTTS2 (training code + ASR/JDC utility models)
#   3. Downloads Kokoro-82M weights (327 MB) from HuggingFace
#   4. Downloads StyleTTS2-LibriTTS checkpoint (771 MB) — needed for StyleEncoder
#   5. Creates output + work directories
#
# Safe to re-run — skips already-done steps.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

VENV="$HOME/voice_training/venv"
WORK="$HOME/voice_training/kokoro_work"
OUT="$HOME/voice_training/checkpoints/kokoro_robotic"

log() { echo "[setup] $*"; }

# ── 1. Activate venv ──────────────────────────────────────────────────────────
log "Activating venv: $VENV"
source "$VENV/bin/activate"
python --version

# ── 2. Install Python deps ────────────────────────────────────────────────────
log "Installing Python deps..."
pip install -q --upgrade \
    "kokoro>=0.9.2" \
    "misaki[en]" \
    phonemizer \
    soundfile \
    librosa \
    accelerate \
    munch \
    pyyaml \
    huggingface_hub \
    einops \
    SoundFile \
    tqdm

# espeak-ng (needed by phonemizer for G2P)
if ! command -v espeak-ng &>/dev/null; then
    log "Installing espeak-ng..."
    sudo apt-get install -y espeak-ng 2>/dev/null || \
        apt-get install -y espeak-ng 2>/dev/null || \
        log "WARNING: could not install espeak-ng — install manually if phonemizer errors"
fi

# ── 3. Clone StyleTTS2 ────────────────────────────────────────────────────────
log "Setting up StyleTTS2..."
mkdir -p "$WORK"
cd "$WORK"

if [ ! -d StyleTTS2 ]; then
    git clone --depth=1 https://github.com/yl4579/StyleTTS2.git
    log "StyleTTS2 cloned."
else
    log "StyleTTS2 already present — skipping clone."
fi

cd StyleTTS2
# Install StyleTTS2 requirements (skip torch — already installed)
pip install -q -r requirements.txt 2>/dev/null || true

# ── 4. Download model weights ─────────────────────────────────────────────────
log "Downloading model weights..."

python3 - <<'PYEOF'
import sys, os
from pathlib import Path
from huggingface_hub import hf_hub_download

work = Path(os.environ['HOME']) / 'voice_training/kokoro_work'

# Kokoro-82M weights
kokoro_dir = work / 'kokoro'
kokoro_dir.mkdir(parents=True, exist_ok=True)

if not (kokoro_dir / 'kokoro-v1_0.pth').exists():
    print("  [HF] Downloading Kokoro-82M weights (327 MB)...")
    hf_hub_download('hexgrad/Kokoro-82M', 'kokoro-v1_0.pth',
                    local_dir=str(kokoro_dir))
else:
    print("  [HF] Kokoro weights already present.")

if not (kokoro_dir / 'config.json').exists():
    hf_hub_download('hexgrad/Kokoro-82M', 'config.json',
                    local_dir=str(kokoro_dir))

# StyleTTS2-LibriTTS checkpoint (provides StyleEncoder weights)
libritts_dir = work / 'libritts'
libritts_dir.mkdir(parents=True, exist_ok=True)
dest_pth  = libritts_dir / 'Models' / 'LibriTTS' / 'epochs_2nd_00020.pth'
dest_yml  = libritts_dir / 'Models' / 'LibriTTS' / 'config.yml'

if not dest_pth.exists():
    print("  [HF] Downloading StyleTTS2-LibriTTS checkpoint (771 MB)...")
    hf_hub_download('yl4579/StyleTTS2-LibriTTS',
                    'Models/LibriTTS/epochs_2nd_00020.pth',
                    local_dir=str(libritts_dir))
else:
    print("  [HF] LibriTTS checkpoint already present.")

if not dest_yml.exists():
    hf_hub_download('yl4579/StyleTTS2-LibriTTS',
                    'Models/LibriTTS/config.yml',
                    local_dir=str(libritts_dir))

print("  All weights ready.")
PYEOF

# ── 5. Create output directories ──────────────────────────────────────────────
mkdir -p "$OUT"
mkdir -p "$WORK/data"

log "=== Setup complete. Run 'python ~/voice_training/train_kokoro.py' to start training. ==="
