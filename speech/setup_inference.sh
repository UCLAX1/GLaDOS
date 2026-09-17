#!/bin/bash
# setup_inference.sh — one-time setup for GLaDOS TTS inference.
# Works on Mac, Linux, and Jetson Orin (Ubuntu 22).
#
# Usage:
#   bash setup_inference.sh
#
# What it does:
#   1. Detects platform (Mac / Linux x86 / Jetson ARM)
#   2. Installs espeak-ng (required by misaki G2P)
#   3. Installs Python deps (uses NVIDIA Jetson PyTorch wheel on Jetson)
#   4. Pre-downloads model files from HuggingFace into ~/glados_tts/
#      (optional — glados_tts.py auto-downloads on first run anyway)
set -e

DEST="$HOME/glados_tts"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$DEST"

# ── 1. Detect platform ────────────────────────────────────────────────────
OS="$(uname -s)"
ARCH="$(uname -m)"
IS_JETSON=false

if [ "$OS" = "Linux" ] && [ "$ARCH" = "aarch64" ]; then
    # Check for Jetson-specific file
    if [ -f /etc/nv_tegra_release ] || [ -f /proc/device-tree/model ]; then
        IS_JETSON=true
    fi
fi

echo "==> Platform: $OS $ARCH (Jetson: $IS_JETSON)"

# ── 2. Install espeak-ng ─────────────────────────────────────────────────
echo "==> Checking espeak-ng..."
if ! command -v espeak-ng &>/dev/null; then
    if [ "$OS" = "Darwin" ]; then
        brew install espeak-ng
    elif command -v apt-get &>/dev/null; then
        if [ "$(id -u)" = "0" ]; then
            apt-get install -y espeak-ng
        else
            sudo apt-get install -y espeak-ng
        fi
    else
        echo "  WARNING: Could not install espeak-ng automatically."
        echo "  Install it manually: https://github.com/espeak-ng/espeak-ng"
    fi
else
    echo "  espeak-ng already installed."
fi

# ── 3. Install Python deps ────────────────────────────────────────────────
echo "==> Installing Python deps..."

if [ "$IS_JETSON" = true ]; then
    echo "  Jetson detected — using NVIDIA PyTorch wheel."
    echo "  If PyTorch is not yet installed, follow:"
    echo "  https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/"
    echo "  Then re-run this script. Skipping torch install."
    # Install everything except torch (managed by NVIDIA on Jetson)
    pip install kokoro "misaki[en]" inflect soundfile huggingface_hub \
        --extra-index-url https://pypi.ngc.nvidia.com --quiet || \
    pip install kokoro "misaki[en]" inflect soundfile huggingface_hub --quiet
else
    pip install kokoro "misaki[en]" inflect soundfile huggingface_hub --quiet
fi

# ── 4. Pre-download model files (optional — script auto-downloads anyway) ─
echo "==> Pre-downloading model files from HuggingFace..."
python3 -c "
from huggingface_hub import hf_hub_download
import shutil, os
dest = os.path.expanduser('$DEST')
for f in ['glados_kmodel.pth', 'glados.pt']:
    path = hf_hub_download(repo_id='yifanfang/glados-kokoro', filename=f, repo_type='model')
    shutil.copy(path, os.path.join(dest, f))
    print(f'  cached {f}')
"

# config ships in the repo
cp "$SCRIPT_DIR/config_kokoro.json" "$DEST/config_kokoro.json"
cp "$SCRIPT_DIR/glados_tts.py"      "$DEST/glados_tts.py"

echo ""
echo "Setup complete. Test with:"
echo "  python3 $SCRIPT_DIR/glados_tts.py 'The cake is a lie.'"
