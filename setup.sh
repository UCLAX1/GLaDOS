#!/bin/bash
# setup.sh — one-time setup. Same command on both machines:
#
#     ./setup.sh
#
# It detects where it is running:
#   Mac    → brain, voice, listening, sim, GUI   (listening/venv)
#   Jetson → camera + vision                      (vision/demo/.venv)
#
#   ./setup.sh vision   (Mac) also set up webcam vision on the Mac, used
#                       automatically when the Jetson is not reachable
#
# Safe to re-run; it only installs what is missing.
set -e
cd "$(dirname "$0")"

say()  { printf "\n\033[1m%s\033[0m\n" "$*"; }
ok()   { printf "  \033[32m✓\033[0m %s\n" "$*"; }
warn() { printf "  \033[33m!\033[0m %s\n" "$*"; }

if [ "$(uname -s)" = "Darwin" ]; then
    # ── Mac ───────────────────────────────────────────────────────────────────
    say "GLaDOS setup — Mac (brain + voice + GUI)"

    # System libraries: PortAudio (mic/speaker), espeak-ng (Kokoro phonemes), llama.cpp
    if command -v brew >/dev/null 2>&1; then
        for pkg in portaudio espeak-ng; do
            brew list "$pkg" >/dev/null 2>&1 && ok "$pkg" || { echo "  installing $pkg…"; brew install "$pkg"; }
        done
        command -v llama-server >/dev/null 2>&1 && ok "llama-server" \
            || { echo "  installing llama.cpp…"; brew install llama.cpp; }
    else
        warn "Homebrew not found — install portaudio, espeak-ng and llama.cpp yourself"
    fi

    # Python environment
    PY=""
    for c in python3.12 python3.11 python3.10 python3; do
        command -v "$c" >/dev/null 2>&1 && { PY="$c"; break; }
    done
    [ -n "$PY" ] || { echo "No python3 found"; exit 1; }
    if [ ! -x listening/venv/bin/python ]; then
        echo "  creating listening/venv with $PY…"
        "$PY" -m venv listening/venv
    fi
    ok "venv: listening/venv ($(listening/venv/bin/python --version 2>&1))"
    echo "  installing Python packages (first run takes a few minutes)…"
    listening/venv/bin/python -m pip install -q --upgrade pip
    listening/venv/bin/python -m pip install -q -r requirements.txt
    ok "Python packages"

    # Model
    GGUF="central_ai/benchmarking_tools/Bonsai-8B-Q1_0.gguf"
    if [ -f "$GGUF" ]; then
        ok "model: $GGUF"
    else
        echo "  downloading Bonsai-8B (1.1 GB)…"
        listening/venv/bin/python -m pip install -q huggingface_hub
        listening/venv/bin/python -c "from huggingface_hub import hf_hub_download as d; d('prism-ml/Bonsai-8B-gguf','Bonsai-8B-Q1_0.gguf',local_dir='central_ai/benchmarking_tools')" \
            || warn "download failed — put Bonsai-8B-Q1_0.gguf at $GGUF"
    fi

    # Optional: webcam vision on this Mac (fallback when the Jetson is not there)
    if [ "${1:-}" = "vision" ]; then
        say "Webcam vision on this Mac"
        "$PY" vision/demo/setup_demo.py
    elif [ -x vision/demo/.venv/bin/python ]; then
        ok "webcam vision (fallback when the Jetson is unreachable)"
    else
        warn "no webcam vision on this Mac — without the Jetson the demo runs without vision."
        echo "      To use the Mac's webcam instead:  ./setup.sh vision"
    fi

    say "Done. Start with:  ./start.sh"
else
    # ── Jetson ────────────────────────────────────────────────────────────────
    say "GLaDOS setup — Jetson (camera + vision)"
    python3 vision/demo/setup_demo.py
    say "Done. Start with:  ./start.sh"
fi
