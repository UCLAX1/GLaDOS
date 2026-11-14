#!/bin/bash
# start_mac.sh — start the full GLaDOS pipeline on the Mac
set -e
cd "$(dirname "$0")"

JETSON_IP="${JETSON_IP:-100.73.92.3}"

echo "======================================"
echo "  GLaDOS Mac Startup"
echo "  Jetson: $JETSON_IP"
echo "======================================"

# install pyaudio if missing
source listening/venv/bin/activate
python3 -c "import pyaudio" 2>/dev/null || pip install -q pyaudio

# Terminal 1: llama.cpp server
echo "[1/3] Starting llama.cpp server..."
llama-server -m central_ai/benchmarking_tools/Bonsai-8B-Q1_0.gguf --port 8080 &
LLAMA_PID=$!

# wait for server to be ready
echo "      Waiting for model to load..."
until curl -sf http://localhost:8080/health >/dev/null 2>&1; do sleep 1; done
echo "      llama.cpp ready"

# Terminal 2: speech daemon
echo "[2/3] Starting speech daemon..."
python3 speech/glados_daemon.py &
DAEMON_PID=$!
sleep 1

# cleanup on exit
trap "echo; echo 'Shutting down...'; kill $LLAMA_PID $DAEMON_PID 2>/dev/null; exit" INT TERM

# Terminal 3: main pipeline (foreground)
echo "[3/3] Starting GLaDOS pipeline..."
echo "      Speak into the mic. Ctrl-C to stop."
echo ""
JETSON_IP=$JETSON_IP python3 run_glados.py

kill $LLAMA_PID $DAEMON_PID 2>/dev/null
