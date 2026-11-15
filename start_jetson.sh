#!/bin/bash
# start_jetson.sh — start vision demo + perception server on the Jetson
set -e
cd "$(dirname "$0")"

VENV="vision/demo/.venv/bin/python"
WEB_PORT="${WEB_PORT:-8081}"

echo "======================================"
echo "  GLaDOS Jetson Startup"
echo "  Demo: http://$(hostname -I | awk '{print $1}'):$WEB_PORT"
echo "======================================"

# setup venv if not built yet
if [ ! -f "$VENV" ]; then
    echo "[setup] Building demo venv (first time, ~5 min)..."
    python3 vision/demo/setup_demo.py
fi

# cleanup on exit
cleanup() {
    echo; echo "Shutting down..."
    kill $DEMO_PID $SERVER_PID 2>/dev/null
    exit
}
trap cleanup INT TERM

# vision demo — owns the ZED, serves the video (/stream) AND the perception
# data (/status) that the Mac's VisionClient polls. vision_server.py is no
# longer started: it would fight the demo for the camera and see nothing.
echo "[1/1] Starting vision demo on :$WEB_PORT..."
$VENV vision/demo/run_demo.py --source zed --no-window --web $WEB_PORT &
DEMO_PID=$!
SERVER_PID=""

echo ""
echo "  Camera stream:  http://$(hostname -I | awk '{print $1}'):$WEB_PORT/stream"
echo "  Perception API: http://$(hostname -I | awk '{print $1}'):$WEB_PORT/status"
echo "  Ctrl-C to stop."
echo ""

wait $DEMO_PID
