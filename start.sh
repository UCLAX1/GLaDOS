#!/bin/bash
# start.sh — start GLaDOS. Same command on both machines:
#
#     ./start.sh
#
#   Jetson → camera + vision. Serves video (/stream) and what it sees (/status) on :8081.
#   Mac    → LLM server + voice + listening + sim + GUI, reading vision from the Jetson.
#
# Start the Jetson first, then the Mac. Ctrl-C (or closing the window) stops everything.
#
# Options (environment variables):
#   JETSON_IP=100.73.92.3   Jetson address (Mac side). Empty = run without vision.
#   WEB_PORT=8081           vision port (both sides must agree; Mac: VISION_DEMO_PORT)
#   GLADOS_GAP_MS=480       silence before an utterance counts as finished
#   LLAMA_ARGS="-t 8"       extra llama-server flags
set -e
cd "$(dirname "$0")"

if [ "$(uname -s)" = "Darwin" ]; then
    # ── Mac ───────────────────────────────────────────────────────────────────
    [ -x listening/venv/bin/python ] || { echo "Not set up yet — run ./setup.sh first"; exit 1; }
    export JETSON_IP="${JETSON_IP-100.73.92.3}"
    export VISION_DEMO_PORT="${WEB_PORT:-8081}"

    # A previous run that did not exit cleanly would hold the port and the model
    pkill -f "llama-server.*--port 8080" 2>/dev/null || true
    pkill -f "speech/glados_daemon.py"   2>/dev/null || true

    echo "GLaDOS — Mac   (vision from: ${JETSON_IP:-none})"
    exec listening/venv/bin/python gui_launch.py
else
    # ── Jetson ────────────────────────────────────────────────────────────────
    VENV="vision/demo/.venv/bin/python"
    WEB_PORT="${WEB_PORT:-8081}"
    [ -x "$VENV" ] || { echo "Not set up yet — run ./setup.sh first"; exit 1; }

    # Only one process can hold the ZED. Clear out anything left from last time.
    pkill -f "vision/demo/run_demo.py" 2>/dev/null || true
    pkill -f "vision_server.py"        2>/dev/null || true
    sleep 1

    IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
    echo "GLaDOS — Jetson"
    echo "  video:      http://$IP:$WEB_PORT/stream"
    echo "  perception: http://$IP:$WEB_PORT/status"
    echo "  On the Mac: JETSON_IP=$IP ./start.sh"
    exec "$VENV" vision/demo/run_demo.py --source zed --no-window --web "$WEB_PORT"
fi
