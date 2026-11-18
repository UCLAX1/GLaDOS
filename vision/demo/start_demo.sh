#!/usr/bin/env bash
# start_demo.sh — one command (or one click) to start the vision demo.
#
# Picks the right flags for the machine it runs on:
#   * ZED Mini plugged in (Jetson): --source zed --rotate $DEMO_ROTATE (default 180,
#     the ZED is mounted upside down; DEMO_ROTATE=0 when it's upright)
#   * otherwise: --source auto (laptop webcam), no rotation
#   * no screen (SSH session): --no-window, use the browser view instead
# and refuses to start a second copy (two copies fight over the camera and CPU).
#
# Usage (any directory):
#   vision/demo/start_demo.sh                    # pick flags automatically
#   vision/demo/start_demo.sh --fullscreen       # extra flags go to run_demo.py
#   DEMO_ROTATE=0 vision/demo/start_demo.sh      # ZED mounted upright
#   vision/demo/start_demo.sh --install-desktop  # add a desktop / app-menu icon
#   vision/demo/start_demo.sh --stop             # stop a running demo

set -u
DEMO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$DEMO_DIR/../.." && pwd)"
PY="$DEMO_DIR/.venv/bin/python"
# a python process whose command line STARTS with the venv python and runs run_demo.py
# (anchored, so shells or editors that merely mention run_demo.py don't count)
PATTERN='^[^ ]*demo/\.venv/bin/python[^ ]* ([^ ]+ )*[^ ]*run_demo\.py'

# ── Desktop icon ──────────────────────────────────────────────────────────────
if [ "${1:-}" = "--install-desktop" ]; then
    entry="[Desktop Entry]
Type=Application
Name=GLaDOS Vision Demo
Comment=All vision features on one camera
Exec=$DEMO_DIR/start_demo.sh --fullscreen
Icon=camera-web
Terminal=true
Categories=Science;"
    mkdir -p "$HOME/.local/share/applications"
    printf '%s\n' "$entry" > "$HOME/.local/share/applications/glados-vision-demo.desktop"
    if [ -d "$HOME/Desktop" ]; then
        printf '%s\n' "$entry" > "$HOME/Desktop/glados-vision-demo.desktop"
        chmod +x "$HOME/Desktop/glados-vision-demo.desktop"
        command -v gio >/dev/null && gio set "$HOME/Desktop/glados-vision-demo.desktop" metadata::trusted true 2>/dev/null
    fi
    echo "Installed: app menu + desktop icon 'GLaDOS Vision Demo' (opens fullscreen)."
    exit 0
fi

# ── Stop ──────────────────────────────────────────────────────────────────────
if [ "${1:-}" = "--stop" ]; then
    if pkill -f "$PATTERN"; then echo "Stopping the demo (it releases the camera)."; else echo "No demo running."; fi
    exit 0
fi

# ── Checks ────────────────────────────────────────────────────────────────────
if [ ! -x "$PY" ]; then
    echo "Not set up yet. Run once (needs internet):  python3 $DEMO_DIR/setup_demo.py"
    exit 1
fi

if running=$(pgrep -af "$PATTERN"); then
    echo "The demo is already running:"
    echo "  $running"
    echo "Open it in a browser (see the addresses it printed), or stop it first:  $0 --stop"
    exit 1
fi

# ── Flags for this machine ────────────────────────────────────────────────────
args=()
if command -v lsusb >/dev/null && lsusb | grep -q "2b03:f682"; then
    args+=(--source zed --rotate "${DEMO_ROTATE:-180}")
    echo "ZED Mini found: --source zed --rotate ${DEMO_ROTATE:-180}"
elif command -v lsusb >/dev/null && lsusb | grep -q "2b03:f681"; then
    echo "ZED Mini only half-detected (motion sensor, no video): flip its USB-C cable at the camera."
    echo "Starting with --source auto meanwhile."
    args+=(--source auto)
else
    args+=(--source auto)
fi

if [ "$(uname)" = "Linux" ] && [ -z "${DISPLAY:-}" ] && [ -z "${WAYLAND_DISPLAY:-}" ]; then
    echo "No screen in this session: browser view only (--no-window)."
    args+=(--no-window)
fi

cd "$REPO"
exec "$PY" "$DEMO_DIR/run_demo.py" "${args[@]}" "$@"
