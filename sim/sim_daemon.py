#!/usr/bin/env python3
"""
sim_daemon.py — persistent MuJoCo sim viewer for GLaDOS.

Keep this running alongside run_glados.py. The viewer stays open; gestures
from run_glados.py land in /tmp/glados_sim_queue/ and play on the live sim.

Usage:
    mjpython sim/sim_daemon.py
"""

import importlib
import json
import os
import pathlib
import sys
import time

if os.environ.get("GLADOS_HARDWARE"):
    print("⛔  sim_daemon is SIM ONLY — GLADOS_HARDWARE is set. Aborting.")
    sys.exit(1)

import mujoco
import mujoco.viewer

sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))
from control.mujoco_control import MujocoControl
from actions.sequence import Sequence, TICK_RATE

QUEUE_DIR = pathlib.Path("/tmp/glados_sim_queue")
MODEL_XML  = pathlib.Path(__file__).parent / "model" / "glados.xml"

QUEUE_DIR.mkdir(exist_ok=True)

model = mujoco.MjModel.from_xml_path(str(MODEL_XML))
data  = mujoco.MjData(model)
robot = MujocoControl(model, data)


def _pop_action() -> "str | None":
    """Return the oldest queued action name and delete its file, or None."""
    files = sorted(QUEUE_DIR.glob("*.json"))
    if not files:
        return None
    f = files[0]
    try:
        payload = json.loads(f.read_text())
        return payload.get("action")
    except Exception:
        return None
    finally:
        f.unlink(missing_ok=True)


def _play_action(action_name: str, tick_fn, running_fn) -> None:
    """Import and play one action script on the running sim."""
    try:
        mod = importlib.import_module(f"actions.scripts.{action_name}")
        fn  = getattr(mod, action_name)
        seq = fn(Sequence(robot, tick_fn=tick_fn, running_fn=running_fn))
        seq.play()
        print(f"[sim] played {action_name!r}")
    except Exception as e:
        print(f"[sim] action {action_name!r} failed: {e}")


print("[sim] viewer starting — polling /tmp/glados_sim_queue/ for gestures")

with mujoco.viewer.launch_passive(model, data) as viewer:
    def tick():
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(TICK_RATE)

    while viewer.is_running():
        action = _pop_action()
        if action:
            _play_action(action, tick, viewer.is_running)
        else:
            tick()
