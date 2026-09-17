"""
run_action.py — run any action in actions/ on sim or hardware.

Sim (loop):     mjpython actions/run_action.py nod
Sim (once):     mjpython actions/run_action.py nod --once
Hardware:       GLADOS_HARDWARE=1 python3 actions/run_action.py nod
"""

import os, sys, importlib
sys.path.insert(0, ".")

if len(sys.argv) < 2:
    print("Usage: run_action.py <action_name> [--once]")
    print("  e.g. mjpython actions/run_action.py nod")
    sys.exit(1)

action_name = sys.argv[1]
once = "--once" in sys.argv

module = importlib.import_module(f"actions.scripts.{action_name}")
fn     = getattr(module, action_name)

if os.environ.get("GLADOS_HARDWARE"):
    from control.hardware_control import HardwareControl
    from actions.sequence import Sequence
    robot = HardwareControl()
    seq = fn(Sequence(robot))
    seq.play() if once else seq.loop()
else:
    from sim.runner import GladosSim
    sim = GladosSim()
    with sim.launch() as viewer:
        seq = fn(sim.sequence(viewer))
        seq.play() if once else seq.loop()
