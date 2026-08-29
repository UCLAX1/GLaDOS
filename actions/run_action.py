"""
run_action.py — run any action in actions/ on sim or hardware.

Sim:      mjpython actions/run.py nod
Hardware: GLADOS_HARDWARE=1 python3 actions/run.py nod
"""

import os, sys, importlib
sys.path.insert(0, ".")

if len(sys.argv) < 2:
    print("Usage: run_action.py <action_name>")
    print("  e.g. mjpython run_action.py nod")
    sys.exit(1)

action_name = sys.argv[1]
module = importlib.import_module(f"actions.scripts.{action_name}")
fn     = getattr(module, action_name)

if os.environ.get("GLADOS_HARDWARE"):
    from control.hardware_control import HardwareControl
    from actions.sequence import Sequence
    robot = HardwareControl()
    fn(Sequence(robot)).loop()
else:
    from sim.runner import GladosSim
    sim = GladosSim()
    with sim.launch() as viewer:
        fn(sim.sequence(viewer)).loop()
