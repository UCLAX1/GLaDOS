"""
run_action.py — run any action in actions/ on sim or hardware.

Sim (loop):     mjpython actions/run_action.py nod
Sim (once):     mjpython actions/run_action.py nod --once
Hardware:       GLADOS_HARDWARE=1 python3 actions/run_action.py nod
"""

import os, sys
sys.path.insert(0, ".")

if len(sys.argv) < 2:
    print("Usage: run_action.py <action_name> [--once]")
    print("  e.g. mjpython actions/run_action.py nod")
    sys.exit(1)

loop = "--once" not in sys.argv

if os.environ.get("GLADOS_HARDWARE"):
    from control.hardware_control import HardwareControl
    HardwareControl().run_action(sys.argv[1], loop=loop)
else:
    from sim.runner import GladosSim
    sim = GladosSim()
    with sim.launch():
        sim.robot.run_action(sys.argv[1], loop=loop)
