"""
run_action.py — run any action in actions/ on sim or hardware.

Sim:      mjpython actions/run_action.py nod
Hardware: GLADOS_HARDWARE=1 python3 actions/run_action.py nod
"""

import os, sys
sys.path.insert(0, ".")

if len(sys.argv) < 2:
    print("Usage: run_action.py <action_name>")
    print("  e.g. mjpython run_action.py nod")
    sys.exit(1)

if os.environ.get("GLADOS_HARDWARE"):
    from control.hardware_control import HardwareControl
    HardwareControl().run_action(sys.argv[1], loop=True)
else:
    from sim.runner import GladosSim
    sim = GladosSim()
    with sim.launch():
        sim.robot.run_action(sys.argv[1], loop=True)
