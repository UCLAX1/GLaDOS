"""
Self-check for the qpos/ctrl indexing fix in mujoco_control.py.

Run: sim/venv/bin/python control/test_mujoco_control.py
"""

import math
import os
import sys

import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from control.mujoco_control import MujocoControl

XML_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "sim", "model", "glados.xml")


def demo():
    import params  # sim/params.py, put on path by control_interface
    model = params.load_model()
    data = mujoco.MjData(model)
    robot = MujocoControl(model, data)

    assert not hasattr(MujocoControl, "_INDEX"), "_INDEX should be gone"

    targets = {
        "main_swivel": 45.0,
        "lower_arm":   30.0,
        "tilt":        5.0,
        "nod":         -10.0,
        "eye":         1.5,
    }
    robot._command(**targets)
    for _ in range(2000):          # let the position actuators settle
        mujoco.mj_step(model, data)

    for joint, target in targets.items():
        got = robot.get_position(joint)
        assert abs(got - target) < robot.ARRIVE_TOL, f"{joint}: expected {target}, got {got}"

    # same command in deg and rad must write the same ctrl
    robot._command(main_swivel=30.0, upper_arm=-10.0, lower_arm=40.0)
    deg_ctrl = data.ctrl.copy()
    robot._command(main_swivel=math.radians(30), upper_arm=math.radians(-10), lower_arm=math.radians(40), unit="rad")
    assert np.allclose(deg_ctrl, data.ctrl), "deg and rad commands disagree"

    # move() blocks until it has arrived, no manual stepping
    robot.move(main_swivel=-40.0, lower_arm=20.0)
    assert abs(robot.get_position("main_swivel") + 40) < robot.ARRIVE_TOL
    assert abs(robot.get_position("lower_arm") - 20) < robot.ARRIVE_TOL

    try:
        robot.get_position("bogus")
        assert False, "expected ValueError"
    except ValueError:
        pass

    print("OK")


if __name__ == "__main__":
    demo()
