"""
Self-check for the qpos/ctrl indexing fix in mujoco_control.py.

Run: sim/venv/bin/python control/test_mujoco_control.py
"""

import os
import sys

import mujoco

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from control.mujoco_control import MujocoControl

XML_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "sim", "model", "glados.xml")


def demo():
    model = mujoco.MjModel.from_xml_path(XML_PATH)
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
    robot.move(**targets)
    for _ in range(2000):          # let the position actuators settle
        mujoco.mj_step(model, data)

    for joint, target in targets.items():
        got = robot.get_position(joint)
        assert abs(got - target) < 1.0, f"{joint}: expected {target}, got {got}"

    try:
        robot.get_position("bogus")
        assert False, "expected ValueError"
    except ValueError:
        pass

    print("OK")


if __name__ == "__main__":
    demo()
