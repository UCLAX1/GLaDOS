"""
mujoco_control.py

MuJoCo implementation of ControlInterface.

Converts human-readable units (degrees, mm) to MuJoCo's native units
(radians, meters) and writes directly to data.ctrl.
"""

import math
import mujoco

from control.control_interface import ControlInterface


class MujocoControl(ControlInterface):

    # Actuator index in data.ctrl — must match order in glados.xml
    _INDEX = {
        "main_swivel": 0,
        "lower_arm":   1,
        "neck":        2,
        "head":        3,
        "eye":         4,
    }

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        self.model = model
        self.data  = data

    # ── ControlInterface implementation ────────────────────────────────────────

    def _send_batch(self, joints: dict[str, float]) -> None:
        """
        Write all joint commands in one command so everything move simultaneously and smoothly instead stilted 1-by-1

        Converts degrees → radians for rotation joints, mm → meters for eye.
        """
        ctrl = self.data.ctrl.copy()

        for joint, value in joints.items():
            idx = self._INDEX[joint]
            if joint == "eye":
                ctrl[idx] = value / 1000.0          # mm → m
            else:
                ctrl[idx] = math.radians(value)      # deg → rad

        self.data.ctrl[:] = ctrl                     # single write


    def get_position(self, joint: str) -> float:
        """
        Read the current joint position from the sim state.

        Returns degrees for rotation joints, mm for the eye. Raises ValueError for unknown joint names.
        """
        if joint not in self._INDEX:
            raise ValueError(
                f"Unknown joint {joint!r}. Valid joints: {list(self._INDEX)}"
            )
        
        idx = self._INDEX[joint]
        raw = self.data.qpos[idx]
        if joint == "eye":
            return raw * 1000.0                      # m → mm
        else:
            return math.degrees(raw)                 # rad → deg

    def shutdown(self) -> None:
        """Stop moving for sim"""
        pass
