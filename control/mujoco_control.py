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

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        self.model = model
        self.data  = data

        self._act  = {}   # joint name → actuator index (for data.ctrl)
        self._qpos = {}   # joint name → qpos address  (for data.qpos)

        # glados.xml names actuators/joints as "{name}_actuator" / "{name}_joint"
        for name in self.LIMITS:
            aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{name}_actuator")
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,    f"{name}_joint")
            if aid == -1:
                raise ValueError(f"No actuator named {name!r} in model")
            if jid == -1:
                raise ValueError(f"No joint named {name!r} in model")
            self._act[name]  = aid
            self._qpos[name] = model.jnt_qposadr[jid]

    # ── ControlInterface implementation ────────────────────────────────────────

    def _send_batch(self, joints: dict[str, float]) -> None:
        """
        Write all joint commands in one command so everything move simultaneously and smoothly instead stilted 1-by-1

        Converts degrees → radians for rotation joints, mm → meters for eye.
        """
        for joint, value in joints.items():
            idx = self._act[joint]
            if joint == "eye":
                self.data.ctrl[idx] = value / 1000.0        # mm → m
            else:
                self.data.ctrl[idx] = math.radians(value)   # deg → rad


    def get_position(self, joint: str) -> float:
        """
        Read the current joint position from the sim state.

        Returns degrees for rotation joints, mm for the eye. Raises ValueError for unknown joint names.
        """
        if joint not in self._qpos:
            raise ValueError(
                f"Unknown joint {joint!r}. Valid joints: {list(self._qpos)}"
            )

        idx = self._qpos[joint]
        raw = self.data.qpos[idx]
        if joint == "eye":
            return raw * 1000.0                      # m → mm
        else:
            return math.degrees(raw)                 # rad → deg

    def shutdown(self) -> None:
        """No-op — sim has no motors to power down."""
        pass
