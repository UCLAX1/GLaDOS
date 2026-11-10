"""
mujoco_control.py

MuJoCo implementation of ControlInterface.

Converts human-readable units (degrees, mm) to MuJoCo's native units
(radians, meters) and writes directly to data.ctrl.
"""

import math
import time
import mujoco
import mujoco.viewer

from control.control_interface import ControlInterface  # first: puts sim/ on sys.path for params
import params
from actions.sequence import TICK_RATE


# logical joint name → name used in glados.xml (the rest match 1:1)
XML_NAME = {"main_swivel": "upper_arm_yaw", "upper_arm": "upper_arm_pitch"}


class MujocoControl(ControlInterface):

    def __init__(self, model: mujoco.MjModel | None = None, data: mujoco.MjData | None = None) -> None:
        self.model  = params.load_model() if model is None else model
        self.data   = mujoco.MjData(self.model) if data is None else data
        self.viewer = None   # set by launch()

        self._act  = {}   # joint name → actuator index (for data.ctrl)
        self._qpos = {}   # joint name → qpos address  (for data.qpos)

        # glados.xml names actuators/joints as "{xml_name}_actuator" / "{xml_name}_joint"
        # Joints not yet modelled are skipped so actions using the rest still run.
        missing = []
        for name in self.LIMITS:
            xml = XML_NAME.get(name, name)
            aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{xml}_actuator")
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT,    f"{xml}_joint")
            if aid == -1 or jid == -1:
                missing.append(name)
                continue
            self._act[name]  = aid
            self._qpos[name] = self.model.jnt_qposadr[jid]
        if missing:
            print(f"[sim] joints not in model (skipped): {missing}")

    # ── Sim: viewer + stepping ─────────────────────────────────────────────────

    def launch(self):
        """Open the passive viewer (use `with robot.launch():`; macOS needs mjpython)."""
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        return self.viewer

    def step(self, n: int = 1) -> None:
        for _ in range(n):
            mujoco.mj_step(self.model, self.data)

    def settle(self, steps: int = 4000) -> None:
        """Step physics until the position actuators have reached their targets."""
        self.step(steps)

    def head_position(self):
        """head_mount site position (m) in the world frame, read from the sim."""
        mujoco.mj_forward(self.model, self.data)  # site_xpos is stale until a forward pass
        sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "head_mount")
        return self.data.site_xpos[sid].copy()

    def _tick(self) -> None:
        self.step()
        if self.viewer:
            self.viewer.sync()
        time.sleep(TICK_RATE)

    def _running(self) -> bool:
        return self.viewer is None or self.viewer.is_running()

    # ── ControlInterface implementation ────────────────────────────────────────

    def _send_batch(self, joints: dict[str, float], unit: str = "deg") -> None:
        """
        Write all joint commands in one command so everything move simultaneously and smoothly instead stilted 1-by-1

        Rotation joints take `unit` ("deg" or "rad"), converted to radians; eye is mm → meters.
        """
        if unit not in ("deg", "rad"):
            raise ValueError(f"unit must be 'deg' or 'rad', got {unit!r}")
        for joint, value in joints.items():
            if joint not in self._act:
                continue   # joint not in model yet
            idx = self._act[joint]
            if joint == "eye":
                self.data.ctrl[idx] = value / 1000.0        # mm → m
            else:
                self.data.ctrl[idx] = math.radians(value) if unit == "deg" else value


    def get_position(self, joint: str) -> float:
        """
        Read the current joint position from the sim state.

        Returns degrees for rotation joints, mm for the eye. Raises ValueError for unknown joint names.
        """
        if joint not in self.LIMITS:
            raise ValueError(f"Unknown joint {joint!r}")
        if joint not in self._qpos:
            return 0.0   # joint not in model yet; report neutral

        idx = self._qpos[joint]
        raw = self.data.qpos[idx]
        if joint == "eye":
            return raw * 1000.0                      # m → mm
        else:
            return math.degrees(raw)                 # rad → deg

    def shutdown(self) -> None:
        """No-op — sim has no motors to power down."""
        pass
