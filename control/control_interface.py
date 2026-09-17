"""
control_interface.py

Abstract base class for the GLaDOS arm motor interface.
"""

import math
from abc import ABC, abstractmethod


class ControlInterface(ABC):

    # ── Joint limits ───────────────────────────────────
    # Rotation joints in degrees, eye in mm.

    JOINTS = ["main_swivel", "lower_arm", "tilt", "nod", "eye"]

    LIMITS = {
        "main_swivel": (-180.0, 180.0),   # rad: ±3.14159
        "lower_arm":   (0.0,    85.9),    # rad: 0 to 1.5
        "tilt":        (-17.2,  17.2),    # rad: ±0.3  (side to side)
        "nod":         (-34.4,  22.9),    # rad: -0.6 to 0.4  (forward/back)
        "eye":         (-2.0,   2.0),     # m:   ±0.002  (mm here)
    }

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _clamp(self, joint: str, value: float) -> float:
        """Clamp value to the joint's allowed range."""
        lo, hi = self.LIMITS[joint]
        return max(lo, min(hi, value))

    # ── Movement ──────────────────────────────────────────────────────────────

    def move(
        self,
        main_swivel: float | None = None,
        lower_arm:   float | None = None,
        tilt:        float | None = None,
        nod:         float | None = None,
        eye:         float | None = None,
    ) -> None:
        """
        Command one or more joints simultaneously.

        Pass only the joints you want to move — omitted joints stay unchanged.
        All values are clamped to joint limits before sending.

        Units: degrees for rotation joints, mm for eye.

        Examples:
            robot.move(tilt=-15, nod=10)
            robot.move(main_swivel=90, lower_arm=30, tilt=5, nod=-10, eye=1.5)
        """
        updates = {
            "main_swivel": main_swivel,
            "lower_arm":   lower_arm,
            "tilt":        tilt,
            "nod":         nod,
            "eye":         eye,
        }
        batch = {
            joint: self._clamp(joint, value)
            for joint, value in updates.items()
            if value is not None
        }
        if batch:
            self._send_batch(batch)

    # ── Abstract methods (must implement in subclass) ─────────────────────────

    @abstractmethod
    def _send_batch(self, joints: dict[str, float]) -> None:
        """
        Send position commands to one or more joints.

        Here is where the specific implementation differences between Mujoco and hardware will lie

        Args:
            joints: mapping of joint name → target value (degrees or mm)
        """
        ...

    @abstractmethod
    def get_position(self, joint: str) -> float:
        """
        Read the current position of a joint.
        Returns degrees for rotation joints, mm for eye.
        Raises ValueError for unknown joint names.
        """
        ...

    @abstractmethod
    def shutdown(self) -> None:
        """Safely stop all motors."""
        ...
