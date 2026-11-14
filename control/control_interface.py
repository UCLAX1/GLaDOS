"""
control_interface.py

Abstract base class for the GLaDOS arm motor interface.
"""

import importlib, math, pathlib, sys, time
from abc import ABC, abstractmethod
from actions.sequence import Sequence
sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / "sim"))
sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / "sim" / "F_I_K" / "arm"))
import params
from helper import solve


def ik(x: float, y: float, z: float) -> tuple[float, float, float]:
    """(main_swivel, upper_arm, lower_arm) in rad for head_mount at (x, y, z) m; raises if unreachable."""
    ang = solve(x, y, z)
    if ang is None:
        raise ValueError(f"({x}, {y}, {z}) is outside the arm's reach")
    return ang


class ControlInterface(ABC):

    # ── Joint limits ───────────────────────────────────
    # Rotation joints in degrees, eye in mm.

    LIMITS = params.LIMITS_DEG  # from sim/model/parameter.csv
    ARRIVE_TOL = 2.0       # deg: how close counts as "arrived" for move(); PD sag is ~1 deg
    ARRIVE_TOL_EYE = 0.2   # mm

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _clamp(self, joint: str, value: float, unit: str = "deg") -> float:
        """Clamp value to the joint's allowed range (in `unit`; the eye is always mm)."""
        if joint not in self.LIMITS:
            raise ValueError(f"Unknown joint {joint!r}. Valid joints: {list(self.LIMITS)}")
        lo, hi = self.LIMITS[joint]
        if unit == "rad" and joint != "eye":
            lo, hi = math.radians(lo), math.radians(hi)
        return max(lo, min(hi, value))

    # ── Movement ──────────────────────────────────────────────────────────────

    def move(self, *, timeout: float = 3.0, **kwargs) -> None:
        """
        Move one or more joints and return once the arm has got there.

        Pass only the joints you want to move — omitted joints stay unchanged.
        All values are clamped to joint limits. Blocks, ticking, until every commanded joint is within
        ARRIVE_TOL of its target (or `timeout` seconds pass, or the sim window closes).

        Joints: main_swivel, upper_arm, lower_arm, tilt, nod, eye.
        Units:  rotation joints in `unit` ("deg" default, or "rad"), eye always mm.
        unit="cart_pos": pass position=(x, y, z) in meters (head_mount, from the top of the upper
        arm, z negative = below). ik picks main_swivel/upper_arm/lower_arm, so don't pass those;
        tilt/nod stay in degrees. Raises ValueError if the point is unreachable.

        Examples:
            robot.move(tilt=-15, nod=10)
            robot.move(main_swivel=90, lower_arm=30, tilt=5, nod=-10, eye=1.5)
            robot.move(position=(-0.3, 0.1, -0.55), unit="cart_pos")
        """
        self._wait_until_reached(self._command(**kwargs), timeout)

    def _command(
        self,
        main_swivel: float | None = None,
        upper_arm:   float | None = None,
        lower_arm:   float | None = None,
        tilt:        float | None = None,
        nod:         float | None = None,
        eye:         float | None = None,
        unit:        str = "deg",
        position:    tuple[float, float, float] | None = None,
    ) -> dict[str, float]:
        """
        Set the goal and return immediately, no ticking. For callers that tick themselves
        (Sequence, streaming demos). Same arguments as move(). Returns the targets in degrees (eye mm).
        """
        if unit == "cart_pos":
            if position is None:
                raise ValueError("unit='cart_pos' needs position=(x, y, z)")
            if any(v is not None for v in (main_swivel, upper_arm, lower_arm)):
                raise ValueError("unit='cart_pos' picks main_swivel/upper_arm/lower_arm itself")
            main_swivel, upper_arm, lower_arm = ik(*position)
            tilt, nod = (None if v is None else math.radians(v) for v in (tilt, nod))
            unit = "rad"
        elif position is not None:
            raise ValueError("position only works with unit='cart_pos'")

        updates = {
            "main_swivel": main_swivel,
            "upper_arm":   upper_arm,
            "lower_arm":   lower_arm,
            "tilt":        tilt,
            "nod":         nod,
            "eye":         eye,
        }
        batch = {
            joint: self._clamp(joint, value, unit)
            for joint, value in updates.items()
            if value is not None
        }
        if batch:
            self._send_batch(batch, unit)
        return {j: v if unit == "deg" or j == "eye" else math.degrees(v) for j, v in batch.items()}

    def _wait_until_reached(self, targets: dict[str, float], timeout: float) -> None:
        """Tick until each joint is within ARRIVE_TOL of its target (degrees; eye mm)."""
        from actions.sequence import TICK_RATE
        for _ in range(int(timeout / TICK_RATE)):
            if not self._running():
                return
            if all(abs(self.get_position(j) - v) < (self.ARRIVE_TOL_EYE if j == "eye" else self.ARRIVE_TOL)
                   for j, v in targets.items()):
                return
            self._tick()

    # ── Actions ───────────────────────────────────────────────────────────────

    def run_action(self, name: str, loop: bool = False) -> None:
        """
        Play an emote from actions/scripts/{name}.py (once, or until stopped if loop=True).

        Example:
            robot.run_action("nod")
        """  
        fn = getattr(importlib.import_module(f"actions.scripts.{name}"), name)
        seq = fn(Sequence(self, tick_fn=self._tick, running_fn=self._running))
        seq.loop() if loop else seq.play()

    # ── Tick hooks (override in subclass if time has to be advanced by hand) ──

    def _tick(self) -> None:
        """Advance one control tick. Hardware just waits; the sim steps physics here."""
        from actions.sequence import TICK_RATE
        time.sleep(TICK_RATE)

    def _running(self) -> bool:
        """False once playback should stop (e.g. the sim window was closed)."""
        return True

    # ── Abstract methods (must implement in subclass) ─────────────────────────

    @abstractmethod
    def _send_batch(self, joints: dict[str, float], unit: str = "deg") -> None:
        """
        Send position commands to one or more joints.

        Here is where the specific implementation differences between Mujoco and hardware will lie

        Args:
            joints: mapping of joint name → target value (eye in mm, rotation joints in `unit`)
            unit:   "deg" or "rad" for the rotation joints
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
