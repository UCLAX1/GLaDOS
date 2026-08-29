"""
sequence.py

Backend-agnostic pose sequencer. Works on sim and hardware identically.

Usage
-----
Sim:
    from sim.runner import GladosSim
    sim = GladosSim()
    with sim.launch() as viewer:
        (sim.sequence(viewer)
            .pose(tilt=-15, nod=15, duration=0.4)
            .pose(tilt=0,   nod=0,  duration=0.5)
            .loop())

Hardware (once HardwareControl exists):
    from control.hardware_control import HardwareControl
    from actions.sequence import Sequence
    robot = HardwareControl(port="/dev/ttyUSB0")
    (Sequence(robot)
        .pose(tilt=-15, nod=15, duration=0.4)
        .pose(tilt=0,   nod=0,  duration=0.5)
        .loop())

The .pose() chain is identical — only the setup lines change.
"""

import time

TICK_RATE = 0.002  # seconds per step


def _smoothstep(t: float) -> float:
    """Ease in/out: slow start, fast middle, slow end. t in [0, 1]."""
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


class Sequence:
    """
    Chainable pose builder. Backend-agnostic — works on sim and hardware.

    Parameters
    ----------
    robot       : ControlInterface  (sim or hardware)
    tick_fn     : called every loop tick. On sim, steps physics + syncs viewer.
                  On hardware, defaults to a plain sleep.
    running_fn  : returns False when the sequence should stop.
                  On sim, tied to viewer.is_running(). On hardware, always True.

    Methods
    -------
    .pose(duration, lerp, additive, **joints)
        Move joints to target and hold.
        lerp=True      smoothly interpolate from current position to target
        additive=True  target = current_position + given_value (relative move)

    .wait(duration)   hold current position for duration seconds
    .play()           run once, returns self
    .loop(n=None)     repeat n times (None = until stopped)
    """

    def __init__(self, robot, tick_fn=None, running_fn=None):
        self._robot      = robot
        self._tick       = tick_fn    or (lambda: time.sleep(TICK_RATE))
        self._is_running = running_fn or (lambda: True)
        self._steps      = []   # (joints, duration, lerp, additive)

    def pose(
        self,
        duration: float = 0.5,
        lerp: bool = True,
        additive: bool = False,
        **joints,
    ) -> "Sequence":
        """
        Add a pose step.

        duration  : seconds to complete the move
        lerp      : smoothly interpolate to target (default True)
        additive  : move relative to current position instead of absolute (default False)
        **joints  : degrees for rotation joints, mm for eye
        """
        self._steps.append((joints, duration, lerp, additive))
        return self

    def wait(self, duration: float = 0.5) -> "Sequence":
        """Hold the current position for duration seconds."""
        self._steps.append(({}, duration, False, False))
        return self

    def play(self) -> "Sequence":
        """Execute all steps once. Returns self for chaining."""
        for joints, duration, lerp, additive in self._steps:

            # ── wait step ────────────────────────────────────────────────────
            if not joints:
                deadline = time.time() + duration
                while time.time() < deadline:
                    if not self._is_running():
                        return self
                    self._tick()
                continue

            # ── resolve additive → absolute targets ──────────────────────────
            targets = {}
            for joint, value in joints.items():
                if additive:
                    current = self._robot.get_position(joint)
                    targets[joint] = current + value
                else:
                    targets[joint] = value

            # ── lerp: interpolate smoothly to target ─────────────────────────
            if lerp:
                starts  = {j: self._robot.get_position(j) for j in targets}
                t_start = time.time()
                while True:
                    if not self._is_running():
                        return self
                    elapsed = time.time() - t_start
                    t       = _smoothstep(elapsed / max(duration, 1e-6))
                    interp  = {
                        j: starts[j] + (targets[j] - starts[j]) * t
                        for j in targets
                    }
                    self._robot.move(**interp)
                    self._tick()
                    if elapsed >= duration:
                        break

            # ── no lerp: jump to target, hold ────────────────────────────────
            else:
                self._robot.move(**targets)
                deadline = time.time() + duration
                while time.time() < deadline:
                    if not self._is_running():
                        return self
                    self._tick()

        return self

    def loop(self, n: int | None = None) -> None:
        """Play n times. n=None runs until stopped (viewer closed or Ctrl-C)."""
        count = 0
        try:
            while (n is None or count < n) and self._is_running():
                self.play()
                count += 1
        except KeyboardInterrupt:
            pass
