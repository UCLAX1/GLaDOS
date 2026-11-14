"""
pid.py

Standalone, backend-agnostic PID controller.

The controller is unit-agnostic: feed it a measurement and a setpoint in the
same units (radians, meters, degrees, ...) and it returns a control effort in
whatever units the caller uses for the output (here: N·m torque for the hinge
joints, N for the sliding eye).
"""

from __future__ import annotations


class PIDController:
    """Discrete-time PID controller with integral anti-windup and output clamping.

    output = kp*e + ki*∫e dt + kd*de/dt
    """

    def __init__(
        self,
        kp: float = 0.0,
        ki: float = 0.0,
        kd: float = 0.0,
        setpoint: float = 0.0,
        output_limits: tuple[float, float] | None = None,
        integral_limits: tuple[float, float] | None = None,
        derivative_on_measurement: bool = True,
        anti_windup_gain: float = 0.0,
    ) -> None:
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)
        self.output_limits = output_limits
        self.integral_limits = integral_limits
        self.derivative_on_measurement = bool(derivative_on_measurement)
        self.anti_windup_gain = float(anti_windup_gain)
        self._setpoint = float(setpoint)
        self.reset()

    # ── setpoint ────────────────────────────────────────────────────────────

    @property
    def setpoint(self) -> float:
        return self._setpoint

    @setpoint.setter
    def setpoint(self, value: float) -> None:
        self._setpoint = float(value)

    # ── introspection ───────────────────────────────────────────────────────

    @property
    def integral(self) -> float:
        """Current value of the accumulated integral term."""
        return self._integral

    @property
    def last_output(self) -> float:
        """Most recent control output."""
        return self._last_output

    # ── control loop ────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Clear integral and history; call before a new control session."""
        self._integral = 0.0
        self._prev_error = 0.0
        self._prev_measurement: float | None = None
        self._last_output = 0.0

    def update(
        self,
        measurement: float,
        dt: float,
        setpoint: float | None = None,
    ) -> float:
        """Compute one control step and return the output.

        Args:
            measurement: current process value (same units as setpoint).
            dt: elapsed time since the previous update, in seconds. Values
                <= 0 do not advance the controller; the last output is returned.
            setpoint: optional per-call override; also stored as self.setpoint.
        """
        if setpoint is not None:
            self.setpoint = setpoint

        if dt <= 0.0:
            return self._last_output

        error = self._setpoint - float(measurement)

        p = self.kp * error

        self._integral += error * dt
        self._integral = self._clamp(self._integral, self.integral_limits)
        i = self.ki * self._integral

        if self.derivative_on_measurement:
            if self._prev_measurement is None:
                d = 0.0
            else:
                d = -self.kd * (float(measurement) - self._prev_measurement) / dt
        else:
            d = self.kd * (error - self._prev_error) / dt

        raw = p + i + d
        output = self._clamp(raw, self.output_limits)

        # Back-calculation anti-windup: bleed the integral down when the output
        # saturates so it does not wind up beyond what the actuator can apply.
        if self.anti_windup_gain > 0.0 and raw != output:
            self._integral -= self.anti_windup_gain * (raw - output) * dt
            self._integral = self._clamp(self._integral, self.integral_limits)

        self._prev_error = error
        self._prev_measurement = float(measurement)
        self._last_output = output
        return output

    # ── helpers ─────────────────────────────────────────────────────────────

    @staticmethod
    def _clamp(value: float, limits: tuple[float, float] | None) -> float:
        if limits is None:
            return value
        lo, hi = limits
        return max(lo, min(hi, value))
