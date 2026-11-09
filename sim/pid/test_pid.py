"""
test_pid.py — unit checks for sim/pid/pid.py. No MuJoCo needed.

Run: python3 sim/pid/test_pid.py
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))  # sim/ on sys.path

from pid import PIDController


def approx(a, b, tol=1e-9):
    assert abs(a - b) <= tol, f"expected {b}, got {a}"


def test_proportional():
    pid = PIDController(kp=2.0)
    approx(pid.update(measurement=0.0, dt=0.1, setpoint=1.0), 2.0)


def test_integral():
    pid = PIDController(ki=1.0)
    pid.update(1.0, 0.1)          # error = 0 - 1 = -1, integral = -0.1
    approx(pid.integral, -0.1)
    approx(pid.update(1.0, 0.1), -0.2)   # integral = -0.2


def test_derivative_on_measurement():
    pid = PIDController(kd=1.0, setpoint=0.0)
    pid.update(0.0, 1.0)                     # first sample: no derivative
    approx(pid.update(1.0, 1.0), -1.0)       # d = -1 * (1 - 0) / 1


def test_derivative_on_error():
    pid = PIDController(kd=1.0, setpoint=0.0, derivative_on_measurement=False)
    pid.update(0.0, 1.0)                     # error 0 -> 0, d = 0
    approx(pid.update(1.0, 1.0), -1.0)       # d = (-1 - 0) / 1


def test_output_clamp():
    pid = PIDController(kp=10.0, output_limits=(-5.0, 5.0))
    approx(pid.update(0.0, 0.1, setpoint=1.0), 5.0)


def test_integral_clamp():
    pid = PIDController(ki=1.0, integral_limits=(-0.1, 0.1))
    pid.update(1.0, 0.1)   # integral = -0.1 (clamped)
    pid.update(1.0, 0.1)   # would be -0.2, stays -0.1
    approx(pid.integral, -0.1)


def test_back_calculation_anti_windup():
    pid = PIDController(ki=10.0, output_limits=(-1.0, 1.0), anti_windup_gain=1.0)
    out = 0.0
    for _ in range(100):
        out = pid.update(0.0, 0.1, setpoint=1.0)
    assert out <= 1.0
    assert abs(pid.integral) < 1.0


def test_reset():
    pid = PIDController(kp=1.0)
    pid.update(0.0, 0.1, setpoint=1.0)
    pid.reset()
    assert pid.integral == 0.0
    approx(pid.update(0.0, 0.1, setpoint=1.0), 1.0)


def test_dt_guard():
    pid = PIDController(kp=1.0)
    first = pid.update(0.0, 0.1, setpoint=1.0)
    assert pid.update(0.0, 0.0, setpoint=1.0) == first
    assert pid.update(0.0, -0.1, setpoint=1.0) == first


def test_setpoint_property():
    pid = PIDController(kp=1.0, setpoint=2.0)
    pid.setpoint = 3.0
    approx(pid.update(2.0, 0.1), 1.0)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  {name} OK")
    print("OK")
