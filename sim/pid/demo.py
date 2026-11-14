#!/usr/bin/env python3
"""
demo.py — SIM ONLY. Torque-control the GLaDOS arm with a Python PID loop.

The arm joints use MuJoCo dcmotor actuators switched to input="voltage".
Each tick the Python PIDController computes a joint torque; that torque is
converted to a motor terminal voltage with back-EMF compensation:

    v = (R/K) * (tau / gear) + K * (gear * qvel)

and written to data.ctrl. Motor physics (current/torque saturation, gear,
armature, back-EMF) are handled by MuJoCo downstream of v.

Run:
    mjpython pid/demo.py              # hold a pose, open the viewer
    mjpython pid/demo.py --track      # sine-wave tracking on some joints
    python3  pid/demo.py --headless   # no viewer; print mean tracking error

On macOS the passive viewer requires mjpython, not plain python3.
"""

from __future__ import annotations

import argparse
import math
import pathlib
import sys
import time

import mujoco

ROOT = pathlib.Path(__file__).parents[2]  # project root
SIM = pathlib.Path(__file__).parents[1]   # sim/
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SIM))

import params
from pid import PIDController

TICK_RATE = 0.002  # seconds of wall-clock sleep per step (matches actions/sequence.py)

# logical joint name -> joint name in glados.xml
JOINT_XML = {
    "main_swivel": "upper_arm_yaw_joint",
    "upper_arm":   "upper_arm_pitch_joint",
    "lower_arm":   "lower_arm_joint",
    "tilt":        "tilt_joint",
    "nod":         "nod_joint",
    "eye":         "eye_joint",
}

# logical joint name -> dcmotor actuator name in glados.xml
MOTORS = {
    "main_swivel": "upper_arm_yaw_motor",
    "upper_arm":   "upper_arm_pitch_motor",
    "lower_arm":   "lower_arm_motor",
    "tilt":        "tilt_actuator",
    "nod":         "nod_actuator",
}
EYE_ACTUATOR = "eye_actuator"

V_SUPPLY = 12.0  # bus voltage clamp (glados.xml also sets ctrlrange="-12 12")

# Pose-hold targets. Rotation joints in degrees, eye in mm.
POSE = {
    "main_swivel": 30.0,
    "upper_arm": -10.0,
    "lower_arm": 40.0,
    "tilt": 10.0,
    "nod": -10.0,
    "eye": 1.5,
}

# --track: sine target = base + amp * sin(2*pi*freq*t). Degrees.
TRACK = {
    "main_swivel": {"base_deg": 0.0,  "amp_deg": 35.0, "freq": 0.3},
    "lower_arm":   {"base_deg": 40.0, "amp_deg": 20.0, "freq": 0.25},
    "nod":         {"base_deg": 0.0,  "amp_deg": 10.0, "freq": 0.4},
}

# PID gains: joint torque (N·m) per radian of joint error.
GAINS = {
    "main_swivel": {"kp": 400.0,  "ki": 40.0,  "kd": 20.0},
    "upper_arm":   {"kp": 1500.0, "ki": 150.0, "kd": 60.0},
    "lower_arm":   {"kp": 1200.0, "ki": 120.0, "kd": 50.0},
    "tilt":        {"kp": 150.0,  "ki": 15.0,  "kd": 8.0},
    "nod":         {"kp": 150.0,  "ki": 15.0,  "kd": 8.0},
}

# Joint torque output limits (N·m) for the Python PID, before voltage mapping.
TORQUE_LIMITS = {
    "main_swivel": (-60.0, 60.0),
    "upper_arm":   (-80.0, 80.0),
    "lower_arm":   (-60.0, 60.0),
    "tilt":        (-5.0, 5.0),
    "nod":         (-5.0, 5.0),
}


def to_sim(joint: str, value: float) -> float:
    """Human units (deg / mm) -> sim units (rad / m)."""
    if joint == "eye":
        return value / 1000.0
    return math.radians(value)


def from_sim(joint: str, value: float) -> float:
    """Sim units (rad / m) -> human units (deg / mm)."""
    if joint == "eye":
        return value * 1000.0
    return math.degrees(value)


def integral_limits(joint: str) -> tuple[float, float]:
    """Cap the integral term at half the torque range (anti-windup)."""
    lo, hi = TORQUE_LIMITS[joint]
    ki = GAINS[joint]["ki"] or 1.0
    span = max(abs(lo), abs(hi)) / (2.0 * ki)
    return (-span, span)


def build_controllers() -> dict:
    return {
        joint: PIDController(
            kp=GAINS[joint]["kp"],
            ki=GAINS[joint]["ki"],
            kd=GAINS[joint]["kd"],
            output_limits=TORQUE_LIMITS[joint],
            integral_limits=integral_limits(joint),
        )
        for joint in MOTORS
    }


def target(joint: str, t: float, tracking: bool) -> float:
    """Setpoint in sim units for one joint at time t (seconds)."""
    if tracking and joint in TRACK:
        cfg = TRACK[joint]
        value = cfg["base_deg"] + cfg["amp_deg"] * math.sin(2 * math.pi * cfg["freq"] * t)
        return to_sim(joint, value)
    return to_sim(joint, POSE[joint])


def main() -> None:
    ap = argparse.ArgumentParser(description="Torque-control the GLaDOS arm with a Python PID loop.")
    ap.add_argument("--track", action="store_true", help="sine-wave tracking instead of a static pose")
    ap.add_argument("--headless", action="store_true", help="no viewer; print mean tracking error and exit")
    ap.add_argument("--steps", type=int, default=6000, help="simulation steps for --headless (default 6000 = 12 s)")
    args = ap.parse_args()

    model = params.load_model()
    data = mujoco.MjData(model)

    qpos_adr: dict[str, int] = {}
    for joint, xml in JOINT_XML.items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, xml)
        if jid == -1:
            raise SystemExit(f"joint {xml!r} not found in model")
        qpos_adr[joint] = model.jnt_qposadr[jid]

    motor: dict[str, dict] = {}
    for joint, act_name in MOTORS.items():
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, act_name)
        if aid == -1:
            raise SystemExit(f"actuator {act_name!r} not found in model")
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, JOINT_XML[joint])
        motor[joint] = {
            "R": model.actuator_gainprm[aid][0],
            "K": model.actuator_gainprm[aid][1],
            "gear": model.actuator_gear[aid][0],
            "ctrl_adr": aid,  # MuJoCo 3.10+: one control slot per actuator
            "qpos_adr": model.jnt_qposadr[jid],
            "dof_adr": model.jnt_dofadr[jid],
        }

    eye_aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, EYE_ACTUATOR)
    if eye_aid == -1:
        raise SystemExit(f"actuator {EYE_ACTUATOR!r} not found in model")
    eye_ctrl_adr = eye_aid  # MuJoCo 3.10+: one control slot per actuator

    pids = build_controllers()
    dt = model.opt.timestep

    def control(t: float, tracking: bool) -> None:
        for joint, m in motor.items():
            setpoint = target(joint, t, tracking)
            q = data.qpos[m["qpos_adr"]]
            qvel = data.qvel[m["dof_adr"]]
            tau = pids[joint].update(q, dt, setpoint=setpoint)  # joint torque (N·m)
            tau_motor = tau / m["gear"]
            omega_motor = m["gear"] * qvel
            v = (m["R"] / m["K"]) * tau_motor + m["K"] * omega_motor
            data.ctrl[m["ctrl_adr"]] = max(-V_SUPPLY, min(V_SUPPLY, v))
        # The eye is still a position actuator: write its target position (m).
        data.ctrl[eye_ctrl_adr] = target("eye", t, tracking)

    if args.headless:
        errors = {joint: 0.0 for joint in JOINT_XML}
        for step in range(args.steps):
            t = step * dt
            control(t, args.track)
            mujoco.mj_step(model, data)
            for joint in JOINT_XML:
                errors[joint] += abs(target(joint, t, args.track) - data.qpos[qpos_adr[joint]])
        t_last = (args.steps - 1) * dt
        print(f"[pid] headless run: {args.steps} steps ({args.steps * dt:.1f} s), dt={dt} s, track={args.track}")
        for joint in JOINT_XML:
            mean = errors[joint] / args.steps
            final = abs(target(joint, t_last, args.track) - data.qpos[qpos_adr[joint]])
            unit = "mm" if joint == "eye" else "deg"
            print(f"  {joint:12s} mean |error| = {from_sim(joint, mean):8.3f} {unit}   final |error| = {from_sim(joint, final):8.3f} {unit}")
        return

    from mujoco import viewer as mujoco_viewer

    with mujoco_viewer.launch_passive(model, data) as viewer:
        t0 = time.time()
        while viewer.is_running():
            t = time.time() - t0
            control(t, args.track)
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(TICK_RATE)


if __name__ == "__main__":
    main()
