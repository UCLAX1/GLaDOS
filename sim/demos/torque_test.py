#!/usr/bin/env python3
"""torque_test.py — SIM ONLY. Validate the 2-link static holding-torque formula in MuJoCo.

Model mapping:
    C = upper_arm_pitch pivot, B = lower_arm pivot, A = head_mount site
    L1, L2 = rod lengths (m), m1, m2 = rod masses (kg)
    F = external downward force at A = head weight = HEAD_MASS * g
    alpha = signed upper_arm_pitch angle from the downward vertical (rad)
    beta  = lower_arm angle from the upper-arm extension (rad)

Formula (joint-axis torques, N·m):

    tau_B = (F + m2*g/2) * L2 * sin(alpha + beta)

    tau_C = (F + m2*g + m1*g/2) * L1 * sin(alpha)
          + (F + m2*g/2) * L2 * sin(alpha + beta)

Ground truth is MuJoCo's data.qfrc_bias at zero velocity: at rest the actuator
torque needed to hold a pose equals qfrc_bias at that joint's DOF.

Known approximation: the 3 kg head cylinder's center of mass sits 0.10 m below A
(glados.xml head body pos "0 0 -0.10" relative to neck_platform), but the formula
puts F exactly at A. This script reports the original formula AND a corrected
variant where F acts at A - HEAD_COM_OFFSET.

Run (on macOS the viewer needs mjpython):
    mjpython demos/torque_test.py                      # headless: full alpha x beta grid
    mjpython demos/torque_test.py --joint lower_arm    # sweep beta, alpha fixed
    mjpython demos/torque_test.py --joint upper_arm    # sweep alpha, beta fixed
    mjpython demos/torque_test.py --hold               # apply formula torques, report drift
    mjpython demos/torque_test.py --view --pose 0 40   # watch the arm hold a pose
"""

from __future__ import annotations

import argparse
import math
import pathlib
import sys
import time

import mujoco
import numpy as np

ROOT = pathlib.Path(__file__).parents[2]  # project root
SIM = pathlib.Path(__file__).parents[1]   # sim/
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SIM))

import params

G = 9.81
L1, L2 = params.L1, params.L2
M1, M2 = params.M1, params.M2
HEAD_MASS = params.HEAD_MASS
F = HEAD_MASS * G  # external downward force at A (N)
HEAD_COM_OFFSET = 0.10  # m below head_mount; see glados.xml head body pos "0 0 -0.10"

JOINT_XML = {
    "yaw":       "upper_arm_yaw_joint",
    "upper_arm": "upper_arm_pitch_joint",
    "lower_arm": "lower_arm_joint",
    "tilt":      "tilt_joint",
    "nod":       "nod_joint",
    "eye":       "eye_joint",
}

MOTORS = {
    "yaw":       "upper_arm_yaw_motor",
    "upper_arm": "upper_arm_pitch_motor",
    "lower_arm": "lower_arm_motor",
    "tilt":      "tilt_actuator",
    "nod":       "nod_actuator",
}
EYE_ACTUATOR = "eye_actuator"

# Valid joint ranges in radians (from parameter.csv via params.LIMITS_DEG).
PITCH_RANGE = tuple(math.radians(x) for x in params.LIMITS_DEG["upper_arm"])  # [-π/6, +π/6]
LOWER_RANGE = tuple(math.radians(x) for x in params.LIMITS_DEG["lower_arm"])  # [0, +π/2]


def formula(alpha: float, beta: float, head_offset: float = 0.0) -> tuple[float, float]:
    """Return (tau_B, tau_C) in N·m for one arm pose.

    head_offset=0 puts F exactly at A (the user's formula). head_offset=HEAD_COM_OFFSET
    puts F at the head's actual center of mass (A - offset), which is the corrected variant.
    """
    s = math.sin(alpha + beta)
    tau_B = (F * (L2 + head_offset) + M2 * G * L2 / 2.0) * s
    tau_C = (F * L1 + M2 * G * L1 + M1 * G * L1 / 2.0) * math.sin(alpha) + tau_B
    return tau_B, tau_C


def _addresses(model: mujoco.MjModel):
    qpos, dof, act = {}, {}, {}
    for logical, xml in JOINT_XML.items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, xml)
        if jid == -1:
            raise SystemExit(f"joint {xml!r} not found in model")
        qpos[logical] = model.jnt_qposadr[jid]
        dof[logical] = model.jnt_dofadr[jid]
    for logical, name in MOTORS.items():
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if aid == -1:
            raise SystemExit(f"actuator {name!r} not found in model")
        act[logical] = aid
    eye = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, EYE_ACTUATOR)
    if eye == -1:
        raise SystemExit(f"actuator {EYE_ACTUATOR!r} not found in model")
    return qpos, dof, act, eye


def set_pose(model: mujoco.MjModel, data: mujoco.MjData, qpos: dict, alpha: float, beta: float) -> None:
    """Zero every joint, then place the arm at (alpha, beta) with zero velocity."""
    mujoco.mj_resetData(model, data)
    data.qpos[qpos["upper_arm"]] = alpha
    data.qpos[qpos["lower_arm"]] = beta
    data.qvel[:] = 0.0


def mujoco_hold_torques(model, data, qpos, dof, alpha, beta):
    """MuJoCo ground truth: joint torques (N·m) needed to hold (alpha, beta) at rest.

    Returns (tau_B_truth, tau_C_truth) in the same order as formula().
    """
    set_pose(model, data, qpos, alpha, beta)
    mujoco.mj_forward(model, data)
    return data.qfrc_bias[dof["lower_arm"]], data.qfrc_bias[dof["upper_arm"]]


def _sweep(joint: str, points: int, fixed_alpha_deg: float, fixed_beta_deg: float):
    alphas = np.linspace(PITCH_RANGE[0], PITCH_RANGE[1], points)
    betas = np.linspace(LOWER_RANGE[0], LOWER_RANGE[1], points)
    fa, fb = math.radians(fixed_alpha_deg), math.radians(fixed_beta_deg)
    if joint == "lower_arm":
        return [(fa, b) for b in betas]
    if joint == "upper_arm":
        return [(a, fb) for a in alphas]
    return [(a, b) for a in alphas for b in betas]


def validate(model, data, joint: str, points: int, fixed_alpha_deg: float,
             fixed_beta_deg: float, tol: float) -> None:
    """Compare formula torques against qfrc_bias for a sweep of poses."""
    qpos, dof, _, _ = _addresses(model)
    cases = _sweep(joint, points, fixed_alpha_deg, fixed_beta_deg)

    variants = (
        ("original  (F at A)", 0.0),
        ("corrected (F at head COM)", HEAD_COM_OFFSET),
    )

    print(f"[torque_test] {len(cases)} poses, joint={joint}, "
          f"fixed alpha={fixed_alpha_deg}°, fixed beta={fixed_beta_deg}°")
    print(f"  constants: L1={L1:.3f} m, L2={L2:.3f} m, m1={M1} kg, m2={M2} kg, "
          f"F={F:.2f} N, g={G}")
    print(f"  {'variant':28s} {'tau_B max|err|':>16s} {'tau_B mean|err|':>16s} "
          f"{'tau_C max|err|':>16s} {'tau_C mean|err|':>16s}  verdict")

    for label, offset in variants:
        err_B = np.empty(len(cases))
        err_C = np.empty(len(cases))
        for i, (alpha, beta) in enumerate(cases):
            tb, tc = formula(alpha, beta, offset)
            gb, gc = mujoco_hold_torques(model, data, qpos, dof, alpha, beta)
            err_B[i] = abs(tb - gb)
            err_C[i] = abs(tc - gc)

        verdict = "PASS" if err_B.max() <= tol and err_C.max() <= tol else "FAIL"
        print(f"  {label:28s} {err_B.max():16.6f} {err_B.mean():16.6f} "
              f"{err_C.max():16.6f} {err_C.mean():16.6f}  {verdict}")

    print(f"  tolerance: {tol} N·m")
    print("  note: the original formula puts F at A (head_mount), but the head COM sits")
    print("        {:.2f} m below A; the corrected variant moves F there. The small residual".format(HEAD_COM_OFFSET))
    print("        that remains is the neck_platform sphere + eye geoms (~14 g each from")
    print("        MuJoCo's default density), which the 2-link formula does not model.")


def _apply(model, data, qpos, act, eye, head_offset: float):
    """Feedforward: compute formula torques from the current pose and write them.

    The arm joints get the formula torque (infinite-torque motors); tilt/nod/eye are
    position actuators held at 0 so the head assembly stays rigid during the test.
    Returns (tau_B, tau_C) for logging.
    """
    a = data.qpos[qpos["upper_arm"]]
    b = data.qpos[qpos["lower_arm"]]
    tb, tc = formula(a, b, head_offset)
    data.ctrl[act["upper_arm"]] = tc
    data.ctrl[act["lower_arm"]] = tb
    data.ctrl[act["yaw"]] = 0.0
    data.ctrl[act["tilt"]] = 0.0   # position-actuator target = 0
    data.ctrl[act["nod"]] = 0.0    # position-actuator target = 0
    data.ctrl[eye] = 0.0
    return tb, tc


def _drift_report(data, qpos, dof, alpha0: float, beta0: float) -> None:
    drift_up = math.degrees(data.qpos[qpos["upper_arm"]] - alpha0)
    drift_lo = math.degrees(data.qpos[qpos["lower_arm"]] - beta0)
    print(f"    upper_arm drift={drift_up:+.5f}°  qvel={data.qvel[dof['upper_arm']]:+.3e} rad/s")
    print(f"    lower_arm drift={drift_lo:+.5f}°  qvel={data.qvel[dof['lower_arm']]:+.3e} rad/s")


def hold_test(model, data, poses, steps: int, head_offset: float) -> None:
    """Apply formula torques to the infinite-torque motors and measure drift."""
    qpos, dof, act, eye = _addresses(model)

    for alpha_deg, beta_deg in poses:
        alpha0, beta0 = math.radians(alpha_deg), math.radians(beta_deg)
        set_pose(model, data, qpos, alpha0, beta0)
        for _ in range(steps):
            _apply(model, data, qpos, act, eye, head_offset)
            mujoco.mj_step(model, data)
        print(f"[hold] head_offset={head_offset:+.2f} m  target (alpha={alpha_deg:+.1f}°, beta={beta_deg:+.1f}°)")
        _drift_report(data, qpos, dof, alpha0, beta0)


def view(model, data, alpha_deg: float, beta_deg: float, head_offset: float) -> None:
    """Hold (alpha_deg, beta_deg) live with the formula torques and open the viewer."""
    qpos, dof, act, eye = _addresses(model)
    alpha0, beta0 = math.radians(alpha_deg), math.radians(beta_deg)
    set_pose(model, data, qpos, alpha0, beta0)
    print(f"[view] pose (alpha={alpha_deg:+.1f}°, beta={beta_deg:+.1f}°), "
          f"head_offset={head_offset:+.2f} m")

    from mujoco import viewer as mujoco_viewer

    with mujoco_viewer.launch_passive(model, data) as v:
        while v.is_running():
            tb, tc = _apply(model, data, qpos, act, eye, head_offset)
            mujoco.mj_step(model, data)
            v.sync()
            time.sleep(0.002)


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate the GLaDOS arm static holding-torque formula.")
    ap.add_argument("--joint", choices=["lower_arm", "upper_arm", "both"], default="both",
                    help="which formula/joint to sweep (default: both)")
    ap.add_argument("--points", type=int, default=41,
                    help="sweep samples per joint (default: 41)")
    ap.add_argument("--fixed-alpha", type=float, default=0.0,
                    help="degrees; alpha held fixed during --joint lower_arm sweep")
    ap.add_argument("--fixed-beta", type=float, default=0.0,
                    help="degrees; beta held fixed during --joint upper_arm sweep")
    ap.add_argument("--tol", type=float, default=0.25,
                    help="N·m tolerance for PASS/FAIL (default: 0.25)")
    ap.add_argument("--hold", action="store_true",
                    help="apply formula torques to infinite-torque motors and report drift")
    ap.add_argument("--hold-steps", type=int, default=6000,
                    help="simulation steps for --hold (default: 6000 = 12 s)")
    ap.add_argument("--view", action="store_true",
                    help="open the viewer and hold a pose with the formula torques")
    ap.add_argument("--pose", nargs=2, type=float, default=[0.0, 40.0],
                    metavar=("ALPHA_DEG", "BETA_DEG"),
                    help="pose for --view, and first pose for --hold (default: 0 40)")
    args = ap.parse_args()

    if args.view:
        model = params.load_torque_model()
        data = mujoco.MjData(model)
        view(model, data, args.pose[0], args.pose[1], HEAD_COM_OFFSET)
        return

    if args.hold:
        model = params.load_torque_model()
        data = mujoco.MjData(model)
        a0, b0 = args.pose
        poses = [(a0, b0), (0.0, 0.0), (15.0, 30.0), (-10.0, 60.0), (0.0, 90.0)]
        for label, offset in (("original (F at A)", 0.0),
                              ("corrected (F at head COM)", HEAD_COM_OFFSET)):
            print(f"\n=== hold test: {label} ===")
            hold_test(model, data, poses, args.hold_steps, offset)
        return

    model = params.load_model()
    data = mujoco.MjData(model)
    validate(model, data, args.joint, args.points, args.fixed_alpha, args.fixed_beta, args.tol)


if __name__ == "__main__":
    main()
