#!/usr/bin/env python3
"""torque_visualize.py — SIM ONLY. Watch the arm hold a pose with formula torques.

Opens the MuJoCo viewer on the infinite-torque model and feeds the 2-link
holding-torque formula into the arm motors every step (feedforward):

    tau_B = (F + m2*g/2) * L2 * sin(alpha + beta)          -> lower_arm_motor
    tau_C = (F + m2*g + m1*g/2) * L1 * sin(alpha) + tau_B  -> upper_arm_pitch_motor

The head assembly (tilt/nod/eye) is position-locked at 0 so it stays rigid.

Keys:
    T    toggle between the original formula (F at A) and the corrected one
         (F at the head COM, 0.10 m below A) — the original visibly sags.
    R    reset the arm back to the command pose.

Run: mjpython demos/torque_visualize.py [--pose ALPHA_DEG BETA_DEG]
"""

from __future__ import annotations

import argparse
import math
import pathlib
import sys
import time

import mujoco
import mujoco.viewer

ROOT = pathlib.Path(__file__).parents[2]   # project root
SIM = pathlib.Path(__file__).parents[1]    # sim/
DEMOS = pathlib.Path(__file__).parent      # sim/demos/
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SIM))
sys.path.insert(0, str(DEMOS))

import params
from torque_test import formula, _addresses, set_pose, HEAD_COM_OFFSET

FONT = mujoco.mjtFontScale.mjFONTSCALE_150
TOP_LEFT = mujoco.mjtGridPos.mjGRID_TOPLEFT
TOP_RIGHT = mujoco.mjtGridPos.mjGRID_TOPRIGHT

GLFW_KEY_T = 84
GLFW_KEY_R = 82


def main() -> None:
    ap = argparse.ArgumentParser(description="Watch the arm hold a pose with formula torques.")
    ap.add_argument("--pose", nargs=2, type=float, default=[0.0, 40.0],
                    metavar=("ALPHA_DEG", "BETA_DEG"),
                    help="pose to hold (default: 0 40)")
    args = ap.parse_args()

    model = params.load_torque_model()
    data = mujoco.MjData(model)
    qpos, dof, act, eye = _addresses(model)

    alpha0, beta0 = math.radians(args.pose[0]), math.radians(args.pose[1])
    state = {"offset": HEAD_COM_OFFSET}  # start corrected so the arm holds still

    def reset() -> None:
        set_pose(model, data, qpos, alpha0, beta0)

    def key_cb(key: int) -> None:
        if key == GLFW_KEY_T:
            state["offset"] = 0.0 if state["offset"] > 0.0 else HEAD_COM_OFFSET
            print(f"[visualize] formula offset = {state['offset']:.2f} m")
        elif key == GLFW_KEY_R:
            reset()
            print("[visualize] pose reset")

    reset()

    with mujoco.viewer.launch_passive(model, data, key_callback=key_cb) as viewer:
        while viewer.is_running():
            a = data.qpos[qpos["upper_arm"]]
            b = data.qpos[qpos["lower_arm"]]
            tb, tc = formula(a, b, state["offset"])

            data.ctrl[act["upper_arm"]] = tc
            data.ctrl[act["lower_arm"]] = tb
            data.ctrl[act["yaw"]] = 0.0
            data.ctrl[act["tilt"]] = 0.0   # position actuator target = 0
            data.ctrl[act["nod"]] = 0.0    # position actuator target = 0
            data.ctrl[eye] = 0.0
            mujoco.mj_step(model, data)

            label = "corrected (F at head COM)" if state["offset"] > 0.0 else "original (F at A)"
            drift_up = math.degrees(a - alpha0)
            drift_lo = math.degrees(b - beta0)

            viewer.set_texts([
                (FONT, TOP_LEFT,
                 f"target  alpha={args.pose[0]:+.1f} deg  beta={args.pose[1]:+.1f} deg",
                 f"actual  alpha={math.degrees(a):+.2f} deg  beta={math.degrees(b):+.2f} deg"),
                (FONT, TOP_RIGHT,
                 f"tau_B = {tb:+.3f} N m\ntau_C = {tc:+.3f} N m",
                 f"drift  alpha={drift_up:+.3f} deg  beta={drift_lo:+.3f} deg\nformula: {label}"),
            ])

            viewer.sync()
            time.sleep(0.002)


if __name__ == "__main__":
    main()
