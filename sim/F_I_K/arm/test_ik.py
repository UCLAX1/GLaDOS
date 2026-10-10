"""fk -> ik -> sim round trip. Run: python sim/F_I_K/arm/test_ik.py"""
import sys, pathlib
import mujoco
import numpy as np

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))  # sim/ (params)
sys.path.insert(0, str(HERE.parents[2]))  # repo root (control)
import params
from fk import fk_head_mount
from control.control_interface import ik
from control.mujoco_control import MujocoControl

# (yaw, pitch, lower_arm) in deg
CASES = [(0, 0, 0), (0, 0, 90), (45, 10, 30), (-60, -20, 70), (89, 29, 5), (-30, 15, 45)]
ANGLE_TOL_DEG = 0.01
FK_TOL = 1e-6   # m, ik angles -> fk position
SIM_TOL = 1e-4  # m, settled sim head_mount vs fk with gravity off (pure kinematics)
SAG_TOL = 0.03  # m, with gravity on the PD actuators settle a bit low (~14 mm worst case)


def sim_error(model, data, robot, got, pos):
    mujoco.mj_resetData(model, data)
    robot._command(position=pos, unit="cart_pos")
    assert np.allclose(data.ctrl[:3], got), "cart_pos ctrl != ik angles"
    robot.settle()
    return np.linalg.norm(robot.head_position() - pos)


def check(model, data, robot, angles_deg):
    pos = fk_head_mount(*np.deg2rad(angles_deg))                  # 1. fk: where does this pose put the head?
    got = ik(*pos)                                                # 2. ik: which angles reach that point?
    assert np.allclose(np.rad2deg(got), angles_deg, atol=ANGLE_TOL_DEG), (angles_deg, np.rad2deg(got))
    assert np.allclose(fk_head_mount(*got), pos, atol=FK_TOL)     # 3. fk(ik(p)) == p

    g = model.opt.gravity.copy()
    model.opt.gravity[:] = 0                                      # 4. drive the sim via move(cart_pos), read the real site
    err = sim_error(model, data, robot, got, pos)
    assert err < SIM_TOL, f"{angles_deg}: sim head_mount is {err*1000:.3f} mm off fk"
    model.opt.gravity[:] = g
    sag = sim_error(model, data, robot, got, pos)            # 5. same with gravity: expect a small sag
    assert sag < SAG_TOL, f"{angles_deg}: {sag*1000:.1f} mm sag"
    return err, sag


def demo():
    model = params.load_model()
    data = mujoco.MjData(model)
    robot = MujocoControl(model, data)

    for a in CASES:
        err, sag = check(model, data, robot, a)
        print(a, f"sim error {err*1000:.4f} mm (gravity off), {sag*1000:.1f} mm sag (gravity on)")

    try:
        ik(0.5, 0, 0.3)
        assert False, "expected ValueError for an unreachable point"
    except ValueError:
        pass
    print("all ik checks passed")


if __name__ == "__main__":
    demo()
