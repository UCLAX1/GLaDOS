import sys, pathlib
import mujoco
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parents[2]))
import params


def Rz(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [c, -s, 0],
        [s,  c, 0],
        [0,  0, 1]
    ])


def Ry(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [ c, 0, s],
        [ 0, 1, 0],
        [-s, 0, c]
    ])


def fk_head_mount(theta_z, theta_p, theta_y):
    """
    theta_z: upper_arm_yaw_joint angle (rad)
    theta_p: upper_arm_pitch_joint angle (rad), same Y axis as the fixed lean so it just adds to it
    theta_y: lower_arm_joint angle (rad)
    Returns head_mount site position in world frame.
    """
    ALPHA = 0.0       # no fixed lean: zero pose hangs straight down
    L1, L2 = params.L1, params.L2  # from parameter.csv

    p0 = np.array([0, 0, -L2])
    p1 = Ry(theta_y) @ p0 + np.array([0, 0, -L1])
    p2 = Ry(ALPHA + theta_p) @ p1
    p3 = Rz(theta_z) @ p2 + np.array([0, 0, -0.005])

    return p3


def sim_head_mount(theta_z_deg: float, theta_p_deg: float, theta_y_deg: float):
    model = params.load_model()
    data = mujoco.MjData(model)

    # qpos order follows joint order in glados.xml: yaw, pitch, elbow
    data.qpos[0] = np.deg2rad(theta_z_deg)
    data.qpos[1] = np.deg2rad(theta_p_deg)
    data.qpos[2] = np.deg2rad(theta_y_deg)
    mujoco.mj_forward(model, data)  # propagate qpos changes to derived quantities (site positions, etc.)

    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "head_mount")
    return data.site_xpos[sid], fk_head_mount(*data.qpos[:3])


if __name__ == "__main__":
    cases = [(180, 0, 20), (0, 0, 0), (40, 0, 40), (0, 25, 0), (30, -20, 40), (-90, 28, 15)]
    rng = np.random.default_rng(0)
    cases += [tuple(rng.uniform(-30, 30, 3)) for _ in range(20)]
    for z, p, y in cases:
        y = abs(y)  # elbow range is 0..total
        sim_pos, fk_pos = sim_head_mount(z, p, y)
        print((z, p, y), "sim:", sim_pos, "fk:", fk_pos)
        assert np.allclose(sim_pos, fk_pos, atol=1e-6), "fk != sim"
    print("all fk checks passed")
