import mujoco
import numpy as np


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


def fk_head_mount(theta_z, theta_y):
    """
    theta_z: main_swivel_joint angle (rad)
    theta_y: lower_arm_joint angle (rad)
    Returns head_mount site position in world frame.
    """
    ALPHA = 0.3491    # fixed 20 deg upper_arm tilt
    L1 = 0.6092       # upper_arm -> lower_arm origin offset
    L2 = 0.3048       # lower_arm origin -> head_mount site offset

    p0 = np.array([0, 0, -L2])
    p1 = Ry(theta_y) @ p0 + np.array([0, 0, -L1])
    p2 = Ry(ALPHA) @ p1
    p3 = Rz(theta_z) @ p2 + np.array([0, 0, -0.005])

    return p3


def sim_head_mount(model_path: str, theta_z_deg: float, theta_y_deg: float):
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    data.qpos[0] = np.deg2rad(theta_z_deg)
    data.qpos[1] = np.deg2rad(theta_y_deg)
    mujoco.mj_forward(model, data)  # propagate qpos changes to derived quantities (site positions, etc.)

    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "head_mount")
    return data.site_xpos[sid], fk_head_mount(data.qpos[0], data.qpos[1])


if __name__ == "__main__":
    sim_pos, fk_pos = sim_head_mount("model/glados.xml", theta_z_deg=180, theta_y_deg=20)
    print("sim: ", sim_pos)
    print("fk:  ", fk_pos)
    sim_pos, fk_pos = sim_head_mount("model/glados.xml", theta_z_deg=0, theta_y_deg=0)
    print("sim: ", sim_pos)
    print("fk:  ", fk_pos)
    sim_pos, fk_pos = sim_head_mount("model/glados.xml", theta_z_deg=40, theta_y_deg=40)
    print("sim: ", sim_pos)
    print("fk:  ", fk_pos)
        
