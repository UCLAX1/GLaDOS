import time
import numpy as np
from scipy.spatial.transform import Rotation as R
from robot_context import RobotContext

from py_trees import logging as log_tree
# from actions import make_bt

import mujoco
import mujoco.viewer

# def pol2cart(r, theta):
#   return r * np.array([np.cos(theta), np.sin(theta)])


# # temporary function, will implement as a pid loop later
# # motor values between -pi/2 and pi/2 radians
# def move_arm(model, data, base_angle, shoulder_angle, elbow_angle):
#   base_motor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "base_motor")
#   shoulder_motor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "shoulder_motor")
#   elbow_motor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "elbow_motor")

#   data.ctrl[base_motor_id] = base_angle
#   data.ctrl[shoulder_motor_id] = shoulder_angle
#   data.ctrl[elbow_motor_id] = elbow_angle

# # target rotation is relative to the elbow, where z is parallel to the elbow
# def move_head(model, data, target_rotation: R):

#   # zyx (extrinsic)
#   # ZYX (intrinsic)

#   euler_angles = target_rotation.as_euler("zyx")
#   angle_x = euler_angles[2]
#   angle_y = euler_angles[1]
#   angle_z = euler_angles[0]

#   x_motor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "head_motor_x")
#   y_motor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "head_motor_y")
#   z_motor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "head_motor_z")

#   data.ctrl[x_motor_id] = angle_x
#   data.ctrl[y_motor_id] = angle_y
#   data.ctrl[z_motor_id] = angle_z

log_tree.level = log_tree.Level.DEBUG
np.set_printoptions(precision=2, suppress=True)

model = mujoco.MjModel.from_xml_path('./new_glados.mujoco.xml')
data = mujoco.MjData(model)

# tree = make_bt()
# tree.tick_once()
# exit(0)


with mujoco.viewer.launch_passive(model, data) as viewer:

  shoulder_length = np.linalg.norm(model.body("elbow").pos)
  elbow_length = np.linalg.norm(model.body("platform").pos)
  head_radius = np.linalg.norm(model.body("head").pos)
  # print(shoulder_length)
  # print(elbow_length)
  # print(head_radius)

  robot_context = RobotContext(
    model,
    data,
    shoulder_length=shoulder_length,
    elbow_length=elbow_length,
    head_radius=head_radius,
  )

  head_yaw_slider_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "head_yaw_slider")
  head_pitch_slider_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "head_pitch_slider")

  arm_theta_slider_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "arm_theta_slider")
  arm_phi_slider_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "arm_phi_slider")
  arm_radius_slider_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "arm_radius_slider")

  # initialize sliders manually so they don't initialize to bad values
  data.ctrl[head_yaw_slider_id] = 0
  data.ctrl[head_pitch_slider_id] = 0

  data.ctrl[arm_theta_slider_id] = 0
  data.ctrl[arm_phi_slider_id] = 180
  data.ctrl[arm_radius_slider_id] = 1.3


  # Close the viewer automatically after 30 wall-seconds.
  start = time.time()
  while viewer.is_running():
    step_start = time.time()

    # mujoco.mj_kinematics(m, d)

    head_yaw = np.deg2rad(data.ctrl[head_yaw_slider_id])
    head_pitch = np.deg2rad(data.ctrl[head_pitch_slider_id])

    arm_theta = np.deg2rad(data.ctrl[arm_theta_slider_id])
    arm_phi = np.deg2rad(data.ctrl[arm_phi_slider_id])
    arm_radius = data.ctrl[arm_radius_slider_id]

    target_arm_position_vector = arm_radius * np.array([np.cos(arm_theta) * np.sin(arm_phi), np.sin(arm_theta) * np.sin(arm_phi), np.cos(arm_phi)])

    mocap_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "debug_target")
    mocap_idx = model.body_mocapid[mocap_id]
    data.mocap_pos[mocap_idx] = target_arm_position_vector + np.array([0, 0, 2.4])

    robot_context.set_target_pose_and_solve_ik(target_arm_position_vector, head_yaw, head_pitch)

    robot_context.move_arm()
    robot_context.move_head()

    # target_arm_angles, target_head_rotation = arm_solver.get_target_angles()

    # move_arm(model, data, *target_arm_angles)

    # move_head(model, data, target_head_rotation)


    # mj_step can be replaced with code that also evaluates
    # a policy and applies a control signal before stepping the physics.
    mujoco.mj_step(model, data)

    # Pick up changes to the physics state, apply perturbations, update options from GUI.
    viewer.sync()

    # Rudimentary time keeping, will drift relative to wall clock.
    time_until_next_step = model.opt.timestep - (time.time() - step_start)
    if time_until_next_step > 0:
      time.sleep(time_until_next_step)