import time
import numpy as np
import quaternion

import mujoco
import mujoco.viewer

# inverse kinematics referenced from:
# https://memememememememe.me/post/stewart-platform-math/

# another good resource:
# https://raw.org/research/inverse-kinematics-of-a-stewart-platform/

# https://quaternion.readthedocs.io/en/latest/quaternion/#quaternion.rotate_vectors
def rotate_vector_by_quat(v: np.ndarray, q: np.quaternion):
  q_float_array = quaternion.as_float_array(q)
  s = q_float_array[0] # scalar part
  r = q_float_array[1:] # vector part
  m = np.sum(q_float_array**2) # sum of squares

  return v + 2 * np.cross(r, s*v + np.cross(r, v)) / m

def pol2cart(r, theta):
  return r * np.array([np.cos(theta), np.sin(theta)])

class Glados:

  def __init__(self,
  ):
    """init function

    """

  def find_motor_angles(
    self,
    target_theta,
    target_phi
    ):
    pass


m = mujoco.MjModel.from_xml_path('./new_glados.mujoco.xml')
d = mujoco.MjData(m)

# theta and phi of the normal angle (radians)
theta = 0.0
phi = 0.0

# servo vals in radians
# this function shouldn't be modified
# def move_motors(model, data):
#   motor_a_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "motor_a") # 0 degrees (270 in mujoco)
#   motor_b_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "motor_b") # 120 degrees (30 in mujoco)
#   motor_c_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "motor_c")  # 240 degrees, along x axis (150 in mujoco)

#   # data.ctrl[motor_a_id] = np.rad2deg(motor_0_val)
#   # data.ctrl[motor_b_id] = np.rad2deg(motor_1_val)
#   # data.ctrl[motor_c_id] = np.rad2deg(motor_2_val)

#   data.ctrl[motor_a_id] = motor_0_val
#   data.ctrl[motor_b_id] = motor_1_val
#   data.ctrl[motor_c_id] = motor_2_val


with mujoco.viewer.launch_passive(m, d) as viewer:

  # print(d.body("rod_and_ball").xpos)
#   middle_rod_length = m.geom("middle_rod").size[1] * 2
#   base_half_length = m.geom("base_plate").size[1]
#   base_radius = m.geom("base_plate").size[0]
#   platform_half_length = m.geom("platform").size[1]
#   platform_radius = m.geom("platform").size[0]
#   servo_arm_length = m.geom("servo_arm_a").size[1] * 2
#   rod_length = m.geom("rod_geom_a").size[1] * 2
#   ball_radius = m.geom("ball").size[0] + platform_half_length 
#   base_to_ball_distance = middle_rod_length + base_half_length
#   base_to_servo_distance = base_radius


#   glados = Glados(
#   )

  # Close the viewer automatically after 30 wall-seconds.
  start = time.time()
  while viewer.is_running():
    step_start = time.time()

    # mujoco.mj_kinematics(m, d)

    # theta_slider_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "theta_slider")
    # phi_slider_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "phi_slider")

    # theta = np.deg2rad(d.ctrl[theta_slider_id])
    # phi = np.deg2rad(d.ctrl[phi_slider_id])

    target_normal_vector = np.array([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)])

    # motor_angles = glados.find_motor_angles()
    # move_motors(m, d, *motor_angles)

    # mj_step can be replaced with code that also evaluates
    # a policy and applies a control signal before stepping the physics.
    mujoco.mj_step(m, d)

    # Pick up changes to the physics state, apply perturbations, update options from GUI.
    viewer.sync()

    # Rudimentary time keeping, will drift relative to wall clock.
    time_until_next_step = m.opt.timestep - (time.time() - step_start)
    if time_until_next_step > 0:
      time.sleep(time_until_next_step)