import time
import numpy as np
from scipy.spatial.transform import Rotation as R

import mujoco
import mujoco.viewer

# inverse kinematics referenced from:
# https://memememememememe.me/post/stewart-platform-math/

# another good resource:
# https://raw.org/research/inverse-kinematics-of-a-stewart-platform/

# https://quaternion.readthedocs.io/en/latest/quaternion/#quaternion.rotate_vectors
# def rotate_vector_by_quat(v: np.ndarray, q: np.quaternion):
#   q_float_array = quaternion.as_float_array(q)
#   s = q_float_array[0] # scalar part
#   r = q_float_array[1:] # vector part
#   m = np.sum(q_float_array**2) # sum of squares

#   return v + 2 * np.cross(r, s*v + np.cross(r, v)) / m

def pol2cart(r, theta):
  return r * np.array([np.cos(theta), np.sin(theta)])

class Glados:

  def __init__(self,
  ):
    """init function

    """

    # motor angles
    self.shoulder_angle = 0.0
    self.elbow_angle = 0.0

    # normal vector of the head relative to the ELBOW
    self.head_normal: np.ndarray = np.array([0.0, 0.0, -1.0])

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
phi = np.pi


# temporary function, will implement as a pid loop later
# motor values between -pi/2 and pi/2 radians
def move_arm(model, data, shoulder_angle, elbow_angle):
  shoulder_motor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "shoulder_motor")
  elbow_motor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "elbow_motor")

  data.ctrl[shoulder_motor_id] = shoulder_angle
  data.ctrl[elbow_motor_id] = elbow_angle

def move_head(model, data, target_normal_vector):

  base_vector = np.array([0, 0, -1])

  axis_vector = np.cross(base_vector, target_normal_vector)

  sin_value = np.linalg.norm(axis_vector)
  if (sin_value == 0):
    axis_vector = np.array([0, 0, 0])
  else:
    angle = np.arcsin(sin_value)
    axis_vector = (axis_vector / sin_value) * angle

  # print(axis_vector)

  r = R.from_rotvec(axis_vector)

  euler_angles = r.as_euler("zyx", degrees=False)
  x_angle = euler_angles[2]
  y_angle = euler_angles[1]
  z_angle = euler_angles[0]
  # print(euler_angles)

  # # ROTATION MATRIX: zyx

  x_motor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "x_motor")
  y_motor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "y_motor")
  z_motor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "z_motor")

  data.ctrl[x_motor_id] = x_angle
  data.ctrl[y_motor_id] = y_angle
  data.ctrl[z_motor_id] = z_angle



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

    theta_slider_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "theta_slider")
    phi_slider_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "phi_slider")

    theta = np.deg2rad(d.ctrl[theta_slider_id])
    phi = np.deg2rad(d.ctrl[phi_slider_id])

    target_normal_vector = np.array([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)])

    move_head(m, d, target_normal_vector)

    # target_normal_vector = np.array([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)])

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