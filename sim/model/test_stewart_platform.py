import time
import numpy as np
import quaternion

import mujoco
import mujoco.viewer

# https://quaternion.readthedocs.io/en/latest/quaternion/#quaternion.rotate_vectors
def rotate_vector_by_quat(v: np.ndarray, q: np.quaternion):
  q_float_array = quaternion.as_float_array(q)
  s = q_float_array[0] # scalar part
  r = q_float_array[1:] # vector part
  m = np.sum(q_float_array**2) # sum of squares

  return v + 2 * np.cross(r, s*v + np.cross(r, v)) / m

def pol2cart(r, theta):
  return r * np.array([np.cos(theta), np.sin(theta)])



class StewartPlatform:

  DEFAULT_PLATFORM_NORMAL_VECTOR = np.array([0, 0, 1])

  def __init__(self,
    rod_length,
    servo_arm_length,
    platform_radius,
    base_to_servo_distance,
    base_to_ball_distance,
    ball_radius,
  ):
    """
    init function

    :param rod_length: length of the rod that connects the servo arm to the platform
    :param servo_arm_length: length of the arm that connects the servo to the rod
    :param platform_radius: radius of the platform
    :param base_to_servo_distance: distance from the middle of the base to the servo
    :param base_to_ball_distance: length from the middle of the base to the center of the ball
    :param ball_radius: distance from the center of the ball to the center of the platform
    """
    # all values in meters and radians
    self.rod_length = rod_length
    self.servo_arm_length = servo_arm_length
    self.platform_radius = platform_radius
    self.platform_edge_vectors = np.array([
      [*pol2cart(self.platform_radius, 0 * np.pi / 3), 0.0],
      [*pol2cart(self.platform_radius, 2 * np.pi / 3), 0.0],
      [*pol2cart(self.platform_radius, 4 * np.pi / 3), 0.0],
    ])
    self.base_to_servo_distance = base_to_servo_distance
    self.servo_vectors = np.array([
      [*pol2cart(self.base_to_servo_distance, 0 * np.pi / 3), 0.0],
      [*pol2cart(self.base_to_servo_distance, 2 * np.pi / 3), 0.0],
      [*pol2cart(self.base_to_servo_distance, 4 * np.pi / 3), 0.0],
    ])

    self.base_to_ball_distance = base_to_ball_distance
    self.base_to_ball_center_vector = np.array([0, 0, self.base_to_ball_distance])
    self.ball_radius = ball_radius

  def find_servo_angles(
    self,
    target_platform_normal_vector
    ):

    platform_rotation_axis = np.cross(self.DEFAULT_PLATFORM_NORMAL_VECTOR, target_platform_normal_vector)
    sin_value = np.linalg.norm(platform_rotation_axis)
    if (sin_value == 0):
      platform_rotation_axis = np.array([0, 0, 0])
    else:
      angle = np.arcsin(sin_value)
      platform_rotation_axis = (platform_rotation_axis / sin_value) * angle

    # get quaternion from axis-angle representation
    # see https://quaternion.readthedocs.io/en/latest/quaternion/#quaternion.from_rotation_vector
    platform_rotation_quat = quaternion.from_rotation_vector(platform_rotation_axis)

    ball_center_to_edge_vector = self.ball_radius * target_platform_normal_vector

    servo_angles = []

    for i in range(3):
      platform_edge_vector = self.platform_edge_vectors[i]
      servo_vector = self.servo_vectors[i]

      rotated_platform_edge_vector = rotate_vector_by_quat(platform_edge_vector, platform_rotation_quat)

      servo_to_platform_edge_vector = self.base_to_ball_center_vector + ball_center_to_edge_vector + rotated_platform_edge_vector - servo_vector
      l_z = servo_to_platform_edge_vector[2]
      l = np.linalg.norm(servo_to_platform_edge_vector)
      a = self.servo_arm_length
      s = self.rod_length

      servo_angle = np.arcsin(l_z / l) - np.arccos((l**2 + a**2 - s**2) / (2 * l * a))

      # print(l_z / l)
      # print((l**2 + a**2 - s**2) / (2 * l * a))

      servo_angles.append(servo_angle)

    return np.array(servo_angles)



m = mujoco.MjModel.from_xml_path('./stewart_platform.mujoco.xml')
d = mujoco.MjData(m)

# theta and phi of the normal angle (radians)
theta = 0.0
phi = 0.0

# theta, phi in radians
# this code is a little broken and needs fixing
# def get_servo_vals_from_angles(theta, phi):
#   amplitude = np.sin(phi) * ROD_LENGTH / SERVO_ARM_LENGTH
#   offset = 2 * np.pi / 3

#   val1 = np.arcsin(amplitude * np.sin(theta) + (INITIAL/SERVO_ARM_LENGTH))
#   val1 = np.clip(val1 + LEVEL_PLATFORM_SERVO_ANGLE, MIN_SERVO_ANGLE, MAX_SERVO_ANGLE)  

#   val2 = np.arcsin(amplitude * np.sin(theta + offset) + (INITIAL/SERVO_ARM_LENGTH))
#   val2 = np.clip(val2 + LEVEL_PLATFORM_SERVO_ANGLE, MIN_SERVO_ANGLE, MAX_SERVO_ANGLE)  

#   val3 = np.arcsin(amplitude * np.sin(theta + (2 * offset)) + (INITIAL/SERVO_ARM_LENGTH))
#   val3 = np.clip(val3 + LEVEL_PLATFORM_SERVO_ANGLE, MIN_SERVO_ANGLE, MAX_SERVO_ANGLE)  

#   return val1, val2, val3

# servo vals in radians
# this function shouldn't be modified
def move_servos(model, data, servo_0_val, servo_1_val, servo_2_val):
  servo_a_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "servo_a") # 0 degrees (270 in mujoco)
  servo_b_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "servo_b") # 120 degrees (30 in mujoco)
  servo_c_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "servo_c")  # 240 degrees, along x axis (150 in mujoco)

  # data.ctrl[servo_a_id] = np.rad2deg(servo_0_val)
  # data.ctrl[servo_b_id] = np.rad2deg(servo_1_val)
  # data.ctrl[servo_c_id] = np.rad2deg(servo_2_val)

  data.ctrl[servo_a_id] = servo_0_val
  data.ctrl[servo_b_id] = servo_1_val
  data.ctrl[servo_c_id] = servo_2_val


with mujoco.viewer.launch_passive(m, d) as viewer:

  # print(d.body("rod_and_ball").xpos)
  middle_rod_length = m.geom("middle_rod").size[1] * 2
  base_half_length = m.geom("base_plate").size[1]
  base_radius = m.geom("base_plate").size[0]
  platform_half_length = m.geom("platform").size[1]
  platform_radius = m.geom("platform").size[0]
  servo_arm_length = m.geom("servo_arm_a").size[1] * 2
  rod_length = m.geom("rod_geom_a").size[1] * 2
  ball_radius = m.geom("ball").size[0] + platform_half_length 
  base_to_ball_distance = middle_rod_length + base_half_length
  base_to_servo_distance = base_radius


  stewart_platform = StewartPlatform(
    rod_length = rod_length,
    servo_arm_length = servo_arm_length,
    platform_radius = platform_radius,
    base_to_servo_distance = base_radius,
    base_to_ball_distance = base_to_ball_distance,
    ball_radius = ball_radius,
  )

  # Close the viewer automatically after 30 wall-seconds.
  start = time.time()
  while viewer.is_running():
    step_start = time.time()

    # theta = 0
    # # phi = 0.1
    # phi = 0.00001

    # mujoco.mj_kinematics(m, d)

    # debug_rod_geom.pos = servo_geom_a
    # debug_rod_geom
    # platform_geom = d.geom("platform")
    # print(platform_site_a.xpos - servo_geom_a.xpos)
    # print(platform_site_a.xpos)
    # print(StewartPlatform.PLATFORM_EDGE_VECTORS[0])
    # print(d.geom("ball").xpos - d.geom("platform").xpos)
    # exit(0)

    # theta, phi = update_angles(theta, phi)
    # servo_vals = get_servo_vals_from_angles(theta, phi)

    # t = 0.005 * np.sin(2 * np.pi * time.time() / 2)

    # theta = 0
    # phi = t
    # phi = 0.0001
    theta_slider_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "theta_slider")
    phi_slider_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "phi_slider")

    theta = np.deg2rad(d.ctrl[theta_slider_id])
    phi = np.deg2rad(d.ctrl[phi_slider_id])

    target_platform_normal_vector = np.array([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)])

    servo_angles = stewart_platform.find_servo_angles(target_platform_normal_vector)
    move_servos(m, d, *servo_angles)

    # placeholder test code because get_servo_vals_from_angles() is broken
    # move_servos(m, d, t, -t, 0)

    # mj_step can be replaced with code that also evaluates
    # a policy and applies a control signal before stepping the physics.
    mujoco.mj_step(m, d)

    # Pick up changes to the physics state, apply perturbations, update options from GUI.
    viewer.sync()

    # Rudimentary time keeping, will drift relative to wall clock.
    time_until_next_step = m.opt.timestep - (time.time() - step_start)
    if time_until_next_step > 0:
      time.sleep(time_until_next_step)