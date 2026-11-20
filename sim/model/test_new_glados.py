import time
import numpy as np
from scipy.spatial.transform import Rotation as R

import mujoco
import mujoco.viewer

# def pol2cart(r, theta):
#   return r * np.array([np.cos(theta), np.sin(theta)])

class Glados:

  def __init__(
    self,
    shoulder_length,
    elbow_length,
    head_radius,
  ):
    """init function

    :param shoulder_length: length from the base joint to the elbow joint
    :param elbow_length: length from the elbow joint to the neck 
    :param head_radius: distance from the platform to the eye

    """

    # self.max_head_pitch_correction = np.deg2rad(30)
    # self.max_head_yaw_correction = np.deg2rad(30)

    self.max_head_pitch_correction = np.deg2rad(60)
    self.max_head_yaw_correction = np.deg2rad(60)

    self.shoulder_length = shoulder_length
    self.elbow_length = elbow_length
    self.head_radius = head_radius

    # global vector relative to the top base
    self.target_arm_position_vector = np.array([0.0, 0.0, -1.0])

    # relative to the elbow, where the no rotation means the platform is horizontal relative to the elbow
    self.target_head_rotation = R.from_quat([0, 0, 0, 1])

    self.target_head_yaw = 0.0
    self.target_head_pitch = 0.0

    # motor angles 
    self.base_angle = 0.0 # azimuth, 0 is along +x
    self.shoulder_angle = 0.0 # zenith, 0 is straight down
    self.elbow_angle = 0.0 # zenith, 0 is parallel to shoulder

  def set_target_pose(
    self,
    model,
    data,
    target_arm_position_vector,
    target_head_yaw,
    target_head_pitch,
    ):
    self.target_arm_position_vector = target_arm_position_vector
    self.target_head_yaw = target_head_yaw
    self.target_head_pitch = target_head_pitch

    self._set_target_motor_angles(model, data)

  # vectors are in global space
  def _set_target_motor_angles(self, model, data):
    # T
    arm_vector_length = np.linalg.norm(self.target_arm_position_vector)
    arm_vector_x = self.target_arm_position_vector[0]
    arm_vector_y = self.target_arm_position_vector[1]
    arm_vector_z = self.target_arm_position_vector[2]

    mocap_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "debug_target")
    mocap_idx = m.body_mocapid[mocap_id]
    data.mocap_pos[mocap_idx] = self.target_arm_position_vector + np.array([0, 0, 2.4])

    self.base_angle = np.atan2(arm_vector_y, arm_vector_x)

    shoulder_angle_cos = (self.shoulder_length**2 + arm_vector_length**2 - self.elbow_length**2) / (2 * self.shoulder_length * arm_vector_length)
    shoulder_angle_cos = np.clip(shoulder_angle_cos, -1, 1)
    # if np.abs(shoulder_angle_cos) > 1:
    #   print("CLAMPING SHOULDER ANGLE")
    #   shoulder_angle_cos = np.sign(shoulder_angle_cos)
    self.shoulder_angle = np.pi - np.arccos(arm_vector_z / arm_vector_length) - np.arccos(shoulder_angle_cos)
    # print(np.arccos(arm_vector_z / arm_vector_length), np.arccos(shoulder_angle_cos))

    elbow_angle_cos = (arm_vector_length**2 - self.shoulder_length**2 - self.elbow_length**2) / (2 * self.shoulder_length * self.elbow_length)
    elbow_angle_cos = np.clip(elbow_angle_cos, -1, 1)
    # if np.abs(elbow_angle_cos) > 1:
    #   print("CLAMPING ELBOW ANGLE")
    #   elbow_angle_cos = np.sign(elbow_angle_cos)
    self.elbow_angle = np.arccos(elbow_angle_cos)

    # global_elbow_angle = np.pi - (self.shoulder_angle + self.elbow_angle)

    # how much the head has to rotate in pitch so the head angle matches target_head_pitch
    head_pitch_correction_angle = -(self.shoulder_angle + self.elbow_angle + self.target_head_pitch - np.pi / 2.0)
    head_pitch_correction_angle = np.clip(head_pitch_correction_angle, -self.max_head_pitch_correction, self.max_head_pitch_correction)

    head_yaw_correction_angle = -(self.target_head_yaw - self.base_angle)
    head_yaw_correction_angle = np.clip(head_yaw_correction_angle, -self.max_head_yaw_correction, self.max_head_yaw_correction)

    # mocap_head_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "debug_target_head")
    # mocap_head_idx = m.body_mocapid[mocap_head_id]
    # data.mocap_pos[mocap_head_idx] = self.target_arm_position_vector + self.head_radius * np.array([np.cos(self.target_head_pitch) * np.cos(self.target_head_yaw), np.cos(self.target_head_pitch) * np.sin(self.target_head_yaw), -np.sin(self.target_head_pitch)]) + np.array([0, 0, 2.4])

    self.target_head_rotation = R.from_euler("zyx", [
      0.0, 
      head_yaw_correction_angle, 
      head_pitch_correction_angle
    ])

    # mocap_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "debug_target_head")
    # mocap_idx = m.body_mocapid[mocap_id]
    # data.mocap_pos[mocap_idx] = self.target_arm_position_vector + self.head_radius * self.target_head_rotation.inv().apply([0, 0, -1]) + np.array([0, 0, 2.4])

np.set_printoptions(precision=2, suppress=True)

m = mujoco.MjModel.from_xml_path('./new_glados.mujoco.xml')
d = mujoco.MjData(m)


# temporary function, will implement as a pid loop later
# motor values between -pi/2 and pi/2 radians
def move_arm(model, data, base_angle, shoulder_angle, elbow_angle):
  base_motor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "base_motor")
  shoulder_motor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "shoulder_motor")
  elbow_motor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "elbow_motor")

  data.ctrl[base_motor_id] = base_angle
  data.ctrl[shoulder_motor_id] = shoulder_angle
  data.ctrl[elbow_motor_id] = elbow_angle

# normal vector is relative to the elbow
def move_head(model, data, target_rotation: R):

  euler_angles = target_rotation.as_euler("zyx")
  angle_x = euler_angles[2]
  angle_y = euler_angles[1]
  angle_z = euler_angles[0]
  # print(euler_angles)

  # zyx (extrinsic)
  # ZYX (intrinsic)

  # ROTATION MATRIX: zyx

  x_motor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "head_motor_x")
  y_motor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "head_motor_y")
  z_motor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "head_motor_z")

  data.ctrl[x_motor_id] = angle_x
  data.ctrl[y_motor_id] = angle_y
  data.ctrl[z_motor_id] = angle_z



with mujoco.viewer.launch_passive(m, d) as viewer:

  # print(d.body("rod_and_ball").xpos)
  # shoulder_rod_length = m.geom("shoulder_rod").size[2] * 2
  shoulder_length = np.linalg.norm(m.body("elbow").pos)
  elbow_length = np.linalg.norm(m.body("platform").pos)
  head_radius = np.linalg.norm(m.body("head").pos)
  # print(shoulder_length)
  # print(elbow_length)
  # print(head_radius)

  head_yaw_slider_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "head_yaw_slider")
  head_pitch_slider_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "head_pitch_slider")

  glados_theta_slider_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "glados_theta_slider")
  glados_phi_slider_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "glados_phi_slider")
  glados_radius_slider_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "glados_radius_slider")

  d.ctrl[head_yaw_slider_id] = 0
  d.ctrl[head_pitch_slider_id] = 0

  d.ctrl[glados_theta_slider_id] = 0
  d.ctrl[glados_phi_slider_id] = 180
  d.ctrl[glados_radius_slider_id] = 1.3

  glados = Glados(
    shoulder_length=shoulder_length,
    elbow_length=elbow_length,
    head_radius=head_radius,
  )

  # Close the viewer automatically after 30 wall-seconds.
  start = time.time()
  while viewer.is_running():
    step_start = time.time()

    # mujoco.mj_kinematics(m, d)


    head_yaw = np.deg2rad(d.ctrl[head_yaw_slider_id])
    head_pitch = np.deg2rad(d.ctrl[head_pitch_slider_id])

    glados_theta = np.deg2rad(d.ctrl[glados_theta_slider_id])
    glados_phi = np.deg2rad(d.ctrl[glados_phi_slider_id])
    glados_radius = d.ctrl[glados_radius_slider_id]

    target_head_position_vector = glados_radius * np.array([np.cos(glados_theta) * np.sin(glados_phi), np.sin(glados_theta) * np.sin(glados_phi), np.cos(glados_phi)])

    glados.set_target_pose(m, d, target_head_position_vector, head_yaw, head_pitch)

    # target_head_direction_vector_relative_to_elbow = glados.get_target_head_direction_vector_relative_to_elbow()
    # print(f"{glados_theta:.2f}, {glados_phi:.2f}")
    # print("target head position vector:", target_head_position_vector)
    # print("target head direction vector:", target_head_direction_vector)
    # print("elbow unit vector:", elbow_unit_vector)

    # print("shoulder:", glados.shoulder_angle, "elbow:", glados.elbow_angle)
    move_arm(m, d, glados.base_angle, glados.shoulder_angle, glados.elbow_angle)

    move_head(m, d, glados.target_head_rotation)


    # mj_step can be replaced with code that also evaluates
    # a policy and applies a control signal before stepping the physics.
    mujoco.mj_step(m, d)

    # Pick up changes to the physics state, apply perturbations, update options from GUI.
    viewer.sync()

    # Rudimentary time keeping, will drift relative to wall clock.
    time_until_next_step = m.opt.timestep - (time.time() - step_start)
    if time_until_next_step > 0:
      time.sleep(time_until_next_step)