# import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

class ArmSolver:

  def __init__(
    self,
    shoulder_length,
    elbow_length,
    head_radius,
  ):
    """init function

    :param shoulder_length: length from the base joint to the elbow joint
    :param elbow_length: length from the elbow joint to the neck 
    :param head_radius: distance from the platform to the the middle of the head

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

  def get_target_angles(self):
    arm_angles = [
        self.base_angle, 
        self.shoulder_angle, 
        self.elbow_angle,
    ]

    return arm_angles, self.target_head_rotation

  def set_target_pose(
    self,
    target_arm_position_vector,
    target_head_yaw,
    target_head_pitch,
    ):
    self.target_arm_position_vector = target_arm_position_vector
    self.target_head_yaw = target_head_yaw
    self.target_head_pitch = target_head_pitch

    self._set_target_motor_angles()

  # vectors are in global space
  def _set_target_motor_angles(self):
    # T
    arm_vector_length = np.linalg.norm(self.target_arm_position_vector)
    arm_vector_x = self.target_arm_position_vector[0]
    arm_vector_y = self.target_arm_position_vector[1]
    arm_vector_z = self.target_arm_position_vector[2]


    self.base_angle = np.atan2(arm_vector_y, arm_vector_x)

    shoulder_angle_cos = (self.shoulder_length**2.0 + arm_vector_length**2.0 - self.elbow_length**2.0) / (2.0 * self.shoulder_length * arm_vector_length)
    shoulder_angle_cos = np.clip(shoulder_angle_cos, -1.0, 1.0)
    # if np.abs(shoulder_angle_cos) > 1:
    #   print("CLAMPING SHOULDER ANGLE")
    #   shoulder_angle_cos = np.sign(shoulder_angle_cos)
    self.shoulder_angle = np.pi - np.arccos(arm_vector_z / arm_vector_length) - np.arccos(shoulder_angle_cos)
    # print(np.arccos(arm_vector_z / arm_vector_length), np.arccos(shoulder_angle_cos))

    elbow_angle_cos = (arm_vector_length**2.0 - self.shoulder_length**2.0 - self.elbow_length**2.0) / (2.0 * self.shoulder_length * self.elbow_length)
    elbow_angle_cos = np.clip(elbow_angle_cos, -1.0, 1.0)
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