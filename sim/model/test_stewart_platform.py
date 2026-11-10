import time
import numpy as np

import mujoco
import mujoco.viewer

m = mujoco.MjModel.from_xml_path('./stewart_platform.mujoco.xml')
d = mujoco.MjData(m)

# theta and phi of the normal angle (radians)
theta = 0.0
phi = 0.0

def update_angles():
  global theta 
  global phi
  # 1 rev/s = 2pirad / s
  theta_period = 4
  phi_period = 8

  theta_amplitude = np.deg2rad(5)
  phi_amplitude = np.deg2rad(5)

  # theta = theta_amplitude * np.sin(2 * np.pi * time.time() / theta_period)
  theta = 0
  phi = phi_amplitude * np.cos(2 * np.pi * time.time() / phi_period)
  # phi = 0 

# all values in meters and radians
ROD_LENGTH = 0.08
SERVO_ARM_LENGTH = 0.03
# LEVEL_PLATFORM_SERVO_ANGLE = np.deg2rad(150)
LEVEL_PLATFORM_SERVO_ANGLE = np.deg2rad(0)
INITIAL = 0.002
# MIN_SERVO_ANGLE = np.deg2rad(120)
# MAX_SERVO_ANGLE = np.deg2rad(180)
MIN_SERVO_ANGLE = np.deg2rad(-45)
MAX_SERVO_ANGLE = np.deg2rad(45)


# theta, phi in radians
# this code is a little broken and needs fixing
def get_servo_vals_from_angles(theta, phi):
  amplitude = np.sin(phi) * ROD_LENGTH / SERVO_ARM_LENGTH
  offset = 2 * np.pi / 3

  val1 = np.arcsin(amplitude * np.sin(theta) + (INITIAL/SERVO_ARM_LENGTH))
  val1 = np.clip(val1 + LEVEL_PLATFORM_SERVO_ANGLE, MIN_SERVO_ANGLE, MAX_SERVO_ANGLE)  

  val2 = np.arcsin(amplitude * np.sin(theta + offset) + (INITIAL/SERVO_ARM_LENGTH))
  val2 = np.clip(val2 + LEVEL_PLATFORM_SERVO_ANGLE, MIN_SERVO_ANGLE, MAX_SERVO_ANGLE)  


  val3 = np.arcsin(amplitude * np.sin(theta + (2 * offset)) + (INITIAL/SERVO_ARM_LENGTH))
  val3 = np.clip(val3 + LEVEL_PLATFORM_SERVO_ANGLE, MIN_SERVO_ANGLE, MAX_SERVO_ANGLE)  

  return val1, val2, val3

# servo vals in radians
# this function shouldn't be modified
def move_servos(model, data, servo_a_val, servo_b_val, servo_c_val):
  servo_a_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "servo_a") # 30 degrees
  servo_b_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "servo_b") # 150 degrees
  servo_c_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "servo_c")  # 270 degrees

  data.ctrl[servo_a_id] = np.rad2deg(servo_a_val)
  data.ctrl[servo_b_id] = np.rad2deg(servo_b_val)
  data.ctrl[servo_c_id] = np.rad2deg(servo_c_val)


with mujoco.viewer.launch_passive(m, d) as viewer:
  # Close the viewer automatically after 30 wall-seconds.
  start = time.time()
  while viewer.is_running():
    step_start = time.time()

    # update_angles()
    # servo_vals = get_servo_vals_from_angles(theta, phi)

    # for servo_val in servo_vals:
    #   # print(f"{np.rad2deg(servo_val):.2f} ", end='')
    #   print(f"{servo_val:.2f} ", end='')
    # print()

    # move_servos(m, d, *servo_vals)

    # placeholder test code because get_servo_vals_from_angles() is broken
    t = 0.005 * np.sin(2 * np.pi * time.time() / 2)
    move_servos(m, d, t, -t, 0)

    # mj_step can be replaced with code that also evaluates
    # a policy and applies a control signal before stepping the physics.
    mujoco.mj_step(m, d)

    # Pick up changes to the physics state, apply perturbations, update options from GUI.
    viewer.sync()

    # Rudimentary time keeping, will drift relative to wall clock.
    time_until_next_step = m.opt.timestep - (time.time() - step_start)
    if time_until_next_step > 0:
      time.sleep(time_until_next_step)