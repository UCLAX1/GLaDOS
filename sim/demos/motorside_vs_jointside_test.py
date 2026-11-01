#!/usr/bin/env python3
"""
motorside_vs_jointside_test.py — test file to determine whether changes to gear attribute 
for <dcmotor> occur for the motor torque or joint torque
"""

# Initialize
import sys, time, math, pathlib
import mujoco, mujoco.viewer

sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))
import params

model = params.load_model("glados.xml")
data = mujoco.MjData(model)

# Obtain Addresses
act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "upper_arm_yaw_motor")
joint_id = model.actuator_trnid[act_id, 0]
dof_adr = model.jnt_dofadr[joint_id]

import sys, pathlib
import mujoco

# Load model and data
sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))
import params

model = params.load_model(sys.argv[1] if len(sys.argv) > 1 else "push.xml")
data = mujoco.MjData(model)

act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "linact0_actuator")
dof_adr = model.jnt_dofadr[model.actuator_trnid[act_id, 0]]

# Gear ratios to test dynamically
gear_ratios_to_test = [i for i in range(1, 20 + 1)]  # Test gear ratios from 1 to 10
constant_ff_input = 1.0  # Hold input constant across tests

print(f"{'Configured Gear':>16} | {'Motor Torque':>14} | {'Joint Torque':>14} | {'Measured Ratio':>16}")
print("-" * 68)

for gear in gear_ratios_to_test:
    # 1. Dynamically set gear ratio in model memory (First dimension of actuator_gear)
    model.actuator_gear[act_id, 0] = gear

    # 2. Reset simulation state and set constant input
    mujoco.mj_resetData(model, data)
    data.ctrl[act_id + 2] = constant_ff_input  # Feedforward slot (offset +2)

    # 3. Step forward to settle physics
    for _ in range(50):
        mujoco.mj_step(model, data)

    # 4. Extract outputs
    motor_torque = data.actuator_force[act_id]
    joint_torque = data.qfrc_actuator[dof_adr]
    measured_ratio = joint_torque / motor_torque if motor_torque != 0 else 0.0

    print(f"{gear:16.1f} | {motor_torque:14.4f} | {joint_torque:14.4f} | {measured_ratio:16.2f}")