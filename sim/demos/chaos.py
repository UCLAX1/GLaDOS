#!/usr/bin/env python3
"""chaos.py — SIM ONLY. Run from X1_GLaDOS/: mjpython sim/demos/chaos.py"""

import os, sys, math, time, pathlib

if os.environ.get("GLADOS_HARDWARE"):
    print("SIM ONLY — real hardware detected. Aborting.")
    sys.exit(1)

import mujoco, mujoco.viewer
sys.path.insert(0, str(pathlib.Path(__file__).parents[2]))
from control.mujoco_control import MujocoControl

MODEL_XML = pathlib.Path(__file__).parents[1] / "model" / "glados.xml"

model = mujoco.MjModel.from_xml_path(str(MODEL_XML))
data  = mujoco.MjData(model)
robot = MujocoControl(model, data)

start = time.time()

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        t     = time.time() - start
        ramp  = min(1.0, t / 4.0)
        speed = 1.0 + t * 0.5

        robot.move(
            main_swivel = 180.0 * ramp * math.sin(speed * 2.1 * t + 0.0),
            lower_arm   =  60.0 * ramp * math.sin(speed * 3.7 * t + 1.2),  # negative clamped to 0
            tilt        =  15.0 * ramp * math.sin(speed * 5.3 * t + 2.4),
            nod         =  20.0 * ramp * math.sin(speed * 4.1 * t + 0.8),
            eye         =   2.0 * ramp * math.sin(speed * 7.9 * t + 3.1),
        )

        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.002)
