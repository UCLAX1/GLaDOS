#!/usr/bin/env python3
"""view.py — SIM ONLY. Run: mjpython demos/view.py [path/to/model.xml]"""

import sys, time, pathlib
import mujoco, mujoco.viewer

xml_path = sys.argv[1] if len(sys.argv) > 1 else pathlib.Path(__file__).parents[1] / "model" / "push.xml"

model = mujoco.MjModel.from_xml_path(str(xml_path))
data = mujoco.MjData(model)

with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.opt.frame = mujoco.mjtFrame.mjFRAME_BODY
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.002)
