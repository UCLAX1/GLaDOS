#!/usr/bin/env python3
"""view.py — SIM ONLY. Run: mjpython demos/view.py [model.xml name in model/]"""

import sys, time, math, pathlib
import mujoco, mujoco.viewer

sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))
import params

# arg = model file name in model/ (default push.xml)
model = params.load_model(sys.argv[1] if len(sys.argv) > 1 else "push.xml")
data = mujoco.MjData(model)

# drive any linactN_actuator present so slide motion is visible (position
# actuators otherwise hold ctrl=0 forever and never move)
linact_ids = [i for i in range(model.nu)
              if model.actuator(i).name.startswith("linact")]

start = time.time()
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        t = time.time() - start
        for i in linact_ids:
            lo, hi = model.actuator_ctrlrange[i]
            data.ctrl[i] = (lo + hi) / 2 + (hi - lo) / 2 * math.sin(t)
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.002)
