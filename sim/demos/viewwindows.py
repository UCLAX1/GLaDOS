"""Windows simulation viewer. From sim, run: python3 "demos/view (windows).py" glados.xml"""

import math
import pathlib
import sys
import time

import mujoco
import mujoco.viewer

sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))
import params

model = params.load_model(sys.argv[1] if len(sys.argv) > 1 else "push.xml")
data = mujoco.MjData(model)

# Older MuJoCo versions have one control per actuator and expose only nu.
if hasattr(model, "nactuator"):
    linact_controls = [
        model.actuator_ctrladr[i]
        for i in range(model.nactuator)
        if model.actuator(i).name.startswith("linact") and model.actuator_ctrlnum[i] == 1
    ]
else:
    linact_controls = [
        i for i in range(model.nu) if model.actuator(i).name.startswith("linact")
    ]

start = time.time()
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        t = time.time() - start
        for control in linact_controls:
            lo, hi = model.actuator_ctrlrange[control]
            data.ctrl[control] = (lo + hi) / 2 + (hi - lo) / 2 * math.sin(t)
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.002)
