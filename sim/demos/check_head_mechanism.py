"""check_head_mechanism.py — SIM ONLY. Geometry sanity check for the parallel
head mechanism in model/push.xml.

Runs mj_forward at the default pose and prints the
end_effector pose plus the residual (rod_tip <-> plate site gap) on each
of the three closed-loop equality constraints. Residuals should be ~0 at
home; if they aren't, the link lengths in push.xml are geometrically
inconsistent — that's a flag to fix the numbers, not something to tune
around here.

Run: venv/bin/python demos/check_head_mechanism.py
"""

import pathlib
import sys

import mujoco
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))
import params

STATIONS = [
    ("horn0_tip", "plate_0"),
    ("horn1_tip", "plate_1"),
    ("plate2_tip", "plate_2"),
]


def main():
    model = params.load_model("push.xml")
    data = mujoco.MjData(model)

    mujoco.mj_forward(model, data)

    ee_pos = data.site("end_effector").xpos
    print(f"end_effector pose (xpos): {ee_pos}")

    for site1, site2 in STATIONS:
        gap = data.site(site1).xpos - data.site(site2).xpos
        residual = np.linalg.norm(gap)
        print(f"{site1} <-> {site2}: residual = {residual:.6f} m  (gap {gap})")


if __name__ == "__main__":
    main()
