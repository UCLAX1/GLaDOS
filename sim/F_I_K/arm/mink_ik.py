import mujoco
import numpy as np
from mink import Configuration, FrameTask, SE3, solve_ik

model = mujoco.MjModel.from_xml_path("sim/model/glados.xml")
configuration = Configuration(model)

task = FrameTask(
    frame_name="head_mount",
    frame_type="site",
    position_cost=1.0,
    orientation_cost=0.0,
)
task.set_target(SE3.from_translation(np.array([0.5, 0.0, 0.3])))

for _ in range(max_iters):
    vel = solve_ik(configuration, [task], dt=0.01, solver="daqp")
    configuration.integrate_inplace(vel, dt=0.01)