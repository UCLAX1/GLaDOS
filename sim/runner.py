"""
runner.py

Sim-specific setup. Loads the MuJoCo model and produces Sequences
wired to the physics loop and viewer.

Usage:
    from sim.runner import GladosSim

    sim = GladosSim()
    with sim.launch() as viewer:
        sim.sequence(viewer).pose(head=30, duration=0.3).play()
"""

import sys
import time
import pathlib

import mujoco
import mujoco.viewer

sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import params
from control.mujoco_control import MujocoControl
from actions.sequence import Sequence, TICK_RATE


class GladosSim:
    """Loads the MuJoCo model and produces Sequences wired to the sim loop."""

    def __init__(self, xml_path=None):
        self.model = mujoco.MjModel.from_xml_path(str(xml_path)) if xml_path else params.load_model()
        self.data  = mujoco.MjData(self.model)
        self.robot = MujocoControl(self.model, self.data)

    def launch(self):
        """Open the passive viewer. Use as: `with sim.launch() as viewer`."""
        return mujoco.viewer.launch_passive(self.model, self.data)

    def sequence(self, viewer) -> Sequence:
        """Create a Sequence with sim-appropriate tick and running functions."""
        def tick():
            mujoco.mj_step(self.model, self.data)
            viewer.sync()
            time.sleep(TICK_RATE)

        return Sequence(
            robot      = self.robot,
            tick_fn    = tick,
            running_fn = viewer.is_running,
        )
