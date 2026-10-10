"""
runner.py

Opens the sim. Everything you do to the robot afterwards lives in control/.

Usage:
    from sim.runner import GladosSim

    sim = GladosSim()
    with sim.launch():
        sim.robot.move(nod=20)
        sim.robot.run_action("nod", loop=True)
"""

import sys
import pathlib

import mujoco

sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from control.mujoco_control import MujocoControl


class GladosSim:
    def __init__(self, xml_path=None):
        model = mujoco.MjModel.from_xml_path(str(xml_path)) if xml_path else None
        self.robot = MujocoControl(model)
        self.model, self.data = self.robot.model, self.robot.data

    def launch(self):
        """Open the viewer. Use as: `with sim.launch() as viewer`."""
        return self.robot.launch()
