#!/usr/bin/env python3
"""move_and_nod.py — SIM ONLY. Go to a point, nod, go to another point, nod, repeat. Run: mjpython sim/demos/move_and_nod.py"""

import sys, pathlib

ROOT = pathlib.Path(__file__).parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "sim")]
from sim.runner import GladosSim

POINTS = [(-0.3, 0.1, -0.55), (-0.2, -0.15, -0.6)]  # head_mount xyz in m, must be inside the reachable bowl


def run(robot, rounds=None):
    n = 0
    while robot._running() and (rounds is None or n < rounds):
        for point in POINTS:
            robot.move(position=point, unit="cart_pos")
            robot.run_action("nod")
        n += 1


if __name__ == "__main__":
    sim = GladosSim()
    with sim.launch():
        run(sim.robot)
