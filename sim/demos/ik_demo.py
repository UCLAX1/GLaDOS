#!/usr/bin/env python3
"""ik_demo.py — SIM ONLY. head_mount traces a circle via unit='cart_pos'. Run: mjpython demos/ik_demo.py"""

import sys, time, pathlib
import numpy as np

ROOT = pathlib.Path(__file__).parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "sim")]
from sim.runner import GladosSim

CENTER, RADIUS, PERIOD = np.array([-0.2, 0.0, -0.6]), 0.12, 6.0  # m, m, s; circle in the x-y plane


def target(t):
    a = 2 * np.pi * t / PERIOD
    return CENTER + RADIUS * np.array([np.cos(a), np.sin(a), 0.0])


if __name__ == "__main__":
    sim = GladosSim()
    with sim.launch():
        t0 = time.time()
        while sim.robot._running():
            sim.robot._command(position=target(time.time() - t0), unit="cart_pos")  # streams a new goal every tick; raises if the circle leaves the reachable bowl
            sim.robot._tick()
