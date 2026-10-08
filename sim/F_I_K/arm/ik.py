import sys, pathlib
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from helper import solve
from fk import fk_head_mount


def ik(x, y, z):
    """(yaw, pitch, lower_arm) in rad for head_mount at (x, y, z); raises if unreachable."""
    ang = solve(x, y, z)
    if ang is None:
        raise ValueError(f"({x}, {y}, {z}) is outside the arm's reach")
    return ang


if __name__ == "__main__":
    target = (-0.3, 0.1, -0.55)
    ang = ik(*target)
    print("angles (deg):", np.rad2deg(ang))
    assert np.allclose(fk_head_mount(*ang), target, atol=1e-6)
