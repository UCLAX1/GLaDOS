"""Is a head_mount coordinate reachable? Closed-form inverse of fk.fk_head_mount."""
import sys, pathlib
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parents[2]))
import params

L1, L2, DZ = params.L1, params.L2, 0.005  # DZ = fk's fixed 5 mm drop after yaw
TOL = 1e-9


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def solve(x, y, z):
    """Joint angles (yaw, pitch, elbow) in rad that put head_mount at (x, y, z), or None."""
    q = np.array([x, y, z + DZ])
    lo_y, hi_y = np.deg2rad(params.LIMITS_DEG["main_swivel"])
    lo_p, hi_p = np.deg2rad(params.LIMITS_DEG["upper_arm"])
    lo_e, hi_e = np.deg2rad(params.LIMITS_DEG["lower_arm"])

    # elbow: |q|^2 = L1^2 + L2^2 + 2 L1 L2 cos(elbow)
    c = (q @ q - L1**2 - L2**2) / (2 * L1 * L2)
    if abs(c) > 1 + TOL:
        return None
    elbow = np.arccos(np.clip(c, -1, 1))
    if not lo_e - TOL <= elbow <= hi_e + TOL:
        return None

    # pitch+yaw: p1 (elbow-plane vector) is rotated about Y by pitch, then about Z by yaw
    p1 = np.array([-L2 * np.sin(elbow), 0, -L1 - L2 * np.cos(elbow)])
    rho = np.hypot(q[0], q[1])
    for sign in (1, -1):  # p2 lies on the +x or -x side of the yaw axis
        yaw = _wrap(np.arctan2(q[1], q[0]) + (0 if sign == 1 else np.pi)) if rho > TOL else 0.0
        p2 = np.array([sign * rho, 0, q[2]])
        pitch = _wrap(np.arctan2(-p2[2], p2[0]) - np.arctan2(-p1[2], p1[0]))
        if lo_y - TOL <= yaw <= hi_y + TOL and lo_p - TOL <= pitch <= hi_p + TOL:
            return yaw, pitch, elbow
    return None


def is_reachable(x, y, z):
    return solve(x, y, z) is not None


if __name__ == "__main__":
    from fk import fk_head_mount
    rng = np.random.default_rng(0)
    lim = [np.deg2rad(params.LIMITS_DEG[k]) for k in ("main_swivel", "upper_arm", "lower_arm")]
    # every point fk can produce must be reachable, and the angles must round-trip
    for _ in range(2000):
        pt = fk_head_mount(*(rng.uniform(lo, hi) for lo, hi in lim))
        ang = solve(*pt)
        assert ang is not None, pt
        assert np.allclose(fk_head_mount(*ang), pt, atol=1e-6), (pt, ang)
    # known-bad: above the shoulder, too far, inside the fold-up hole
    for bad in [(0.5, 0, 0.3), (0, 0, -0.8), (0, 0, -0.3), (0.45, 0, -0.55)]:
        assert not is_reachable(*bad), bad
    print("helper checks passed")
