"""Freeze parameters from model/parameter.csv -> glados.xml + joint limits.

glados.xml and push.xml are string.Templates; load with load_model(name), not from_xml_path.
rot_* columns are TOTAL range in deg: swivel/head are symmetric (+-half), elbow is 0..total.
"""
import csv, math, pathlib
from string import Template

import mujoco

_HERE = pathlib.Path(__file__).parent
_CSV = _HERE / "model" / "parameter.csv"

with open(_CSV) as f:
    P = {r["component"]: r for r in csv.DictReader(f)}


def _n(comp, col):
    return float(P[comp][col])


L1 = _n("upper_arm", "length_mm") / 1000  # swivel -> elbow (m)
L2 = _n("lower_arm", "length_mm") / 1000  # elbow -> head mount (m)

# joint limits in deg; eye is not in the freeze
LIMITS_DEG = {
    "main_swivel": (-_n("main_swivel", "rot_z_deg") / 2, _n("main_swivel", "rot_z_deg") / 2),
    "upper_arm":   (-_n("upper_arm", "rot_y_deg") / 2, _n("upper_arm", "rot_y_deg") / 2),
    "lower_arm":   (0.0, _n("lower_arm", "rot_y_deg")),
    "tilt":        (-_n("head", "rot_x_deg") / 2, _n("head", "rot_x_deg") / 2),
    "nod":         (-_n("head", "rot_y_deg") / 2, _n("head", "rot_y_deg") / 2),
    "eye":         (-2.0, 2.0),
}


def _vars():
    r = lambda k: math.radians(LIMITS_DEG[k][1])
    return dict(
        swivel=r("main_swivel"), pitch=r("upper_arm"), elbow=r("lower_arm"), tilt=r("tilt"), nod=r("nod"),
        L1=L1, L2=L2, ua_half=L1 / 2, la_half=L2 / 2,
        ua_w=_n("upper_arm", "width_mm") / 2000, la_w=_n("lower_arm", "width_mm") / 2000,
        ua_mass=_n("upper_arm", "mass_kg"), la_mass=_n("lower_arm", "mass_kg"),
        head_mass=_n("head", "mass_kg"),
    )


def xml(name="glados.xml") -> str:
    return Template((_HERE / "model" / name).read_text()).substitute(_vars())


def load_model(name="glados.xml") -> mujoco.MjModel:
    meshes = {f.name: f.read_bytes() for f in (_HERE / "assets").rglob("*.obj")}
    return mujoco.MjModel.from_xml_string(xml(name), meshes)
