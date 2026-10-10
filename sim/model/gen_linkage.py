"""Rewrites the servo/rod/face part of push.xml from the measurements below.
Exists so the three push rods come out right: each rod2 length/angle is derived
from the real measurements (servo, horn, face corner) instead of hand-edited XML.
Run after editing:  python model/gen_linkage.py   (then reload the sim)
Only touches the text between the BEGIN/END gen_linkage markers in push.xml."""
import math, os, re

# ---------- EDIT THESE (mm and degrees) ----------
BASE_CORNER_MM   = 80    # must match CORNER_DISTANCE_MM in assets/base_plate_pushrod/gen_plate.py
BASE_THICK_MM    = 3     # must match THICKNESS_MM in gen_plate.py
FACE_CORNER_MM   = 64    # must match CORNER_DISTANCE_MM in assets/face_plate/gen_face.py
FACE_HEIGHT_MM   = 100   # face plate center above base plate center

SERVO_WIDTH_MM   = 40    # along the plate edge
SERVO_DEPTH_MM   = 20    # toward the plate center
SERVO_HEIGHT_MM  = 36    # up from the plate
SHAFT_BELOW_MM   = 7.5   # servo shaft height below the servo's top
SHAFT_OUT_MM     = 3     # horn standoff out from the servo's outer face

ROD1_MM          = 25    # horn length (shaft -> rod2 joint)
HORN_START_DEG   = -60   # rod1 angle from vertical at servo 0 (neg = up-left seen from outside)
SERVO_TRAVEL_DEG = 120   # servo sweeps 0..this, from HORN_START_DEG toward up-right
STATIONS_DEG     = (90, 210, 330)  # where each servo sits around the plate
# --------------------------------------------------

m = lambda mm: mm / 1000
fmt = lambda *v: " ".join("0" if abs(x) < 1e-9 else f"{x:.5f}".rstrip("0").rstrip(".") for x in v)

base_top = m(BASE_THICK_MM) / 2
servo_r = m(BASE_CORNER_MM - SERVO_DEPTH_MM / 2)          # servo flush against the flat end
servo_z = base_top + m(SERVO_HEIGHT_MM) / 2
shaft_z = base_top + m(SERVO_HEIGHT_MM - SHAFT_BELOW_MM)
shaft_r = servo_r + m(SERVO_DEPTH_MM / 2 + SHAFT_OUT_MM)  # on the outer face
face_z = m(FACE_HEIGHT_MM)
rng = math.radians(SERVO_TRAVEL_DEG)
h0 = math.radians(HORN_START_DEG)

face_sites, stations, servos = [], [], []
for i, deg in enumerate(STATIONS_DEG):
    a = math.radians(deg)
    u = (math.cos(a), math.sin(a))                        # outward
    s = (u[1], -u[0])                                     # along the edge, the way +servo angle swings rod1
    rod1 = (m(ROD1_MM) * math.sin(h0) * s[0], m(ROD1_MM) * math.sin(h0) * s[1], m(ROD1_MM) * math.cos(h0))
    rod2 = (m(FACE_CORNER_MM) * u[0] - shaft_r * u[0] - rod1[0],
            m(FACE_CORNER_MM) * u[1] - shaft_r * u[1] - rod1[1], face_z - shaft_z - rod1[2])
    rod2_len = math.hypot(*rod2)
    face_sites.append(f'      <site name="face_{i}" pos="{fmt(m(FACE_CORNER_MM) * u[0], m(FACE_CORNER_MM) * u[1], 0)}" size="0.003"/>')
    servos.append(f'    <geom name="servo{i}" type="box" size="{fmt(m(SERVO_WIDTH_MM) / 2, m(SERVO_DEPTH_MM) / 2, m(SERVO_HEIGHT_MM) / 2)}" '
                  f'pos="{fmt(servo_r * u[0], servo_r * u[1], servo_z)}" euler="0 0 {fmt(a - math.pi / 2)}" material="servo"/>')
    stations.append(f'''
    <!-- station {i} ({deg} deg): servo shaft (normal to servo face) -> rod1 (horn, sweeps parallel to face) -> rod2 -> face flat-end center -->
    <body name="rod1_{i}" pos="{fmt(shaft_r * u[0], shaft_r * u[1], shaft_z)}">
      <joint name="servo{i}_joint" type="hinge" axis="{fmt(u[0], u[1], 0)}" range="0 {fmt(rng)}" damping="0.05" armature="0.001"/>
      <geom type="capsule" fromto="0 0 0 {fmt(*rod1)}" size="0.003" material="horn"/>
      <body name="rod2_{i}" pos="{fmt(*rod1)}">
        <joint name="rod2_{i}_ball" type="ball" damping="0.001"/>
        <geom type="capsule" fromto="0 0 0 {fmt(*rod2)}" size="0.002" material="rod"/>
        <site name="rod2_{i}_tip" pos="{fmt(*rod2)}" size="0.003"/>
      </body>
    </body>''')

world = f'''    <!-- Center post: without it 3 RSS legs leave the face with 3 free DOF (it would drift sideways/spin).
         Visual only; the face's lift+tilt joints below are what it enforces. -->
    <geom name="center_post" type="capsule" fromto="0 0 {fmt(base_top)} 0 0 {fmt(face_z)}" size="0.003" material="rod"/>

    <!-- GLaDOS face, {FACE_HEIGHT_MM}mm above base. -->
    <body name="face" pos="0 0 {fmt(face_z)}">
      <!-- center guide: face can lift (slide) and tilt (2 hinges), no yaw = 3 DOF for 3 servos -->
      <joint name="face_lift" type="slide" axis="0 0 1" damping="0.05"/>
      <joint name="face_roll" type="hinge" axis="1 0 0" damping="0.01"/>
      <joint name="face_pitch" type="hinge" axis="0 1 0" damping="0.01"/>
      <geom name="face_plate" type="mesh" mesh="face_plate" material="gray"/>
{chr(10).join(face_sites)}
    </body>

    <!-- Servos: sit on the base plate top, flush against each flat end. -->
{chr(10).join(servos)}
{"".join(stations)}
'''
actuators = "".join(f'    <position name="servo{i}" joint="servo{i}_joint" kp="2" ctrlrange="0 {fmt(rng)}"/>\n'
                    for i in range(len(STATIONS_DEG)))

path = os.path.join(os.path.dirname(__file__), "push.xml")
xml = open(path).read()
for tag, body in (("gen_linkage", world), ("gen_linkage actuators", actuators)):
    xml, n = re.subn(rf"(<!-- BEGIN {tag} -->\n).*?(\s*<!-- END {tag} -->)", lambda g: g[1] + body.rstrip("\n") + g[2], xml, flags=re.S)
    assert n == 1, f"push.xml is missing the BEGIN/END {tag} markers"
open(path, "w").write(xml)
print(f"push.xml updated: rod2 is {rod2_len * 1000:.1f} mm")
