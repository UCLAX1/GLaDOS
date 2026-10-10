import math, os

# ---------- EDIT THESE (mm and degrees) ----------
EDGE_DISTANCE_MM   = 55    # center -> middle of each long side (bigger = bigger plate)
CORNER_DISTANCE_MM = 80    # center -> each flat end (smaller = bigger cut; max 110 = sharp point)
THICKNESS_MM       = 3     # plate depth (z)
FIRST_CORNER_DEG   = 90    # direction of the first flat end (90/210/330 matches the stations)
# --------------------------------------------------

EDGE_DISTANCE, CORNER_DISTANCE, THICKNESS = (x / 1000 for x in (EDGE_DISTANCE_MM, CORNER_DISTANCE_MM, THICKNESS_MM))
R = 2 * EDGE_DISTANCE                                   # where the sharp triangle points would be
flat_width = 2 * (R - CORNER_DISTANCE) * math.tan(math.radians(60))
pts = []
for k in range(3):
    a = math.radians(FIRST_CORNER_DEG + 120 * k)
    u, v = (math.cos(a), math.sin(a)), (-math.sin(a), math.cos(a))
    for s in (1, -1):                                   # ccw: right end of the flat, then left end
        pts.append((CORNER_DISTANCE * u[0] - s * flat_width / 2 * v[0],
                    CORNER_DISTANCE * u[1] - s * flat_width / 2 * v[1]))

n, h = len(pts), THICKNESS / 2
with open(os.path.join(os.path.dirname(__file__), "plate.obj"), "w") as f:
    for z in (-h, h):
        for x, y in pts: f.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
    f.write("f " + " ".join(str(i + 1) for i in range(n)) + "\n")
    f.write("f " + " ".join(str(n + i + 1) for i in reversed(range(n))) + "\n")
    for i in range(n):
        j = (i + 1) % n
        f.write(f"f {i+1} {j+1} {n+j+1} {n+i+1}\n")
print(f"plate.obj written: each flat end is {flat_width*1000:.1f} mm wide")
