import math, os

# ---------- EDIT THESE (mm and degrees) ----------
RADIUS_MM   = 70    # head radius
DEPTH_MM    = 80    # extrusion depth (z)
ARC_DEG     = 220   # solid arc; the rest (360 - ARC_DEG) is the open wedge
CENTER_DEG  = 270   # direction of the middle of the solid arc (270 = -y, where the eye sits)
STEP_DEG    = 5     # arc facet size
# --------------------------------------------------

R, h = RADIUS_MM / 1000, DEPTH_MM / 2000
n = round(ARC_DEG / STEP_DEG)
ring = [(R * math.cos(math.radians(CENTER_DEG - ARC_DEG / 2 + ARC_DEG * i / n)),
         R * math.sin(math.radians(CENTER_DEG - ARC_DEG / 2 + ARC_DEG * i / n))) for i in range(n + 1)]
pts = [(0.0, 0.0)] + ring                                # index 0 = pie apex, 1..n+1 = arc
m = len(pts)

with open(os.path.join(os.path.dirname(__file__), "head_220.obj"), "w") as f:
    for z in (-h, h):
        for x, y in pts: f.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
    for i in range(1, m - 1):                            # caps: fan from apex (bottom cw, top ccw)
        f.write(f"f 1 {i+2} {i+1}\n")
        f.write(f"f {m+1} {m+i+1} {m+i+2}\n")
    for i in range(1, m - 1):                            # curved wall
        f.write(f"f {i+1} {i+2} {m+i+2} {m+i+1}\n")
    f.write(f"f 1 2 {m+2} {m+1}\n")                      # wedge face at the arc start
    f.write(f"f {m} 1 {m+1} {2*m}\n")                    # wedge face at the arc end
print(f"head_220.obj written: {ARC_DEG} deg arc, R={RADIUS_MM} mm, depth={DEPTH_MM} mm")
