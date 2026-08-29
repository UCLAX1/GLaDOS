# glados.xml — MJCF model, explained

This is a MuJoCo model file (MJCF format = MuJoCo's XML). It describes a robot as a tree of rigid
bodies connected by joints, plus the shapes to draw/collide and the motors that drive it. This doc
walks through the format piece by piece using `glados.xml` as the running example.

## The big idea: a kinematic tree

A MuJoCo body can contain child `<body>` tags, and each child is attached to its parent by a
`<joint>`. That nesting *is* the robot's skeleton. Read the indentation in the XML as "this is
attached to that":

```
ceiling_mount              (bolted to the world, doesn't move)
 └─ main_swivel            (turns left/right)
     └─ upper_arm          (rigid, no joint of its own)
         └─ lower_arm      (pitches up/down)
             └─ neck       (rotates)
                 └─ head   (pitches up/down)
                     └─ eye  (slides in/out)
```

Each body only knows about its own joint and its own offset from its parent — MuJoCo composes the
whole chain automatically. This is exactly the lamp-arm-plus-eyeball shape the "GLaDOS arm" name
implies.

## Top-level sections of the file

| Section         | What it's for                                                                                                                                                                                                        |
| --------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `<compiler>`  | <br /><br />Global parsing settings.`angle="radian"` means every angle in this file (joint ranges, `euler=...`) is in radians, not degrees.                                                                      |
| `<visual>`    | Viewer-only cosmetics — here, how big the little joint-axis arrows are drawn.                                                                                                                                       |
| `<option>`    | <br />Physics engine settings:`timestep` (simulation step size, seconds) and `gravity`.                                                                                                                          |
| `<asset>`     | Reusable resources, here just`<material>`s (color/shininess) referenced by name later.<br /><br />If you are familar with unity very much like the materials that you put onto game objects which in our case is  |
| `<default>`   | Fallback attributes so you don't repeat yourself. Any`<joint>` here gets `damping="0.5"`, any `<geom>` gets `material="body"`, unless overridden locally.                                                    |
| `<worldbody>` | The actual tree of bodies — this is the robot.                                                                                                                                                                      |
| `<contact>`   | Collision rules (see below).                                                                                                                                                                                         |
| `<actuator>`  | The motors that drive the joints.                                                                                                                                                                                    |

## Anatomy of one body

```xml
<body name="lower_arm" pos="-0.01 0 -0.1">
  <joint name="lower_arm_joint" type="hinge" axis="0 1 0" pos="0 0 0.04" limited="true" range="-1.047 1.047"/>
  <geom type="ellipsoid" size="0.02 0.015 0.04"/>
  ...children...
</body>
```

- **`<body>`** — a rigid link. `pos` is its offset from its *parent's* frame (not the world origin).
  So `lower_arm` sits 0.1m below and 0.01m behind wherever `upper_arm` ends up.
- **`<joint>`** — how this body is allowed to move relative to its parent. No joint = welded rigidly
  to the parent (that's why `upper_arm` never moves independently of `main_swivel`).
  - `type="hinge"` — rotates around an axis, like a door hinge. `axis="0 1 0"` = rotates about Y.
  - `type="slide"` — moves in a straight line instead of rotating (used once, for the eye).
  - `range="-1.047 1.047"` — motion limits in radians (≈ ±60°), enforced because `limited="true"`.
  - `pos = "0 0 0.04"`- shows you the point where the object pivots about. 0 0 0 represents the center of the object but in this case we are pivoting around the top of the lower arm
- **`<geom>`** — the visible/collidable shape for this body (ellipsoid, cylinder, box, plane, ...).
  Purely cosmetic/collision — it has no effect on the kinematics above.

## Geom types: what `size` and `euler` actually mean

The catch with `size`: for every type except `mesh`, the numbers are **half-extents/radii, not full
lengths**. A box with `size="0.1 0.1 0.1"` is a 0.2m cube, not a 0.1m cube.

| `type`      | `size` = (in order)                                                                     | Notes                                                                                                                                   |
| ------------- | ----------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| `plane`     | half-width X, half-width Y, spacing                                                       | Set X or Y to`0` for an infinite plane in that direction. `spacing` is just the visual grid line spacing — doesn't affect physics. |
| `sphere`    | radius                                                                                    | One number.                                                                                                                             |
| `capsule`   | radius, half-length                                                                       | A cylinder with rounded (hemisphere) caps.`half-length` is the straight part only, caps add on top of that.                           |
| `cylinder`  | radius, half-height                                                                       | Flat-ended, no rounded caps. E.g. the horn in`push.xml`: `size="0.05 0.01"` = 0.05m radius, 0.02m tall disc.                        |
| `box`       | half-x, half-y, half-z                                                                    | Half-extents along each local axis.                                                                                                     |
| `ellipsoid` | semi-x, semi-y, semi-z                                                                    | Like a box's half-extents but for a squashed sphere — each number is the radius along that axis.                                       |
| `mesh`      | (none — shape comes from the referenced mesh file, optionally with one uniform`scale`) |                                                                                                                                         |

`capsule` and `cylinder` can also skip `pos`/`euler` entirely and instead use
`fromto="x1 y1 z1 x2 y2 z2"` (two endpoints in the parent frame) — then `size` only needs the
radius, since the length is implied by the two points. Handy for rods/struts.

**`euler`** sets a geom's (or body's) rotation as three angles applied in X, then Y, then Z order
(intrinsic rotations), in whatever unit `<compiler angle="...">` declares — this file uses
`"radian"`, so `euler="0 0.3 0"` on `upper_arm` is a ~17° tilt about Y. `euler` is just one of
several ways to set orientation (`quat`, `axisangle`, `xyaxes`, `zaxis` are the others) — `euler` is
the most human-readable but the *least* precise for anything beyond quick static tilts, since it
compounds rounding across three rotations.

**Converting degrees ↔ radians** (needed any time you're editing `euler`/joint `range` in this
file, since `<compiler angle="radian">` means every angle here is in radians):
- degrees → radians: `degrees * π/180`. E.g. `45 * π/180 ≈ 0.785 rad`.
- radians → degrees: `radians * 180/π` (180/π ≈ 57.2958). E.g. `0.3 * 57.2958 ≈ 17.2°`.

| degrees | radians |
| ------- | ------- |
| 15°     | 0.262   |
| 30°     | 0.524   |
| 45°     | 0.785   |
| 60°     | 1.047   |
| 90°     | 1.571   |
| 180°    | 3.142   |

**To look things up yourself:**

- The canonical reference is the MJCF XML docs: https://mujoco.readthedocs.io/en/stable/XMLreference.html#body-geom
  — every attribute for every element, with the exact size semantics per geom type.
- The modeling guide (prose, more digestible) is at https://mujoco.readthedocs.io/en/stable/modeling.html
- Fastest hands-on way to learn a geom's shape: just load it and look —
  `mjpython demos/view.py model/push.xml` — then nudge the `size`/`euler` numbers and re-run to see
  what changed. The viewer's UI also has a "Model" panel that lists every geom with its live
  resolved size.

## Every joint in glados.xml

| Joint                 | Type  | Axis | Range   | What it does                                         |
| --------------------- | ----- | ---- | ------- | ---------------------------------------------------- |
| `main_swivel_joint` | hinge | Z    | ±180° | Whole arm swivels left/right, like a lazy-susan base |
| `lower_arm_joint`   | hinge | Y    | ±60°  | Elbow — pitches the lower arm up/down               |
| `neck_joint`        | hinge | X    | ±45°  | Rotates the head side-to-side (roll)                 |
| `head_joint`        | hinge | Y    | ±57°  | Nods the head up/down (pitch)                        |
| `eye_joint`         | slide | Z    | ±2mm   | Eye pokes in/out a couple millimeters                |

`upper_arm` has no joint — it's a fixed rigid extension of `main_swivel`, just bent at an angle
via its `euler="0 0.3 0"` (a static rotation baked into the body, not a moving joint).

## The site: `end_effector`

```xml
<site name="end_effector" pos="0 0 -0.05" size="0.0005" rgba="1 0 0 1"/>
```

A `<site>` is a massless, collision-free reference point you can query in code (its position/
orientation in world space) — it doesn't affect physics at all. This one marks the tip of the head;
the comment above it says it's "the reference for inverse kinematics" — i.e. when you write IK code
to aim the head/eye at a target, this is the point whose position you're solving for.

## `<contact><exclude>`: turning off self-collision

```xml
<exclude body1="lower_arm" body2="head"/>
```

Adjacent bodies overlap slightly at their joints by design (that's how joints look physically
connected). Without exclusions, MuJoCo would treat that overlap as a real collision and the physics
would fight itself. Each `<exclude>` just says "these two bodies are never allowed to collide with
each other" — one line per pair of touching links, plus one extra (`lower_arm`/`head`) for parts
that are close enough to clip even though they're not direct parent-child.

## `<actuator>`: how you actually move it

```xml
<position name="lower_arm_actuator" joint="lower_arm_joint" kp="200" ctrlrange="-1.047 1.047"/>
```

Joints on their own are just physical degrees of freedom — nothing drives them. An `<actuator>`
attaches a motor to a joint. `<position>` is a position-controlled actuator: you send it a target
angle (within `ctrlrange`), and it applies force proportional to `kp` (gain — higher = stiffer/
snappier, lower = softer/springier) to chase that target. There's one actuator per joint here, so
in code you control the robot by writing 5 numbers to `data.ctrl` — one per row in the table above,
in `main_swivel → lower_arm → neck → head → eye` order.

## Quick mental model to hold onto

- **Body** = a rigid piece of the robot.
- **Joint** = the one way that piece is allowed to move relative to its parent (or "welded" if none).
- **Geom** = what it looks like / collides as.
- **Site** = a labeled point you can read the position of, no physics.
- **Actuator** = the motor that pushes a joint toward a target.
- **Tree nesting** = the physical assembly order, parent to child.

To see it move: from `sim/`, with the venv active, `mjpython demos/view.py model/glados.xml`,
then drag the `ctrl` sliders in the viewer UI — each one maps directly to a row in the joint table above.
