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

To see it move: from `sim/`, with the venv active, `python3 -m mujoco.viewer --mjcf=model/glados.xml`,
then drag the `ctrl` sliders in the viewer UI — each one maps directly to a row in the joint table above.
