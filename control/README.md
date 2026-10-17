# Controlling GLaDOS

How to make the robot move, in the sim now and on hardware later. You use the same API for both.

```
sim/runner.py            GladosSim: opens the sim, gives you sim.robot
   │
control/control_interface.py   ControlInterface (abstract): move(), run_action(), ik(), limits, clamping
   ├── control/mujoco_control.py    MujocoControl: writes data.ctrl, steps physics, viewer
   └── (future) hardware_control.py HardwareControl: talks to the motors
   │
actions/sequence.py      Sequence: timed, smoothed chains of poses
actions/scripts/*.py     the emotes (nod, scan, idle, ...), one function each
actions/run_action.py    command-line runner for one emote
```

Setup is in [`sim/README.md`](../sim/README.md). On macOS anything that opens the viewer must run with **`mjpython`**, not `python3`.

## Quick start

```python
from sim.runner import GladosSim

sim = GladosSim()
with sim.launch():                                                 # opens the viewer
    sim.robot.move(upper_arm_yaw=45, lower_arm=30)                   # joint angles, degrees
    sim.robot.move(position=(-0.3, 0.1, -0.55), unit="cart_pos")   # a point in space (inverse kinematics)
    sim.robot.move(tilt=-15, nod=10)                               # head
    sim.robot.run_action("nod", loop=True)                         # play an emote until the window closes
```

From the command line: `mjpython actions/run_action.py nod` (loops) or `... nod --once`.

## Joints

Limits come from `sim/model/parameter.csv` (through `sim/params.py`). **Everything you pass is clamped to them**, so
a command outside the range silently stops at the edge.

| name | what it moves | XML joint | range |
|---|---|---|---|
| `upper_arm_yaw` | whole arm yaw, about vertical | `upper_arm_yaw_joint` | ±90° |
| `upper_arm_pitch` | upper arm pitch | `upper_arm_pitch_joint` | ±30° (placeholder) |
| `lower_arm` | lower arm | `lower_arm_joint` | 0..90° |
| `tilt` | head roll (side to side) | `tilt_joint` | ±30° |
| `nod` | head pitch | `nod_joint` | ±30° |
| `eye` | eye in/out | `eye_joint` | ±2 mm |

All joints at 0 = arm hanging straight down. The control names differ from the XML names only for the first two
(`XML_NAME` in `mujoco_control.py`). If a joint is missing from the model it's skipped with a printed warning and
reads back as 0, so actions using the other joints still run.

## `move()`: go there, then return

```python
robot.move(*, timeout=3.0, upper_arm_yaw=, upper_arm_pitch=, lower_arm=, tilt=, nod=, eye=, unit="deg", position=)
```

- Pass only the joints you want to move; the rest stay where they are.
- **Blocks** until every commanded joint is within `ARRIVE_TOL` (2°, or 0.2 mm for the eye) of its target,
  `timeout` seconds pass, or the viewer closes. It doesn't raise on timeout, it just returns.
- All joints in one call move **at the same time** (one `_send_batch`), not one after another.

| `unit` | meaning |
|---|---|
| `"deg"` (default) | rotation joints in degrees, eye in mm |
| `"rad"` | rotation joints in radians, eye still mm |
| `"cart_pos"` | `position=(x, y, z)` in **meters**: where `head_mount` should go. IK picks `upper_arm_yaw`/`upper_arm_pitch`/`lower_arm` (so don't pass those, it raises). `tilt`/`nod` still in degrees. **Raises `ValueError` if the point is unreachable** |

Cartesian frame: origin at the ceiling mount, z up, so targets have negative z. Reachable points sit 570–753 mm
from the upper arm, in a bowl below it. `(-0.3, 0.1, -0.55)` and `(-0.2, -0.15, -0.6)` are safe examples.
Math, workspace and tests: [`sim/F_I_K/arm/README.md`](../sim/F_I_K/arm/README.md).

You can also call the IK yourself without moving:

```python
from control.control_interface import ik
yaw, pitch, lower_arm = ik(-0.3, 0.1, -0.55)   # radians; ValueError if unreachable
```

## `_command()` + `_tick()`: stream goals yourself

`move()` waits on every call, which stalls anything that updates the goal every tick (tracking, smooth motion).
For that use `_command()`: **same arguments as `move()`**, but it only sets the goal and returns. You step time
yourself with `_tick()`, and stop when `_running()` goes false (window closed).

```python
with sim.launch():
    t0 = time.time()
    while sim.robot._running():
        sim.robot._command(position=circle(time.time() - t0), unit="cart_pos")
        sim.robot._tick()
```

That's `sim/demos/ik_demo.py`. `Sequence` and `chaos.py` work the same way.

One tick (`TICK_RATE = 0.002 s`, in `actions/sequence.py`):

- **sim** (`MujocoControl._tick`): one `mj_step` (the position actuators pull toward `data.ctrl`), `viewer.sync()`, sleep 2 ms
- **hardware** (`ControlInterface._tick` default): just sleep, the motors move on their own

## Emotes: `run_action()` and `Sequence`

`robot.run_action("nod")` imports `actions/scripts/nod.py`, calls `nod(seq)` with a `Sequence` bound to this
robot's `_tick`/`_running`, and plays it (`loop=True` repeats until stopped).

An action is one function that chains `.pose()` / `.wait()` calls:

```python
# actions/scripts/wave.py  (filename == function name, no imports)
def wave(seq):
    return (seq
        .pose(upper_arm_yaw=45,  duration=0.4)
        .pose(upper_arm_yaw=-45, duration=0.4)
        .pose(upper_arm_yaw=0,   duration=0.3))
```

`.pose()` options: `duration` (s, default 0.5), `lerp` (smoothstep ease in/out, default on; off = jump and hold),
`additive` (relative to the current position). Joint values are degrees / mm. Full reference:
[`actions/README.md`](../actions/README.md).

Existing actions: `nod`, `scan`, `idle`, `look_away`, `confused_scan`, `curious_peek_horiz`, `curious_peek_vert`,
`eye_blink`, `eye_extend`, `eye_retract`.

Sequences and `move()` combine fine: `sim/demos/move_and_nod.py` does `move(position=..., unit="cart_pos")`,
then `run_action("nod")`, over and over.

## Other things on the robot

| call | what |
|---|---|
| `robot.get_position(joint)` | current angle in degrees (eye mm), read from the sim/encoders |
| `robot.shutdown()` | stop the motors (no-op in sim) |
| `robot.settle(steps=4000)` | **sim only**: step physics with no viewer until the actuators arrive. For tests |
| `robot.head_position()` | **sim only**: `head_mount` xyz (m) as the sim actually has it |
| `robot.model`, `robot.data` | **sim only**: the raw MuJoCo objects |

Headless (tests, no window): build `MujocoControl()` directly, skip `launch()`, and use `_command` + `settle()`.

Note: with gravity on, the position actuators settle a little below the target (PD sag), 0–15 mm at the head
mount depending on pose. That's the sim's controller, not an IK error. With gravity off the IK lands within 0.1 mm.

## Entry points

| run | what it does |
|---|---|
| `mjpython actions/run_action.py <name> [--once]` | play one emote in the sim. `GLADOS_HARDWARE=1 python3 ...` would use `HardwareControl` (not written yet) |
| `mjpython sim/demos/move_and_nod.py` | IK to two points, nod at each |
| `mjpython sim/demos/ik_demo.py` | head_mount traces a circle using `cart_pos` |
| `mjpython sim/demos/chaos.py` | random motion |
| `mjpython sim/demos/view.py [glados.xml]` | just open a model (default `push.xml`), run from `sim/` |
| `mjpython sim/sim_daemon.py` | keeps a viewer open for the full pipeline; `run_glados.py` drops gesture JSON files in `/tmp/glados_sim_queue/` and the daemon plays them |
| `python3 gui_launch.py` | full pipeline + sim in one window (see the root README) |

## Writing the hardware backend

Subclass `ControlInterface` and implement three methods. Everything else (`move`, `cart_pos`/IK, clamping,
`run_action`, sequences) works unchanged.

```python
class HardwareControl(ControlInterface):
    def _send_batch(self, joints, unit="deg"): ...   # {name: target}; rotation in `unit`, eye mm. Send all at once
    def get_position(self, joint): ...               # degrees (eye mm); ValueError on an unknown name
    def shutdown(self): ...                          # stop the motors safely
```

The default `_tick` (sleep) and `_running` (always true) are right for hardware. Put it in
`control/hardware_control.py`, since `run_action.py` already imports it from there.

## Model loading (gotcha)

`glados.xml` and `push.xml` are now **templates** (`${L1}`, `${swivel}`, ...) filled from `parameter.csv`.
Load them with `params.load_model("glados.xml")`, never `mujoco.MjModel.from_xml_path`, or the parse fails.
`MujocoControl()` and `GladosSim()` already do this. Still using `from_xml_path`, so currently broken:

- `sim/sim_daemon.py`
- `GladosSim(xml_path=...)` when you pass a path
- `python3 -m mujoco.viewer --mjcf=model/glados.xml` in `sim/README.md` (use `demos/view.py glados.xml`)

## Tests

```bash
sim/venv/bin/python control/test_mujoco_control.py   # move/_command/deg vs rad/get_position
sim/venv/bin/python sim/F_I_K/arm/test_ik.py         # fk -> ik -> move(cart_pos) -> sim
```
