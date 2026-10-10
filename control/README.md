# Control System for Glados robot

Contains:
- `control_interface.py`: abstract class, the same API for sim and hardware (`move`, `run_action`, `get_position`)
- `mujoco_control.py`: the sim implementation (model, viewer, physics stepping)
- in the future a motor specific implementation

## How it works

```
sim = GladosSim()               # sim/runner.py: builds a MujocoControl (model + physics)
with sim.launch():              # opens the viewer
    sim.robot.move(...)         # set a goal
    sim.robot.run_action(...)   # play an emote
```

**`move()` goes there and returns once it has arrived.** It clamps to joint limits, writes the target into
`data.ctrl`, then ticks until every commanded joint is within 2 degrees of its target (or `timeout`, default 3 s,
or the window closes). A tick = `_tick()`: `mj_step` (the actuators pull the joints toward `ctrl`),
`viewer.sync()` (redraw), 2 ms sleep. On real hardware `_tick()` just sleeps, since the motors move on their own.

If you need to stream a new goal every tick (continuous tracking, `Sequence`, `chaos.py`, `ik_demo.py`) blocking on
each one would stall, so those use `_command()`: same arguments as `move()`, but it only sets the goal and returns,
and you tick yourself.

**`run_action(name)`** loads `actions/scripts/{name}.py`, builds a `Sequence` with the robot's `_tick`/`_running`,
and plays it. The sequence calls `_command()` + `_tick()` in a loop with smooth timing.

### `move()` units

| call | meaning |
|---|---|
| `move(main_swivel=45, lower_arm=30)` | degrees (default), eye is always mm |
| `move(main_swivel=0.8, unit="rad")` | radians |
| `move(position=(-0.3, 0.1, -0.55), unit="cart_pos")` | head_mount xyz in meters from the top of the upper arm (z negative = below); inverse kinematics picks `main_swivel`/`upper_arm`/`lower_arm`; `tilt`/`nod` still in degrees. Raises `ValueError` if unreachable |

## Examples

```python
from sim.runner import GladosSim

sim = GladosSim()
with sim.launch():
    sim.robot.move(main_swivel=45, lower_arm=30)                         # swings there, then returns
    sim.robot.move(position=(-0.3, 0.1, -0.55), unit="cart_pos")         # then to this point
    sim.robot.run_action("nod", loop=True)                               # or: mjpython actions/run_action.py nod
```

Stream goals yourself (`_command` sets the goal only, you tick):

```python
with sim.launch():
    sim.robot._command(main_swivel=45)
    while sim.robot._running():
        sim.robot._tick()
```

Headless (no window, e.g. tests): skip `launch()`; `robot.settle()` steps until it has arrived and
`robot.head_position()` reads where `head_mount` ended up.

Note: with gravity on, the position actuators settle a few mm to ~15 mm below the target (PD sag).
