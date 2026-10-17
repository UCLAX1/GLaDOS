# Actions

Motor sequences for GLaDOS. Each action is a single Python function.

## Writing an action

Create a new file in `actions/scripts/`. The filename and function name must match.

```python
# actions/scripts/wave.py

def wave(seq):
    return (seq
        .pose(upper_arm_yaw=45, duration=0.4)
        .pose(upper_arm_yaw=-45, duration=0.4)
        .pose(upper_arm_yaw=0, duration=0.3))
```

That's the whole file. No imports needed.

### Available joints

| Joint | What it does | Range |
|---|---|---|
| `upper_arm_yaw` | rotates the whole arm left/right | -90° to 90° |
| `upper_arm_pitch` | upper arm pitch | -30° to 30° |
| `lower_arm` | lower arm, bends the arm | 0° to 90° |
| `tilt` | tilts head side to side | -30° to 30° |
| `nod` | nods head forward/back | -30° to 30° |
| `eye` | extends/retracts the eye | -2mm to 2mm |

Ranges come from `sim/model/parameter.csv`; anything outside is clamped. For moving to an xyz point and the rest of the
control API, see [`control/README.md`](../control/README.md).

You can move any combination of joints in one `.pose()` — omitted joints stay where they are.

### `.pose()` options

```python
seq.pose(tilt=-20, nod=10, duration=0.5)
#         ↑ joints to move      ↑ seconds to complete the move
```

| Option | Default | What it does |
|---|---|---|
| `duration` | `0.5` | how long the move takes in seconds |
| `lerp` | `True` | smoothly interpolate to target (ease in/out). Set `False` to jump instantly |
| `additive` | `False` | move relative to current position. `nod=10` means +10° from wherever nod is now |

Examples:

```python
.pose(nod=30, duration=0.4)                   # smooth move to 30° (default)
.pose(nod=30, duration=0.4, lerp=False)       # snap to 30°, hold
.pose(nod=10, duration=0.4, additive=True)    # move +10° from current position
```

Other methods:

```python
.wait(duration=0.5)   # hold current position for duration seconds
.play()               # run the sequence once
.loop()               # repeat forever (until viewer closed or Ctrl-C)
.loop(3)              # repeat 3 times
```

## Running an action

From the `X1_GLaDOS/` root:

```bash
# Simulation
mjpython actions/run_action.py wave

# Hardware
GLADOS_HARDWARE=1 python3 actions/run_action.py wave   # needs control/hardware_control.py (not written yet)
```
