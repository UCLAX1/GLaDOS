# Actions

Motor sequences for GLaDOS. Each action is a single Python function.

## Writing an action

Create a new file in `actions/scripts/`. The filename and function name must match.

```python
# actions/scripts/wave.py

def wave(seq):
    return (seq
        .pose(main_swivel=45, duration=0.4)
        .pose(main_swivel=-45, duration=0.4)
        .pose(main_swivel=0, duration=0.3))
```

That's the whole file. No imports needed.

### Available joints

| Joint | What it does | Range |
|---|---|---|
| `main_swivel` | rotates the whole arm left/right | -180° to 180° |
| `lower_arm` | tilts arm up/down | 0° to 86° |
| `tilt` | tilts head side to side | -17° to 17° |
| `nod` | nods head forward/back | -34° to 23° |
| `eye` | extends/retracts the eye | -2mm to 2mm |

You can move any combination of joints in one `.pose()` — omitted joints stay where they are.

### `.pose()` options

```python
seq.pose(neck=-20, head=10, duration=0.5)
#         ↑ joints to move      ↑ seconds to complete the move
```

| Option | Default | What it does |
|---|---|---|
| `duration` | `0.5` | how long the move takes in seconds |
| `lerp` | `True` | smoothly interpolate to target (ease in/out). Set `False` to jump instantly |
| `additive` | `False` | move relative to current position. `head=10` means +10° from wherever head is now |

Examples:

```python
.pose(head=30, duration=0.4)                   # smooth move to 30° (default)
.pose(head=30, duration=0.4, lerp=False)       # snap to 30°, hold
.pose(head=10, duration=0.4, additive=True)    # move +10° from current position
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
GLADOS_HARDWARE=1 python3 actions/run.py wave
```
