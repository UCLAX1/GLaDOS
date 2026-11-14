# PID torque control

Standalone Python PID control for the MuJoCo arm. The arm joints are driven as
DC motors in voltage mode, and the PID converts joint torque to terminal
voltage with back-EMF compensation.

| File | Purpose |
| --- | --- |
| [`pid.py`](pid.py) | Generic `PIDController` (P/I/D, anti-windup, output limits) |
| [`demo.py`](demo.py) | Torque-control demo for the full arm |
| [`test_pid.py`](test_pid.py) | Unit checks for the controller math (no MuJoCo needed) |
| [`dcmotor.pdf`](dcmotor.pdf) | MuJoCo DC motor model reference |

## Run

From `sim/`, with the venv active:

```bash
mjpython pid/demo.py              # hold a pose, open the viewer
mjpython pid/demo.py --track      # sine-wave tracking on selected joints
python3  pid/demo.py --headless   # no viewer; print mean tracking error
python3  pid/test_pid.py          # unit checks
```

On macOS the passive viewer needs `mjpython`; the headless and test runs use
plain `python3`. If you use conda instead of the project venv, prefix the
commands with `conda run -n <env> ...`.

Requires MuJoCo ≥ 3.3 (the `<dcmotor>` actuator). On MuJoCo 3.10+ the
`input` keyword is a single word — this demo uses `input="voltage"`.

## How it works

1. [`demo.py`](demo.py) loads the same arm as the rest of the sim via
   `params.load_model()`. The five arm joints are `<dcmotor>` actuators with
   `input="voltage"`; the eye remains a `<position>` actuator.
2. Each arm joint gets a [`PIDController`](pid.py). Every step the controller
   reads `data.qpos`, computes a joint torque `tau` (N·m), then converts it to
   a motor terminal voltage using the paper's back-EMF compensation
   ([`dcmotor.pdf`](dcmotor.pdf), Eq. 13):

   ```
   v = (R/K) * (tau / gear) + K * (gear * qvel)
   ```

   where `R` and `K` are read from `model.actuator_gainprm` at runtime, `gear`
   is the transmission ratio, and `qvel` is the joint velocity. The voltage is
   clamped to `V_SUPPLY` (±12 V, mirrored by `ctrlrange` in `glados.xml`) and
   written to `data.ctrl`.
3. MuJoCo applies the motor physics downstream of `v`: current/torque
   saturation, gear reflection, armature inertia, and back-EMF.
4. The eye stays position-controlled, so its target position is written
   straight to `data.ctrl` (no PID).

## Tuning

Gains and limits live at the top of [`demo.py`](demo.py):

- `GAINS` — `kp`/`ki`/`kd` per arm joint, in N·m per radian of joint error
- `TORQUE_LIMITS` — joint torque saturation (N·m), applied before the
  torque-to-voltage mapping
- `integral_limits()` — anti-windup bound derived from the torque range, so the
  integral term alone can never exceed half the available torque
- `POSE` / `TRACK` — hold targets and sine-tracking parameters

The shipped gains are conservative starting points that hold the pose stably
under gravity. Increase `kp` for stiffer tracking, raise `kd` if a joint
oscillates, and raise `ki` to remove steady-state error faster.

In `--track` mode the reported error is dominated by phase lag: the joints
carry `damping` in `glados.xml` (e.g. 80 on the arm pitch joints), which the
PID cannot remove without editing the XML, so a moving sine target is always
trailed slightly. The pose-hold mode converges to well under a degree.
