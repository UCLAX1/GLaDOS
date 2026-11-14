# Sim

MuJoCo simulation of the GLaDOS arm. The XML model is in `model/glados.xml`.

## Setup

```bash
cd sim
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## View the model

```bash
python3 -m mujoco.viewer --mjcf=model/glados.xml
```

Run this from inside the `sim/` folder with the venv active.

## Run demos

Scripts in `demos/` must be run with `mjpython` on macOS (idk weird thing it does):

```bash
mjpython demos/chaos.py # hehe funny thingy
```

## PID torque control (experimental)

A standalone Python PID loop drives the arm's DC-motor actuators: each joint's
PID outputs torque, converted to motor voltage `v = (R/K)*(tau/gear) + K*gear*qvel`
and written to `data.ctrl`. See [`pid/README.md`](pid/README.md).

```bash
mjpython pid/demo.py             # hold a pose, open the viewer
mjpython pid/demo.py --track     # sine-wave tracking
python3  pid/demo.py --headless  # no viewer; print mean tracking error
```

## Static holding-torque formula validation

`demos/torque_test.py` checks the 2-link static holding-torque formula against
MuJoCo's ground-truth holding torque (`data.qfrc_bias` at zero velocity), and can
also apply the formula to an infinite-torque motor model to watch it hold a pose.

The formula puts the head force `F` at `head_mount` (point A); the demo also
reports a head-COM-corrected variant (`F` at the actual head COM, 0.10 m below A).

```bash
mjpython demos/torque_test.py                      # full alpha x beta grid
mjpython demos/torque_test.py --joint lower_arm    # isolate tau_B: sweep beta
mjpython demos/torque_test.py --joint upper_arm    # isolate tau_C: sweep alpha
mjpython demos/torque_test.py --hold               # apply formula torques, report drift
mjpython demos/torque_test.py --view --pose 0 40   # watch the arm hold a pose
mjpython demos/torque_visualize.py --pose 0 40     # live viewer + overlay; T toggles original vs corrected
```

On macOS the viewer modes need `mjpython`; the headless validation also runs with
plain `python3` if MuJoCo is installed.

## Trouble opening the sim?

If `python3 -m mujoco.viewer` isn't working, use `demos/view.py` instead:

```bash
mjpython demos/view.py glados.xml
```

Must be `mjpython`, not `python3` — the passive viewer requires it on macOS.
