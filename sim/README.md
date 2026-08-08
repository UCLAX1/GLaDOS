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
