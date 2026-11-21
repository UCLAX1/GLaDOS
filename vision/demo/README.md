# Vision demo (PDR)

All vision features run at once on one camera and show in one dashboard:

| key | feature | from |
|---|---|---|
| 1 | Person tracking (YOLO26n + ByteTrack, IDs) | `object_tracker.py` |
| 2 | Gestures for each arm (wave, raise, point, thumbs, peace) | `gestures_revised.py` |
| 3 | Face recognition (names from `known_faces/`) | `face_memory.py` |
| 4 | Expression (8 emotions) | `expression2.py` |
| 5 | Distance to each person in metres (ZED SDK only) | `depth/zed_depth.py` |

One process opens the camera. Each feature runs on its own thread at its own speed on the newest frame. Results are matched to tracked people, so each person gets a box with their ID, name and distance, plus their gestures and expression. If a feature fails to load it shows as *unavailable* and the others keep running.

## Setup (once per machine, needs internet)

```bash
cd GLaDOS                       # repo root, branch vision-demo
python3 vision/demo/setup_demo.py
```

Setup needs Python 3.9–3.12, because MediaPipe has no 3.13 wheels. On a Mac, run `brew install python@3.12`, then use `python3.12 vision/demo/setup_demo.py`.

Setup creates `vision/demo/.venv`, installs everything and downloads all models, so the demo itself works offline. On the Jetson it also installs NVIDIA's GPU build of onnxruntime and builds a TensorRT engine for tracking. That takes about 8 minutes, once, and is cached in `vision/demo/models/trt_cache/`. It finishes with a 30-second self-test that prints **GO** or **NOT READY**, with frame rate and time per frame for each feature.

* **insightface needs a C/C++ compiler.** On a Mac run `xcode-select --install`. On Windows install "Build Tools for Visual Studio". If it still fails, the demo runs without face recognition.
* **Faces:** setup doesn't copy registered faces. Copy `vision/known_faces/*.npy` from another machine, or register people live with `r`.

## Run

**One click:** on the Jetson, double-click **GLaDOS Vision Demo** on the desktop or in the app menu. It opens fullscreen. Install the icon once with `vision/demo/start_demo.sh --install-desktop`.

**One command:**

```bash
vision/demo/start_demo.sh                 # Linux / Mac; any extra flags go to run_demo.py
vision\demo\start_demo.bat               # Windows laptop
vision/demo/start_demo.sh --stop          # stop it (releases the camera)
```

`start_demo.sh` picks the flags for the machine it's on:
* ZED Mini plugged in: `--source zed --rotate 180`. The Jetson's ZED is mounted upside down; set `DEMO_ROTATE=0` once it's upright.
* No ZED: the laptop webcam.
* No screen (SSH session): `--no-window`, so use the browser view.
* A demo is already running: it refuses to start a second copy, since two copies fight over the camera.
* Only half the ZED is detected: it tells you to flip the cable.

The icon and launcher run whatever branch is checked out in `~/Documents/GLaDOS`. If the launcher is missing, switch back to `main`.

**By hand:**

```bash
vision/demo/.venv/bin/python vision/demo/run_demo.py            # Windows: vision\demo\.venv\Scripts\python
```

| option | what it does |
|---|---|
| `--source auto` | ZED SDK if present, otherwise webcam 0 (default) |
| `--source webcam:1` | another camera; a ZED without the SDK is cropped to its left lens |
| `--source zed` | ZED SDK: adds depth distance (Jetson) |
| `--source path/to/video.avi` | loops a recording instead of the camera (backup) |
| `--modules track,gesture,face` | run only these features |
| `--rotate 180` | camera mounted upside down (90 / 270 for sideways); depth turns with it |
| `--fullscreen` / `--no-window` | projector / browser only |
| `--web 8080` | browser view port, 0 = off |

**Keys:** `1`–`5` turn a feature on or off, `i` turns auto-idle on or off, `f` toggles fullscreen and `q` quits. To register a face, press `r`, type the name and press Enter (`x` cancels).

**Browser view:** go to `http://<machine-ip>:8080/` from any device on the same network. Tailscale's `http://100.73.92.3:8080/` works from anywhere on the tailnet; campus Wi-Fi often blocks device-to-device traffic. The terminal prints every address at startup. The page shows the same dashboard plus:
* a button for each feature, to turn it on or off
* **Auto-idle: on/off**, and the current mode and room state
* **Register** / **Cancel**, a ✕ on each known name to forget that face, and **Forget all**

**Registering faces:**
* Only the person being registered may be in view. Sampling pauses while 2+ faces are visible, so a bystander can't be recorded under someone else's name.
* Turn your head slowly until it reaches 15/15. That takes a few seconds when faces run on the GPU, and gives up after 90 s.
* When faces run on the CPU (laptop, or GPU unavailable), tracking, gestures and expression **pause while registering** so face recognition gets the CPU. They show as "paused" and resume when it finishes, is cancelled or times out.
* A face that's already registered under another name is refused, and the message says whose it is.
* Names: each face labels one person only, a name belongs to one person at a time, and two "unknown" results in a row clear a wrong name.
* Faces are saved to `vision/known_faces/<name>.npy`, which is gitignored.

## Auto-idle

When nobody is around, the heavy features pause by themselves (`presence.py`). There's no extra model: person tracking watches the room, plus a room check on a tiny grey copy of each frame.

| mode | when | what runs |
|---|---|---|
| **IDLE** | nobody seen for 10 s | tracking only, at 2 Hz. Any motion brings it to full speed at once |
| **AWARE** | someone in view, further than 3 m | tracking + depth |
| **ENGAGED** | someone within 3 m (no ZED: anyone in view), or a face is being registered | everything |

* Waking from idle took 0.5–0.6 s in tests, measured from the first frame with a person until everything was on.
* Going back down has margins, so someone standing near the boundary doesn't flip it on and off. Engaged → aware happens only after 5 s further than 4.2 m. Aware → idle happens only after 10 s with nobody in view.
* The room check reports motion (none/low/medium/high), light (ok/dark/bright) and a blocked camera. It's shown on the panel and in `/status` under `presence`, along with `person_detected`, `person_count` and `nearest_m`.
* It only *pauses* features. A feature you turned off by hand stays off, and face registration's own pause is left alone. If the controller ever stops, every pause lapses within 5 s.
* Turn it off with `i`, the browser button, `--no-idle` or `DEMO_IDLE=0`. Keep it off for a stage demo if an idle-looking dashboard would confuse people. The self-test (`--check`) always runs with it off.
* Tuning: `DEMO_IDLE_AFTER=10` (seconds), `DEMO_ENGAGE_M=3.0` (metres).

## Demo-day checklist

1. Run the self-test the day before on the machine you'll present from: `python3 vision/demo/setup_demo.py --check-only`. Sit in front of the camera while it runs.
2. Record a backup clip with `vision/gesture_eval/record_gestures.py` and test `--source <clip>`.
3. **Laptop (main):** plug HDMI into the projector, then `run_demo.py --fullscreen`.
4. **Jetson (backup, adds depth):** bring the HDMI monitor, keyboard and mouse. Run `run_demo.py --source zed --fullscreen`.
   * Close other GPU users first (LLM, ZED viewers), because only one program can open the ZED.
   * For a laptop view without the venue Wi-Fi, connect the Jetson and laptop to a phone hotspot and open the browser view.
5. If the frame rate gets low, turn off a feature with its number key. The others speed up right away.

## Measured speed

Each feature runs at its own rate. Jetson numbers were measured with everything running at once (live ZED, 1 person). Laptop numbers are estimates until the self-test runs there.

| | Jetson GPU (default) | Jetson CPU (fallback) | laptop (expected) |
|---|---|---|---|
| camera | 28 fps | 18–20 fps | 30 fps |
| tracking | 28 Hz, 20 ms (TensorRT fp16) | 5–7 Hz, ~160 ms (ONNX) | 15–30 Hz (torch, 640 px; MPS on a Mac) |
| gestures | 12 Hz (capped; MediaPipe stays on the CPU) | 9–10 Hz | 15–25 Hz |
| faces | 33 ms per pass, 1 per second + at once for a new person | ~1.4 s per pass | 1 Hz |
| expression | 6 Hz (capped), 11 ms | ~5 Hz, ~110 ms | 6 Hz (capped) |

**GPU on the Jetson.** Tracking runs YOLO26n as a TensorRT engine; face recognition and expression run on CUDA. All three go through `onnxruntime-gpu`, so torch stays the CPU build. Any part that can't use the GPU falls back to the CPU on its own, and the panel says which one each feature is using (`GPU TensorRT fp16`, `(GPU)`, `cpu`).
* No TensorRT engine yet: tracking uses CUDA (~35 ms) and the panel says so. Build it with `setup_demo.py`, not at demo start, because it takes ~8 minutes.
* The engine is specific to this GPU and these TensorRT/model versions. Re-exporting the model makes onnxruntime rebuild it, ~8 minutes at the next start, so run `setup_demo.py` after any model change.
* **GPU clock.** The Jetson's power manager keeps the GPU at ~400–500 MHz of its 1020 MHz because each model call is short. That made tracking 40–60 ms instead of 20 ms in later runs. `sudo jetson_clocks` pins full clocks until reboot. It's a system-wide setting on the shared Jetson, so agree on it first, and use it on demo day.
* To undo: `vision/demo/.venv/bin/pip uninstall -y onnxruntime-gpu && vision/demo/.venv/bin/pip install onnxruntime==1.23.2`.

On the CPU, each library's threads are capped (`workers.onnx_threads`, torch 2 threads). Without the caps the libraries fought over 6 cores and tracking fell to about 1 Hz.

Environment overrides: `DEMO_TRACK_BACKEND=auto|gpu|cpu`, `DEMO_ONNX_GPU=0` (faces/expression on the CPU), `GESTURE_COMPLEXITY=0|1`, `DEMO_YOLO_IMGSZ=320`, `DEMO_FACE_PERIOD=1.0`, `DEMO_EXPR_HZ=6`, `ZED_DEPTH_MODE=PERFORMANCE`, `DEMO_ROTATE=180` (start_demo.sh), `DEMO_FACES_DIR=<dir>` (use another folder of known faces, e.g. for tests).
